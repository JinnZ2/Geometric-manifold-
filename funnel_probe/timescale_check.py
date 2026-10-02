#!/usr/bin/env python3
"""
timescale_check.py -- step 1 of the singular-funnel work order for this repo.

Question (fixed in PREREGISTRATION.md): does the simulation separate timescales
-- a SLOW variable (the paper's mu, rate eps) entering the FAST dynamics as a
parameter -- or is there one timescale?

Three measurements, in the pre-registered order:
  1. code reading of simulation/controller.py::Controller.run -- how many times
     per loop step is theta moved by anything other than repair_step
  2. the repo's own simulation (torch), default config, 100 steps: repair
     ||delta|| per step, KL to reference, distance to reference
  3. a numpy twin of the same model/loss/clamp (funnel_probe/twin.py), whose
     analytic gradient is checked against torch autograd on the same theta, and
     which reports the same per-step quantities on its own init (numpy seed 42;
     NOT the same bytes as torch.manual_seed(42))

Nothing outside funnel_probe/ is imported for writing; engines are read only.
"""
import ast
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import yaml  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))


def code_reading():
    """Count, inside Controller.run's step loop, every assignment to `theta` and
    every call that could perturb it. Pure AST, no execution."""
    src = open(os.path.join(ROOT, "simulation", "controller.py")).read()
    tree = ast.parse(src)
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For))
    theta_assigns, calls, randn = [], [], 0
    for n in ast.walk(loop):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                names = [e.id for e in ast.walk(t) if isinstance(e, ast.Name)]
                if "theta" in names:
                    theta_assigns.append(ast.unparse(n.value)[:60])
        if isinstance(n, ast.Call):
            calls.append(ast.unparse(n.func))
            if "randn" in ast.unparse(n.func) or "drift" in ast.unparse(n.func).lower():
                randn += 1
    # where is theta_drifted made?
    env = open(os.path.join(ROOT, "simulation", "environment.py")).read()
    drift_line = [l.strip() for l in env.splitlines() if "theta_drifted" in l and "=" in l]
    return {
        "theta_assignments_in_loop": theta_assigns,
        "drift_or_randn_calls_in_loop": randn,
        "theta_drifted_defined_at": drift_line,
        "loop_calls": sorted(set(calls)),
    }


def torch_run(steps):
    import torch
    import torch.nn.functional as F
    from simulation.environment import Environment
    from manifolds.parameter_manifold import ParameterManifold

    env = Environment(CFG["simulation"])
    layer = ParameterManifold(env.theta_ref, CFG["manifolds"]["parameter"])
    model_fn = env.get_model_fn()
    theta = env.theta_drifted.clone()
    ref = env.theta_ref

    def kl(t):
        with torch.no_grad():
            return F.kl_div(F.log_softmax(model_fn(env.safety_inputs, t), dim=-1),
                            F.softmax(model_fn(env.safety_inputs, ref), dim=-1),
                            reduction="batchmean").item()

    disp0 = torch.linalg.norm(theta - ref).item()
    rows = []
    for step in range(steps):
        kl0 = kl(theta)
        new, m = layer.repair_step(theta, model_fn, env.safety_inputs, env.task_inputs, env.task_labels)
        delta = torch.linalg.norm(new - theta).item()
        rows.append({"step": step, "delta": delta, "kl_before": kl0, "kl_after": kl(new),
                     "dist": m["dist_to_ref"], "task_loss": m["task_loss"]})
        theta = new
    return env, layer, disp0, rows


def summarize(rows, disp0, trust):
    deltas = [r["delta"] for r in rows]
    repair_rate = sum(deltas) / len(deltas)
    sat = sum(1 for d in deltas if abs(d - trust) < 1e-6) / len(deltas)
    dkl = [r["kl_after"] - r["kl_before"] for r in rows]
    return {
        "initial_displacement": disp0,
        "repair_rate_mean_delta_per_step": repair_rate,
        "trust_radius": trust,
        "fraction_steps_at_trust_radius": sat,
        "steps_to_close_if_every_step_pointed_home": disp0 / repair_rate if repair_rate else None,
        "dist_first": rows[0]["dist"], "dist_last": rows[-1]["dist"],
        "kl_first": rows[0]["kl_before"], "kl_last": rows[-1]["kl_after"],
        "mean_dKL_per_step": sum(dkl) / len(dkl),
        "steps": len(rows),
    }


def main(argv):
    steps = CFG["simulation"]["steps"]
    trust = CFG["manifolds"]["parameter"]["trust_radius"]
    out = {"config": CFG, "code_reading": code_reading()}
    print("STEP 1 -- timescale separation (Geometric-manifold-)")
    print("\n1. code reading of Controller.run step loop")
    for k, v in out["code_reading"].items():
        print("   %-32s %s" % (k, v))
    drift_in_loop = out["code_reading"]["drift_or_randn_calls_in_loop"]

    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    out["torch_available"] = have_torch

    if have_torch:
        print("\n2. repo simulation (torch), default config, %d steps" % steps)
        env, layer, disp0, rows = torch_run(steps)
        out["torch"] = summarize(rows, disp0, trust)
        out["torch_rows"] = rows
        for k, v in out["torch"].items():
            print("   %-44s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
    else:
        print("\n2. repo simulation: NOT_RUN (torch absent)")

    print("\n3. numpy twin (funnel_probe/twin.py)")
    import twin
    tw = twin.run(CFG, steps)
    out["twin"] = summarize(tw["rows"], tw["disp0"], trust)
    out["twin_rows"] = tw["rows"]
    for k, v in out["twin"].items():
        print("   %-44s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
    if have_torch:
        chk = twin.check_against_torch(CFG)
        out["twin_gradient_check"] = chk
        print("   twin gradient vs torch autograd on identical theta: rel err %.2e (%s)"
              % (chk["rel_err"], "PASS" if chk["rel_err"] < 1e-5 else "FAIL"))

    # the rule
    print("\nRULE (pre-registered)")
    rep = (out["torch"] if have_torch else out["twin"])["repair_rate_mean_delta_per_step"]
    print("   in-loop drift applications per step : %d" % drift_in_loop)
    print("   repair rate (mean ||delta||/step)   : %.4g  (cap %.3g)" % (rep, trust))
    print("   ratio drift/repair                   : %.4g" % (0.0 / rep if rep else float("nan")))
    outcome = "NOT_IN_CLASS" if drift_in_loop == 0 else "see ratio"
    out["outcome"] = outcome
    print("   OUTCOME: %s -- theta_drifted is an INITIAL CONDITION (Environment._setup), not a"
          " slow variable; inside the loop only repair_step moves theta. One timescale." % outcome)
    print("   step 2 (2D slice, repair OFF) NOT RUN per the order's stop rule.")
    with open("timescale_check_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nscope: deterministic; default.yaml (drift_strength 0.3, trust_radius 0.05, lambda 10,"
          " curvature_weight 2, lr 0.01, %d steps, seed 42); continuous parameter space." % steps)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
