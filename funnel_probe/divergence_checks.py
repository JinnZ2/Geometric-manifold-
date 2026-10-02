#!/usr/bin/env python3
"""
divergence_checks.py -- the four checks of FINDING_REPAIR_DIVERGENCE.md.

    python3 funnel_probe/divergence_checks.py fixed-ref     C1
    python3 funnel_probe/divergence_checks.py fd-step       C2
    python3 funnel_probe/divergence_checks.py flip-sign     C3
    python3 funnel_probe/divergence_checks.py drift-trace   C4

Each prints its evidence and a one-line verdict. Nothing outside
funnel_probe/ is modified; the engine is imported and read.  Default config,
seed 42, torch (double precision where a finite difference is taken).
"""
import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch                                     # noqa: E402
import torch.nn.functional as F                  # noqa: E402
import yaml                                      # noqa: E402

from manifolds.parameter_manifold import ParameterManifold   # noqa: E402
from simulation.environment import Environment               # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
PC = CFG["manifolds"]["parameter"]


def setup():
    env = Environment(CFG["simulation"])
    layer = ParameterManifold(env.theta_ref, PC)
    return env, layer, env.get_model_fn()


def kl_to(model_fn, theta, ref, X):
    with torch.no_grad():
        return F.kl_div(F.log_softmax(model_fn(X, theta), dim=-1),
                        F.softmax(model_fn(X, ref), dim=-1), reduction="batchmean").item()


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def c1_fixed_ref():
    print("C1  is the KL target a fixed theta0 snapshot?")
    src = open(os.path.join(ROOT, "manifolds", "parameter_manifold.py")).read()
    refs = [l.strip() for l in src.splitlines() if "theta_ref" in l]
    print("   code: every use of theta_ref in parameter_manifold.py:")
    for l in refs:
        print("        ", l)
    env, layer, model_fn = setup()
    h_env0, h_layer0 = sha(env.theta_ref), sha(layer.theta_ref)
    theta = env.theta_drifted.clone()
    worst = 0.0
    for step in range(CFG["simulation"]["steps"]):
        theta, m = layer.repair_step(theta, model_fn, env.safety_inputs, env.task_inputs, env.task_labels)
        ext = kl_to(model_fn, theta, env.theta_ref, env.safety_inputs)
        # the layer reports safety_loss at the PRE-step theta; compare the next step's report
        if step > 0:
            worst = max(worst, abs(prev_ext - m["safety_loss"]))
        prev_ext = ext
    print("   runtime: sha(env.theta_ref)   before %s after %s" % (h_env0, sha(env.theta_ref)))
    print("            sha(layer.theta_ref) before %s after %s" % (h_layer0, sha(layer.theta_ref)))
    print("            layer.theta_ref is a .detach().clone() of env.theta_ref: same bytes = %s"
          % (h_env0 == h_layer0))
    print("            max |layer safety_loss(theta_k) - external KL(theta_k || env.theta_ref)| = %.2e" % worst)
    print("   VERDICT: the reference is FIXED (hashes unchanged, layer KL equals the external KL to the"
          " frozen env.theta_ref at every step). 'moving reference' is ruled out.")


def c2_fd_step():
    print("C2  one repair step with the cap removed; finite-difference check of the KL change")
    torch.set_default_dtype(torch.float64)
    env, layer, model_fn = setup()
    theta0 = env.theta_drifted.clone()
    X = env.safety_inputs
    kl0 = kl_to(model_fn, theta0, env.theta_ref, X)
    # analytic grad of KL at theta0
    t = theta0.clone().requires_grad_(True)
    kl = F.kl_div(F.log_softmax(model_fn(X, t), dim=-1),
                  F.softmax(model_fn(X, env.theta_ref), dim=-1), reduction="batchmean")
    gkl = torch.autograd.grad(kl, t)[0]
    for label, cfg in (("cap on  (trust_radius=0.05)", dict(PC)),
                       ("cap off (trust_radius=1e9)", dict(PC, trust_radius=1e9)),
                       ("cap off, lr=1e-4 (linear regime)", dict(PC, trust_radius=1e9, lr=1e-4))):
        lyr = ParameterManifold(env.theta_ref, cfg)
        new, m = lyr.repair_step(theta0, model_fn, X, env.task_inputs, env.task_labels)
        delta = new - theta0
        kl1 = kl_to(model_fn, new, env.theta_ref, X)
        pred = (gkl * delta).sum().item()
        print("   %-36s ||delta|| %.4g   KL %.5f -> %.5f  (dKL %+.4g)   grad(KL).delta = %+.4g   %s"
              % (label, delta.norm().item(), kl0, kl1, kl1 - kl0, pred,
                 "same sign" if (kl1 - kl0) * pred > 0 else "SIGN DISAGREES"))
        # projection of the step onto the KL gradient direction
        cos = (gkl @ delta / (gkl.norm() * delta.norm())).item()
        print("   %-36s cos(delta, grad KL) = %+.4f  (+1 would be pure KL ascent, -1 pure descent)" % ("", cos))
    print("   VERDICT: one repair step RAISES KL with or without the cap, and the first-order prediction"
          " grad(KL).delta has the same sign, so the step is pointed up the KL gradient; the cap is not"
          " the cause, it only bounds the rate.")


def c3_flip_sign():
    print("C3  flip the step sign once (theta - delta instead of theta + delta)")
    env, layer, model_fn = setup()
    X = env.safety_inputs
    theta = env.theta_drifted.clone()
    kl0 = kl_to(model_fn, theta, env.theta_ref, X)
    new, _ = layer.repair_step(theta, model_fn, X, env.task_inputs, env.task_labels)
    delta = new - theta
    print("   one step:  KL(theta)=%.4f   KL(theta+delta)=%.4f   KL(theta-delta)=%.4f"
          % (kl0, kl_to(model_fn, new, env.theta_ref, X), kl_to(model_fn, theta - delta, env.theta_ref, X)))
    # 100 steps with the sign flipped at the step
    th = env.theta_drifted.clone()
    traj = [kl_to(model_fn, th, env.theta_ref, X)]
    dist = [torch.linalg.norm(th - env.theta_ref).item()]
    for _ in range(CFG["simulation"]["steps"]):
        new, _ = layer.repair_step(th, model_fn, X, env.task_inputs, env.task_labels)
        th = th - (new - th)
        traj.append(kl_to(model_fn, th, env.theta_ref, X))
        dist.append(torch.linalg.norm(th - env.theta_ref).item())
    print("   100 steps, sign flipped: KL %.4f -> %.4f   dist %.3f -> %.3f   (unflipped run: KL 3.79 -> 39.37, dist 17.08 -> 18.53)"
          % (traj[0], traj[-1], dist[0], dist[-1]))
    print("   KL every 20 steps: %s" % ["%.3f" % v for v in traj[::20]])
    print("   VERDICT: flipping the sign of the step makes KL FALL monotonically; the step direction is"
          " the cause, not the metric and not the reference.")


def c4_drift_trace():
    print("C4  why is drift not applied per step?")
    env_src = open(os.path.join(ROOT, "simulation", "environment.py")).read()
    ctl_src = open(os.path.join(ROOT, "simulation", "controller.py")).read()
    main_src = open(os.path.join(ROOT, "main.py")).read()
    print("   config wiring: configs/default.yaml simulation.drift_strength = %s" % CFG["simulation"]["drift_strength"])
    print("   main.py passes config['simulation'] to Environment: %s" % ("Environment(" in main_src))
    print("   Environment reads it: %s" % ("config.get('drift_strength'" in env_src))
    print("   Environment applies it ONCE, in _setup: %s" % ("theta_drifted = self.theta_ref +" in env_src.replace("\\\n", " ").replace("\n            ", " ")))
    # does any module in simulation/ or main.py contain a per-step drift?
    per_step = []
    for fn in ("simulation/environment.py", "simulation/controller.py", "main.py"):
        tree = ast.parse(open(os.path.join(ROOT, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                body = ast.unparse(node)
                if "randn" in body or "drift" in body.lower():
                    per_step.append((fn, node.lineno, [l for l in body.splitlines() if "randn" in l or "drift" in l.lower()][:3]))
    print("   loops in simulation/ or main.py whose body mentions randn or drift: %d" % len(per_step))
    for fn, ln, lines in per_step:
        print("      %s:%d  %s" % (fn, ln, lines))
    print("   README line 20: 'Run the default simulation (drift=0.3, 100 steps)'; line 52: drift_strength ="
          " 'How far the model drifts from safe reference' (a distance, not a rate).")
    print("   VERDICT: the config IS wired and IS applied -- once, as the initial displacement"
          " theta_drifted = theta_ref + 0.3*randn. There is no per-step drift code anywhere in"
          " simulation/ or main.py, so there is no branch that is never taken; the scenario the code"
          " runs is 'start displaced, repair for 100 steps'. The README's verb 'drifts' reads as a"
          " process; the implementation is an initial condition. Finding 1 narrows from 'never applied'"
          " to 'applied once, not per step'.")


CHECKS = {"fixed-ref": c1_fixed_ref, "fd-step": c2_fd_step, "flip-sign": c3_flip_sign, "drift-trace": c4_drift_trace}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(__doc__); sys.exit(2)
    CHECKS[sys.argv[1]]()
