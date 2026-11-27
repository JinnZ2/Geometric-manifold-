#!/usr/bin/env python3
"""EXP-LLG-03 run (2026-11-27): coordinate dependence of kappa_eff under the exact ReLU
rescaling (W1, b1, W2, b2) -> (a W1, a b1, W2 / a, b2) of ToyLLM.

Arm 1 (pre-registered): 20 seeds of (theta_ref, drift direction, inputs) from
experiments/sketches/llg_rescaling.py::setup, alpha in {0.25, 0.5, 1, 2, 4}. Records max
logit difference, kappa_eff (repo function, abs) and the signed Rayleigh quotient.

Arm 2 (pre-registered): the sims/kappa_eff_leading scenario (config.json unchanged: 5 seeds,
sigma in {0.012, 0.016, 0.02}, isotropic + adversarial, 60 steps, null arm with no drift).
Each trajectory is run ONCE in the original coordinates with the repo's
CoupledDynamicalSystem; along it, kappa_eff and theta_dist are re-evaluated in rescaled
coordinates (theta, theta_ref, theta_dot all pushed forward by R_alpha) for alpha in
{0.5, 1, 2}. The function, and therefore basin_kl and the breach step, are identical across
alpha. The criterion sweep and grading rule are those of sims/kappa_eff_leading/run.py,
re-implemented here with its own alarm_step / smooth_causal_median imported read-only.
alpha = 1 is the positive control: it must reproduce the repo kappa_eff series exactly.

Output: raw.json. No repo file is modified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, add_repo_to_path, environment, write_json  # noqa: E402

add_repo_to_path()
from addon_thermodynamic_control.stability import CoupledDynamicalSystem, SpectralCertificate  # noqa: E402
from simulation.environment import Environment, model_fn  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sk = _load("llg_rescaling_sketch", REPO / "experiments/sketches/llg_rescaling.py")
ip18 = _load("kappa_eff_leading_run", REPO / "sims/kappa_eff_leading/run.py")
CERT = SpectralCertificate()


def signed_rayleigh(theta, theta_ref, x, v):
    """Same computation as effective_repair_curvature, without abs()."""
    nsq = (v ** 2).sum().item()
    t = theta.detach().requires_grad_(True)
    curr = F.log_softmax(model_fn(x, t), dim=-1)
    ref = F.softmax(model_fn(x, theta_ref).detach(), dim=-1)
    loss = F.kl_div(curr, ref, reduction="batchmean")
    g = torch.autograd.grad(loss, t, create_graph=True)[0]
    u = v.detach() / nsq ** 0.5
    hv = torch.autograd.grad(g, t, grad_outputs=u)[0].detach()
    return float((u * hv).sum())


def arm1(seeds, alphas):
    rows = []
    for s in seeds:
        theta_ref, theta, v, x = sk.setup(s)
        base = model_fn(x, theta)
        for a in alphas:
            th, rf, va = sk.rescale(theta, a), sk.rescale(theta_ref, a), sk.rescale(v, a)
            rows.append({
                "seed": s, "alpha": a,
                "max_abs_logit_diff": float((model_fn(x, th) - base).abs().max()),
                "kl_to_ref": float(ip18._safety_kl(model_fn, th, rf, x)),
                "kappa_eff": CERT.effective_repair_curvature(model_fn, th, rf, x, va),
                "kappa_signed": signed_rayleigh(th, rf, x, va),
            })
    return rows


def trajectory(seed, sigma, mode, cfg, alphas):
    """Copy of ip18.run_trajectory that also records rescaled-coordinate signals."""
    sc = cfg["scenario"]
    env = Environment({"drift_strength": sc["initial_drift_strength"], "seed": seed})
    mf = env.get_model_fn()
    system = CoupledDynamicalSystem(mf, env.theta_ref, env.task_inputs, cfg["thermo"])
    theta = env.theta_drifted.clone()
    gen = torch.Generator().manual_seed(10_000 + seed)
    ser = {"basin_kl": [], "kappa_repo": []}
    for a in alphas:
        ser[f"kappa_a{a}"], ser[f"theta_dist_a{a}"] = [], []
    for _ in range(sc["steps"]):
        if sigma > 0:
            if mode == "adversarial":
                d = ip18.adversarial_direction(mf, theta, env.theta_ref, env.safety_inputs,
                                               sc["adversarial_power_iters"])
            else:
                d = torch.randn(theta.shape, generator=gen)
            theta = theta + sigma * d
        theta_in = theta.detach().clone()
        theta, st = system.step(theta, env.safety_inputs, env.task_inputs, env.task_labels)
        tdot = theta.detach() - theta_in
        ser["basin_kl"].append(st.basin_kl)
        ser["kappa_repo"].append(st.kappa_eff)
        for a in alphas:
            th, rf, va = (sk.rescale(theta, a), sk.rescale(env.theta_ref, a),
                          sk.rescale(tdot, a))
            ser[f"kappa_a{a}"].append(CERT.effective_repair_curvature(
                mf, th, rf, env.safety_inputs, va))
            ser[f"theta_dist_a{a}"].append(float((th - rf).norm()))
    eps = cfg["thermo"]["epsilon_basin"]
    breach = next((i for i, kl in enumerate(ser["basin_kl"]) if kl > eps), None)
    return {"seed": seed, "sigma": sigma, "mode": mode, "series": ser, "breach_step": breach}


def grade(drift, null, cfg, alpha):
    """sims/kappa_eff_leading/run.py grading rule, signals in alpha-coordinates."""
    warm = cfg["scenario"]["warmup"]
    max_fp, gate = cfg["grading"]["max_null_false_alarm_rate"], cfg["grading"]["cell_fraction_gate"]
    ks, ts = f"kappa_a{alpha}", f"theta_dist_a{alpha}"
    usable = [r for r in drift if r["breach_step"] is not None and r["breach_step"] > warm]
    out = []
    for win in cfg["sweeps"]["smoothing_window"]:
        for crit in cfg["criteria"]:
            def sm(x):
                return ip18.smooth_causal_median(x, win)
            fpk = sum(ip18.alarm_step(sm(r["series"][ks]), crit, warm) is not None
                      for r in null) / len(null)
            fpt = sum(ip18.alarm_step(sm(r["series"][ts]), crit, warm) is not None
                      for r in null) / len(null)
            viable = fpk <= max_fp and fpt <= max_fp
            nA = nlead = 0
            alarms = []
            for r in usable:
                b = r["breach_step"]
                ak = ip18.alarm_step(sm(r["series"][ks]), crit, warm)
                at = ip18.alarm_step(sm(r["series"][ts]), crit, warm)
                lk = None if ak is None else b - ak
                lt = None if at is None else b - at
                leads = lk is not None and lk > 0
                beats = leads and (lt is None or lk > lt)
                nlead += leads
                nA += leads and beats
                alarms.append(ak)
            n = len(usable)
            out.append({"window": win, "criterion": crit["id"], "fp_kappa": fpk,
                        "fp_theta": fpt, "viable": viable,
                        "frac_lead": nlead / n if n else 0.0,
                        "frac_A": nA / n if n else 0.0,
                        "supports_A": viable and n > 0 and nA / n >= gate,
                        "kappa_alarm_steps": alarms})
    viable = [p for p in out if p["viable"]]
    if not usable:
        verdict = "VOID"
    elif not viable:
        verdict = "INCONCLUSIVE"
    elif any(p["supports_A"] for p in viable):
        verdict = "SUPPORTED"
    elif all(p["frac_A"] <= 1 - gate for p in viable):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"alpha": alpha, "n_usable": len(usable), "n_viable": len(viable),
            "verdict": verdict, "per_criterion": out}


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    res = {"experiment": "EXP-LLG-03", "environment": environment()}
    with Timer() as t1:
        res["arm1"] = arm1(list(range(20)), [0.25, 0.5, 1.0, 2.0, 4.0])
    res["arm1_seconds"] = t1.seconds
    cfg = json.loads((REPO / "sims/kappa_eff_leading/config.json").read_text())
    alphas = [0.5, 1.0, 2.0]
    with Timer() as t2:
        drift = [trajectory(s, sg, md, cfg, alphas)
                 for md in cfg["sweeps"]["drift_mode"]
                 for sg in cfg["sweeps"]["drift_sigma"] for s in cfg["seeds"]]
        null = [trajectory(s, 0.0, "isotropic", cfg, alphas) for s in cfg["seeds"]]
    res["arm2_seconds"] = t2.seconds
    res["arm2_control_alpha1_max_abs_diff_vs_repo_kappa"] = max(
        abs(a - b) for r in drift + null
        for a, b in zip(r["series"]["kappa_a1.0"], r["series"]["kappa_repo"]))
    res["arm2_breach_steps"] = [(r["mode"], r["sigma"], r["seed"], r["breach_step"]) for r in drift]
    res["arm2_null_breaches"] = sum(r["breach_step"] is not None for r in null)
    res["arm2_grading"] = [grade(drift, null, cfg, a) for a in alphas]
    res["arm2_series"] = {"drift": drift, "null": null}
    write_json(HERE / "raw.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("arm1", "arm2_series", "arm2_grading")}))
    for g in res["arm2_grading"]:
        print(g["alpha"], g["verdict"], g["n_viable"], g["n_usable"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
