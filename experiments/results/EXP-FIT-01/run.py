#!/usr/bin/env python3
"""EXP-FIT-01 run (2026-11-27): is repair energy a path property or a step-size property?

Part 1: experiments/sketches/fit_discretization.py unchanged (exact Fisher, softmax
        regression, curved fixed-endpoint path), N in {5,10,20,40,80,160}, seeds 0-9.
        C_m / C_1 is read as C_{160/m} / C_{160} for m in {2,4,8}.
Part 2: the main.py parameter-repair trajectory (configs/default.yaml and
        configs/adversarial.yaml), env seeds 0-4. The trajectory is reproduced by calling
        ParameterManifold.repair_step in a loop exactly as simulation/controller.py does
        (the data and policy layers in Controller.run do not modify theta). theta saved
        every step. C_m, L_m recomputed on the path sub-sampled every m-th checkpoint,
        m in {1,2,4,8}, with G = (a) the repo's FisherMetricEstimator.diagonal (16 sampled
        labels, sampler seeded per checkpoint so G(theta_k) is identical across m) and
        (b) the exact full model Fisher on the safety inputs (via Jacobian-vector products).
        Also: sum of 2*KL per sub-step, and the angle between successive steps.
Part 3: same configs, env seeds 0-9 per arm. Arm "lr_change": ParameterManifold.lr is
        multiplied by sqrt(2) at step 50, nothing else changed. Arm "control": no change.
        Per-step energy delta^T diag(F_hat) delta is fed to the repo's
        RepairEnergyAccumulator; fire = recent_trend() > 2.0 at any step in [50, 70).
Output: raw.json.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, add_repo_to_path, environment, write_json  # noqa: E402

add_repo_to_path()
from addon_thermodynamic_control.energy import FisherMetricEstimator, RepairEnergyAccumulator  # noqa: E402
from manifolds.parameter_manifold import ParameterManifold  # noqa: E402
from simulation.environment import Environment  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def part1():
    sk = _load("fit_discretization_sketch", REPO / "experiments/sketches/fit_discretization.py")
    Ns = [5, 10, 20, 40, 80, 160]
    rows = []
    for s in range(10):
        r = sk.run(s, Ns)
        for kind in ("full_fisher", "diag_fisher"):
            by = {m["N"]: m for m in r[kind]}
            base = by[160]
            rows.append({"seed": s, "fisher": kind, "per_N": r[kind],
                         "C_ratio": {str(m): by[160 // m]["C_N"] / base["C_N"] for m in (2, 4, 8)},
                         "L_ratio": {str(m): by[160 // m]["L_N"] / base["L_N"] for m in (2, 4, 8)}})
    return rows


def load_cfg(name):
    return yaml.safe_load((REPO / "configs" / name).read_text())


def trajectory(cfg, seed, lr_change_at=None, lr_factor=1.0):
    sim = dict(cfg["simulation"], seed=seed)
    env = Environment(sim)
    mf = env.get_model_fn()
    layer = ParameterManifold(env.theta_ref, cfg["manifolds"]["parameter"])
    theta = env.theta_drifted.clone()
    path = [theta.clone()]
    for step in range(sim["steps"]):
        if lr_change_at is not None and step == lr_change_at:
            layer.lr = layer.lr * lr_factor
        theta, _ = layer.repair_step(theta, mf, env.safety_inputs, env.task_inputs,
                                     env.task_labels)
        path.append(theta.clone())
    return env, mf, path


def fisher_diag_seeded(est, theta, x, seed, k):
    torch.manual_seed(1_000_003 * seed + k)
    return est.diagonal(theta, x)


def exact_fisher_quad(mf, theta, delta, x):
    """delta^T F delta with F the exact model Fisher averaged over the inputs x."""
    logits, jd = torch.func.jvp(lambda t: mf(x, t), (theta,), (delta,))
    p = F.softmax(logits, dim=-1)
    q = (p * jd ** 2).sum(-1) - ((p * jd).sum(-1)) ** 2
    return float(q.mean())


def mean_kl(mf, a, b, x):
    pa = F.log_softmax(mf(x, a), dim=-1)
    pb = F.log_softmax(mf(x, b), dim=-1)
    return float((pa.exp() * (pa - pb)).sum(-1).mean())


def part2_one(cfg, seed):
    env, mf, path = trajectory(cfg, seed)
    x = env.safety_inputs
    est = FisherMetricEstimator(mf)
    T = len(path) - 1
    gdiag = {}
    out = {"seed": seed, "steps": T}
    for m in (1, 2, 4, 8):
        idx = list(range(0, T + 1, m))
        C = L = Cx = Lx = KL2 = 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            d = (path[b] - path[a]).detach()
            if a not in gdiag:
                gdiag[a] = fisher_diag_seeded(est, path[a], x, seed, a)
            q = float((d ** 2 * gdiag[a]).sum())
            qx = exact_fisher_quad(mf, path[a].detach(), d, x)
            C += q
            L += math.sqrt(max(q, 0.0))
            Cx += qx
            Lx += math.sqrt(max(qx, 0.0))
            KL2 += 2 * mean_kl(mf, path[a], path[b], x)
        out[f"m{m}"] = {"n_steps": len(idx) - 1, "C_diag": C, "L_diag": L,
                        "C_full": Cx, "L_full": Lx, "sum_2KL": KL2}
    deltas = [(path[k + 1] - path[k]) for k in range(T)]
    norms = [float(d.norm()) for d in deltas]
    angles = []
    for d0, d1 in zip(deltas[:-1], deltas[1:]):
        c = float(torch.dot(d0, d1) / (d0.norm() * d1.norm() + 1e-30))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    out["step_norms"] = norms
    out["trust_radius"] = cfg["manifolds"]["parameter"]["trust_radius"]
    out["frac_steps_clipped"] = sum(n >= 0.999 * out["trust_radius"] for n in norms) / T
    out["angle_deg_successive"] = angles
    return out


def part3_one(cfg, seed, change):
    env, mf, path = trajectory(cfg, seed, lr_change_at=50 if change else None,
                               lr_factor=math.sqrt(2) if change else 1.0)
    est = FisherMetricEstimator(mf)
    acc = RepairEnergyAccumulator()
    trends = []
    for k in range(len(path) - 1):
        g = fisher_diag_seeded(est, path[k], env.safety_inputs, seed, k)
        acc.update((path[k + 1] - path[k]).detach(), g)
        trends.append(acc.recent_trend())
    fired = any(trends[t] > 2.0 for t in range(50, min(70, len(trends))))
    return {"seed": seed, "arm": "lr_change" if change else "control", "fired": fired,
            "max_trend_50_70": max(trends[50:70]), "trend": trends,
            "energy": acc.per_step}


def fisher_exact_two_sided(a, b, c, d):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)


def main():
    torch.set_num_threads(1)
    res = {"experiment": "EXP-FIT-01", "environment": environment()}
    with Timer() as t:
        res["part1"] = part1()
    res["part1_seconds"] = t.seconds
    res["part2"], res["part3"] = {}, {}
    with Timer() as t:
        for name in ("default.yaml", "adversarial.yaml"):
            cfg = load_cfg(name)
            res["part2"][name] = [part2_one(cfg, s) for s in range(5)]
    res["part2_seconds"] = t.seconds
    with Timer() as t:
        for name in ("default.yaml", "adversarial.yaml"):
            cfg = load_cfg(name)
            runs = [part3_one(copy.deepcopy(cfg), s, ch) for ch in (True, False) for s in range(10)]
            fa = sum(r["fired"] for r in runs if r["arm"] == "lr_change")
            fc = sum(r["fired"] for r in runs if r["arm"] == "control")
            res["part3"][name] = {"runs": runs, "fires_lr_change": fa, "fires_control": fc,
                                  "fisher_exact_p": fisher_exact_two_sided(fa, 10 - fa, fc, 10 - fc)}
    res["part3_seconds"] = t.seconds
    write_json(HERE / "raw.json", res)
    print("done", res["part1_seconds"], res["part2_seconds"], res["part3_seconds"])


if __name__ == "__main__":
    main()
