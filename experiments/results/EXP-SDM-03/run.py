#!/usr/bin/env python3
"""EXP-SDM-03 run (2026-11-27): snapshot-delta monitoring of a linear monitor in floating point.

Positive control: docs/research/residual-tower/snapshot_delta_test1.py --run, output written
to this directory (the committed results file is not touched); test1_pass and the
deterministic sha256 must match the committed snapshot_delta_test1_results.json.

Float arms, monitor m(theta) = w . theta, w ~ N(0,1) float32:
  f32_inc    m_t = fl32(m_{t-1} + fl32(w32 . d32_t))    (numpy float32 dot)
  f64_inc    m_t = m_{t-1} + w64 . d64_t
  f32_kahan  Kahan-compensated float32 accumulation of the float32 per-step dots
  f32_full   fl32(w32 . theta32_t)                       (full recompute in float32)
Reference: float64 full recompute w64 . theta32_t (theta_t is the stored float32 checkpoint).
d32_t = theta32_t - theta32_{t-1} computed in float32; d64_t the same difference in float64.

(a) Synthetic: N = 200,000 entries (as in Test 1), theta_0 ~ N(0, 0.05^2), each step changes
    2,000 random entries by N(0, 1e-3^2) (random sign), T = 10,000 steps, 20 seeds; errors
    read at t in {10, 100, 1000, 10000} (prefixes of one run per seed).
(b) ToyLLM: the main.py parameter-repair trajectory (ParameterManifold.repair_step loop,
    as in EXP-FIT-01), default (100 steps) and adversarial (200 steps), env seeds 0-4,
    N = 3,152. T is limited to the trajectory length (deviation).
Flips: threshold tau = median of the reference series per run; at every step, a flip is a
disagreement of 1[m_arm > tau] with 1[ref > tau]; flips binned by |ref - tau| / |tau|.
explore_harness.py records: NOT written (see results.md). Output: raw.json.

Implementation (fixed before any synthetic result was seen): in arm (a) the per-step delta is
applied sparsely, as a snapshot-delta monitor would: inc32 = fl32(w32[idx] . d32[idx]) and
inc64 = w64[idx] . d64[idx] over the 2,000 changed entries; the float64 reference is
w64 . theta64 with theta64 the exact float64 image of the float32 checkpoint, kept in sync
sparsely. A first launch with dense per-step deltas and multithreaded BLAS ran at ~60 ms per
step on the loaded machine (projected > 3 h) and was stopped before writing any output.
Seeds run in parallel (Pool(3)).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, add_repo_to_path, environment, write_json  # noqa: E402

TOWER = REPO / "docs/research/residual-tower"
CHECK_T = (10, 100, 1000, 10000)
BINS = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, np.inf]


def positive_control():
    out = HERE / "test1_rerun_results.json"
    with Timer() as t:
        p = subprocess.run([sys.executable, str(TOWER / "snapshot_delta_test1.py"), "--run",
                            "--out", str(out)], cwd=TOWER, capture_output=True, text=True)
    committed = json.loads((TOWER / "snapshot_delta_test1_results.json").read_text())
    rerun = json.loads(out.read_text()) if out.exists() else {}
    return {"returncode": p.returncode, "seconds": t.seconds,
            "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:],
            "rerun_test1_pass": rerun.get("test1_pass"),
            "committed_test1_pass": committed.get("test1_pass"),
            "rerun_deterministic_sha256": rerun.get("deterministic_sha256"),
            "committed_deterministic_sha256": committed.get("deterministic_sha256"),
            "sha_match": rerun.get("deterministic_sha256") == committed.get("deterministic_sha256")}


def monitor_series(thetas32, w32):
    """thetas32: iterator of float32 checkpoints theta_0..theta_T. Returns dict of series."""
    w64 = w32.astype(np.float64)
    it = iter(thetas32)
    th_prev = next(it)
    ref = [float(w64 @ th_prev.astype(np.float64))]
    m32 = np.float32(w32 @ th_prev)
    m64 = ref[0]
    ks, kc = np.float32(m32), np.float32(0.0)
    s = {"ref": ref, "f32_inc": [float(m32)], "f64_inc": [m64], "f32_kahan": [float(ks)],
         "f32_full": [float(np.float32(w32 @ th_prev))]}
    for th in it:
        d32 = th - th_prev
        d64 = th.astype(np.float64) - th_prev.astype(np.float64)
        inc32 = np.float32(w32 @ d32)
        m32 = np.float32(m32 + inc32)
        m64 = m64 + float(w64 @ d64)
        y = np.float32(inc32 - kc)
        tt = np.float32(ks + y)
        kc = np.float32(np.float32(tt - ks) - y)
        ks = tt
        s["ref"].append(float(w64 @ th.astype(np.float64)))
        s["f32_inc"].append(float(m32))
        s["f64_inc"].append(m64)
        s["f32_kahan"].append(float(ks))
        s["f32_full"].append(float(np.float32(w32 @ th)))
        th_prev = th
    return s


def flips(s):
    ref = np.array(s["ref"])
    tau = float(np.median(ref))
    rel = np.abs(ref - tau) / max(abs(tau), 1e-30)
    out = {}
    for arm in ("f32_inc", "f64_inc", "f32_kahan", "f32_full"):
        m = np.array(s[arm])
        flip = (m > tau) != (ref > tau)
        out[arm] = {"total_flips": int(flip.sum()), "steps": int(len(ref)),
                    "by_margin_bin": [int(flip[(rel >= lo) & (rel < hi)].sum()) for lo, hi in zip(BINS[:-1], BINS[1:])],
                    "steps_by_margin_bin": [int(((rel >= lo) & (rel < hi)).sum()) for lo, hi in zip(BINS[:-1], BINS[1:])]}
    return {"tau": tau, "arms": out}


def synthetic(seed, N=200_000, k=2000, T=10_000):
    rng = np.random.default_rng([20261127, seed])
    w32 = rng.standard_normal(N).astype(np.float32)
    w64 = w32.astype(np.float64)
    th = (0.05 * rng.standard_normal(N)).astype(np.float32)
    th64 = th.astype(np.float64)
    ref0 = float(w64 @ th64)
    m32 = np.float32(w32 @ th)
    ks, kc = m32, np.float32(0.0)
    m64 = ref0
    S = {"ref": [ref0], "f32_inc": [float(m32)], "f64_inc": [m64], "f32_kahan": [float(ks)],
         "f32_full": [float(m32)]}
    for _ in range(T):
        idx = rng.choice(N, size=k, replace=False)
        old = th[idx].copy()
        new = (old + (1e-3 * rng.standard_normal(k)).astype(np.float32)).astype(np.float32)
        th[idx] = new
        th64[idx] = new.astype(np.float64)
        d32 = (new - old).astype(np.float32)
        d64 = new.astype(np.float64) - old.astype(np.float64)
        inc32 = np.float32(w32[idx] @ d32)
        m32 = np.float32(m32 + inc32)
        m64 = m64 + float(w64[idx] @ d64)
        y = np.float32(inc32 - kc)
        tt = np.float32(ks + y)
        kc = np.float32(np.float32(tt - ks) - y)
        ks = tt
        S["ref"].append(float(w64 @ th64))
        S["f32_inc"].append(float(m32))
        S["f64_inc"].append(m64)
        S["f32_kahan"].append(float(ks))
        S["f32_full"].append(float(np.float32(w32 @ th)))
    s = S
    ref = np.array(s["ref"])
    errs = {arm: {str(t): abs(s[arm][t] - ref[t]) for t in CHECK_T}
            for arm in ("f32_inc", "f64_inc", "f32_kahan", "f32_full")}
    # drift: error relative to the arm's own t = 0 error (the incremental arms inherit the
    # float32 full dot at t = 0). Added after a 1,000-step smoke test on seed 99 (not one of
    # the 20 seeds) showed the t = 0 offset dominating; reported alongside, not instead.
    drift = {arm: {str(t): abs((s[arm][t] - ref[t]) - (s[arm][0] - ref[0])) for t in CHECK_T}
             for arm in errs}
    return {"seed": seed, "abs_err_at": errs, "drift_at": drift, "ref_at": {str(t): float(ref[t]) for t in CHECK_T},
            "max_abs_err_upto": {arm: {str(t): float(np.max(np.abs(np.array(s[arm][:t + 1]) - ref[:t + 1])))
                                       for t in CHECK_T} for arm in errs},
            "flips": flips(s)}


def toyllm(cfg_name, seed):
    add_repo_to_path()
    import torch
    import yaml

    from manifolds.parameter_manifold import ParameterManifold
    from simulation.environment import Environment
    cfg = yaml.safe_load((REPO / "configs" / cfg_name).read_text())
    env = Environment(dict(cfg["simulation"], seed=seed))
    mf = env.get_model_fn()
    layer = ParameterManifold(env.theta_ref, cfg["manifolds"]["parameter"])
    theta = env.theta_drifted.clone()
    path = [theta.detach().numpy().astype(np.float32).copy()]
    for _ in range(cfg["simulation"]["steps"]):
        theta, _ = layer.repair_step(theta, mf, env.safety_inputs, env.task_inputs, env.task_labels)
        path.append(theta.detach().numpy().astype(np.float32).copy())
    rng = np.random.default_rng([20261127, 7, seed])
    w32 = rng.standard_normal(path[0].shape[0]).astype(np.float32)
    torch.set_num_threads(1)
    s = monitor_series(iter(path), w32)
    ref = np.array(s["ref"])
    return {"config": cfg_name, "seed": seed, "T": len(path) - 1, "N": int(path[0].shape[0]),
            "abs_err_final": {arm: abs(s[arm][-1] - ref[-1]) for arm in ("f32_inc", "f64_inc", "f32_kahan", "f32_full")},
            "max_abs_err": {arm: float(np.max(np.abs(np.array(s[arm]) - ref))) for arm in ("f32_inc", "f64_inc", "f32_kahan", "f32_full")},
            "ref_final": float(ref[-1]), "flips": flips(s)}


def slope(ts, errs):
    x = np.log10(np.array(ts, float))
    y = np.log10(np.maximum(np.array(errs, float), 1e-300))
    A = np.vstack([x, np.ones_like(x)]).T
    b, _a = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b)


def main():
    res = {"experiment": "EXP-SDM-03", "environment": environment()}
    res["positive_control"] = positive_control()
    with Timer() as t:
        with Pool(3) as pool:
            res["synthetic"] = pool.map(synthetic, range(20), chunksize=1)
    res["synthetic_seconds"] = t.seconds
    # pooled log-log slope of RMS error vs T, with bootstrap over seeds
    rng = np.random.default_rng(0)
    res["slopes"] = {}
    for arm, key in [(a, k) for k in ("abs_err_at", "drift_at") for a in ("f32_inc", "f64_inc", "f32_kahan", "f32_full")]:
        E = np.array([[r[key][arm][str(t)] for t in CHECK_T] for r in res["synthetic"]])
        rms = np.sqrt((E ** 2).mean(0))
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(E), len(E))
            boots.append(slope(CHECK_T, np.sqrt((E[idx] ** 2).mean(0))))
        res["slopes"][f"{key}|{arm}"] = {"rms_err_at": dict(zip(map(str, CHECK_T), rms.tolist())),
                              "loglog_slope": slope(CHECK_T, rms),
                              "slope_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    with Timer() as t:
        res["toyllm"] = [toyllm(c, s) for c in ("default.yaml", "adversarial.yaml") for s in range(5)]
    res["toyllm_seconds"] = t.seconds
    write_json(HERE / "raw.json", res)
    print(json.dumps(res["slopes"], indent=1))
    print(json.dumps({k: v for k, v in res["positive_control"].items() if "tail" not in k}))


if __name__ == "__main__":
    main()
