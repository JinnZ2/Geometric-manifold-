#!/usr/bin/env python3
"""EXP-MCR-01 run (2026-11-27): replace vs accumulate in recursive Gaussian and GMM refitting.

1-D part: experiments/sketches/mcr_recursive_gaussian.py::simulate unchanged, n in
{20, 100, 1000}, T = 200, 500 seeds, replace and accumulate. Supplementary (not
pre-registered, labelled as such): 20,000 seeds at n = 20, t <= 10, for a tighter code check.
GMM part: mcr_gmm.py (this directory). Pre-registered: n in {20, 100, 1000}, T = 200,
500 seeds. Run (PILOT for every arm below 500 seeds): replace 500 seeds at n = 20 and 100,
100 seeds at n = 1000; accumulate 50 seeds at n = 20, 10 seeds at n = 100, and 3 seeds at
n = 1000 with T = 100 instead of 200. Compute-limited: accumulate cost grows as n*T^2 and the
machine was shared (load average ~20-30 on 4 cores, this process getting ~0.15 core per
worker). A first launch at larger scale (accumulate 200/50/5 seeds, replace 500 everywhere)
was stopped before writing any output because its projected runtime was > 4 h; the scale was
reduced before any GMM result was seen. Recorded here before any result.
Output: raw.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import mcr_gmm as g  # noqa: E402
from common import REPO, environment, write_json  # noqa: E402

CHECK = (0, 10, 25, 50, 100, 150, 200)
T = 200


def _sketch():
    spec = importlib.util.spec_from_file_location(
        "mcr_sketch", REPO / "experiments/sketches/mcr_recursive_gaussian.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one_d(n, mode, seeds):
    sk = _sketch()
    S = np.array([sk.simulate(n, T, s, mode) for s in range(seeds)]) ** 2
    return {"n": n, "mode": mode, "seeds": seeds,
            "mean_var": S.mean(0).tolist(), "median_var": np.median(S, 0).tolist(),
            "se_mean_var": (S.std(0, ddof=1) / np.sqrt(seeds)).tolist(),
            "analytic_mean_var_replace": [((n - 1) / n) ** (t + 1) for t in range(T + 1)]}


def one_d_supp():
    sk = _sketch()
    S = np.array([sk.simulate(20, 10, s, "replace") for s in range(20000)]) ** 2
    return {"n": 20, "seeds": 20000, "T": 10, "mean_var": S.mean(0).tolist(),
            "se_mean_var": (S.std(0, ddof=1) / np.sqrt(20000)).tolist(),
            "analytic": [(19 / 20) ** (t + 1) for t in range(11)]}


def gmm_job(args):
    n, mode, S = args[:3]
    TT = args[3] if len(args) > 3 else T
    t0 = time.perf_counter()
    chunk = 100 if mode == "replace" else max(1, min(S, 50))
    parts = []
    for start in range(0, S, chunk):
        seeds = list(range(start, min(S, start + chunk)))
        parts.append(g.simulate(n, TT, seeds, mode, 1.0, tail_at=tuple(c for c in CHECK if c <= TT)))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("min_w", "n_above_floor", "sd", "n_alive")}
    chk = [c for c in CHECK if c <= TT]
    tail = {t: np.concatenate([p["tail"][t] for p in parts]) for t in chk}
    return {"n": n, "mode": mode, "seeds": S, "T": TT, "seconds": time.perf_counter() - t0,
            "tail_by_seed": {str(t): tail[t].tolist() for t in chk},
            "min_w_by_seed": {str(t): cat["min_w"][:, t].tolist() for t in chk},
            "n_above_floor_by_seed": {str(t): cat["n_above_floor"][:, t].tolist() for t in chk},
            "sd_by_seed": {str(t): cat["sd"][:, t].tolist() for t in chk},
            "median_sd_per_gen": np.median(cat["sd"], 0).tolist(),
            "median_min_w_per_gen": np.median(cat["min_w"], 0).tolist()}


def dispatch(job):
    kind = job[0]
    if kind == "1d":
        return ("1d", one_d(*job[1:]))
    if kind == "1d_supp":
        return ("1d_supp", one_d_supp())
    return ("gmm", gmm_job(job[1:]))


def main():
    t0 = time.perf_counter()
    jobs = [("gmm", 100, "accumulate", 10), ("gmm", 1000, "accumulate", 3, 100),
            ("gmm", 20, "accumulate", 50), ("gmm", 1000, "replace", 100),
            ("gmm", 100, "replace", 500), ("gmm", 20, "replace", 500), ("1d_supp",)]
    jobs += [("1d", n, m, 500) for n in (20, 100, 1000) for m in ("replace", "accumulate")]
    with Pool(4) as pool:
        results = pool.map(dispatch, jobs, chunksize=1)
    out = {"experiment": "EXP-MCR-01", "environment": environment(),
           "sigma0": g.SIGMA0, "true_tail_mass": g.true_tail_mass(),
           "checkpoints": CHECK, "one_d": [], "gmm": []}
    for kind, r in results:
        if kind == "1d":
            out["one_d"].append(r)
        elif kind == "1d_supp":
            out["one_d_supplementary"] = r
        else:
            out["gmm"].append(r)
    out["runtime_seconds"] = time.perf_counter() - t0
    write_json(HERE / "raw.json", out)
    print("done", out["runtime_seconds"])


if __name__ == "__main__":
    main()
