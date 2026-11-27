#!/usr/bin/env python3
"""EXP-HVD-04 run (2026-11-27): Q7, does K >= 2 recover an unmodelled constraint against a
non-degenerate null? Config: config.json (this directory); pre-registration: prereg.md.
Reuses sims/dark_constraint/run.py (align, load_cases, EDGES, V0, relax for the equivalence
check) read-only. Output: raw.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, environment, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("dark_run", REPO / "sims/dark_constraint/run.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
CFG = json.loads((HERE / "config.json").read_text())
EI = np.array([e[0] for e in D.EDGES])
EJ = np.array([e[1] for e in D.EDGES])
D_REF = np.linalg.norm(D.V0[EI] - D.V0[EJ], axis=1)


def relax(residuals, load=None, kind=None, loc=None, s=0.0):
    rc = CFG["relaxation"]
    target = D_REF * (1.0 + np.asarray(residuals, float))
    V = D.V0.copy()
    if kind == "brace" and s != 0.0:
        bi, bj = loc
        b_t = np.linalg.norm(D.V0[bi] - D.V0[bj]) * (1.0 + s)
    if kind == "tether" and s != 0.0:
        ti, u = loc[0], np.asarray(loc[1], float)
        anchor = D.V0[ti] + s * u
    for _ in range(rc["steps"]):
        d = V[EI] - V[EJ]
        L = np.linalg.norm(d, axis=1)
        g = (2.0 * (L - target) / np.where(L > 1e-12, L, 1.0))[:, None] * d
        g[L <= 1e-12] = 0.0
        grad = np.zeros_like(V)
        np.add.at(grad, EI, g)
        np.add.at(grad, EJ, -g)
        if kind == "brace" and s != 0.0:
            dd = V[bi] - V[bj]
            LL = np.linalg.norm(dd)
            if LL > 1e-12:
                gg = 2.0 * (LL - b_t) * dd / LL
                grad[bi] += gg
                grad[bj] -= gg
        if kind == "tether" and s != 0.0:
            grad[ti] += 2.0 * (V[ti] - anchor)
        if load is not None:
            grad = grad - load
        V = V - rc["lr"] * grad
    return V


def observe(r, load, kind=None, loc=None, s=0.0):
    return (D.align(relax(r, load, kind, loc, s)) - (D.V0 - D.V0.mean(axis=0))).reshape(-1)


def jac(load):
    eps = CFG["relaxation"]["jacobian_epsilon"]
    z = np.zeros(len(D.EDGES))
    base = observe(z, load)
    J = np.zeros((base.size, len(D.EDGES)))
    for e in range(len(D.EDGES)):
        r = z.copy()
        r[e] = eps
        p = observe(r, load)
        r[e] = -eps
        J[:, e] = (p - observe(r, load)) / (2 * eps)
    return base, J


def seed_job(seed):
    cfg_d = {"load": CFG["load"]}
    loads = D.load_cases(max(CFG["n_load_cases"]), cfg_d, seed)
    model = [jac(ld) for ld in loads]
    r_vis = np.random.default_rng(5000 + seed).normal(0, CFG["visible_residual_sd"], len(D.EDGES))
    out = []
    for kind, locs in CFG["locations"].items():
        for li, loc in enumerate(locs):
            for s in CFG["dark_strength"]:
                truths = [observe(r_vis, ld, kind, loc, s) for ld in loads]
                for k in CFG["n_load_cases"]:
                    A = np.vstack([model[i][1] for i in range(k)])
                    y = np.concatenate([truths[i] - model[i][0] for i in range(k)])
                    rh, *_ = np.linalg.lstsq(A, y, rcond=None)
                    out.append({"seed": seed, "kind": kind, "loc": li, "s": s, "k": k,
                                "unexplained": float(np.linalg.norm(A @ rh - y)),
                                "r_hat_err": float(np.linalg.norm(rh - r_vis))})
    return out


def check_equivalence():
    cfg_r = {"relaxation": CFG["relaxation"]}
    ld = D.load_cases(1, {"load": CFG["load"]}, 0)[0]
    r = np.random.default_rng(1).normal(0, 0.03, len(D.EDGES))
    a = D.relax(r, cfg_r, ld, dark=0.1)
    b = relax(r, ld, "brace", (0, 1), 0.1)
    c = D.relax(r, cfg_r, ld, dark=0.0)
    e = relax(r, ld)
    diff = float(max(np.abs(a - b).max(), np.abs(c - e).max()))
    assert diff < 1e-10, diff
    return diff


def main():
    res = {"experiment": "EXP-HVD-04", "environment": environment(), "config": CFG,
           "relax_equivalence_max_abs_diff": check_equivalence()}
    with Timer() as t, Pool(3) as pool:
        rows = [r for part in pool.map(seed_job, CFG["seeds"]) for r in part]
    res["seconds"] = t.seconds
    res["rows"] = rows
    summ = []
    for kind, locs in CFG["locations"].items():
        for li in range(len(locs)):
            for k in CFG["n_load_cases"]:
                sel = [r for r in rows if r["kind"] == kind and r["loc"] == li and r["k"] == k]
                med = {s: float(np.median([r["unexplained"] for r in sel if r["s"] == s])) for s in CFG["dark_strength"]}
                floor = med[0.0]
                rho = spearmanr([r["s"] for r in sel], [r["unexplained"] for r in sel]).statistic
                summ.append({"kind": kind, "loc": li, "k": k, "floor": floor, "median_by_s": med,
                             "ratio_to_floor": {s: (m / floor if floor > 0 else None) for s, m in med.items()},
                             "spearman_rho": float(rho)})
    res["summary"] = summ
    write_json(HERE / "raw.json", res)
    for c in summ:
        print(c["kind"], c["loc"], c["k"], "floor %.2e" % c["floor"], "rho %.2f" % c["spearman_rho"],
              {s: (round(v, 1) if v else v) for s, v in c["ratio_to_floor"].items()})


if __name__ == "__main__":
    main()
