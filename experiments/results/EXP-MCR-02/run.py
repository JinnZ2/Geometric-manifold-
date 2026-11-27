#!/usr/bin/env python3
"""EXP-MCR-02 run (2026-11-27): do standard EWS lead a GMM collapse event, against a matched null?

Toy: ../EXP-MCR-01/mcr_gmm.py (K = 5 2-D GMM; real distribution defined there), EM and
sampling functions reused; the loop is re-implemented here to record per-generation summaries.
Summary statistics: sigma_t (overall sd of the fitted GMM, primary) and smallest component
weight (secondary).
Event 1 (pre-registered): E = first generation t >= 1 at which a component weight < 1/(10K) = 0.02.
Event 2 (pre-registered robustness): first t >= 1 at which tail mass P(|x| > 3 sigma0) < 50% of
generation 0's. Tail mass per generation is estimated with common random numbers (2,000 fixed
standard-normal draws per component, the same in every generation and chain), so the series
is a smooth function of the fitted parameters; this estimator choice is a deviation (the design
did not specify one).
EWS: sketches/mcr_recursive_gaussian.py::ews on the segment [0, E) with rolling window w in
{10, 20, 40}; a run enters a w-cell only if E - w + 1 >= 10 (at least 10 rolling values).
Null at matched generation index: the same segment [0, E) of (a) accumulate runs and (b) replace
runs with no event in [0, E] (generation index matched; (b) is taken per event run, so a replace
run whose event comes later than E is a valid null for that E, an extension of "no event within
the horizon" recorded here before running). AUC = P(tau_event > tau_null) over matched pairs,
ties count 1/2; 95% CI by bootstrap over event runs and null runs (1,000 resamples).
Scale (PILOT, recorded before running): pre-registered 500 seeds. Run: replace 100 seeds at
n = 50 and 100, 50 at n = 300; accumulate 3 seeds at n = 50 and 100, none at n = 300; T = 200.
Reason: shared 4-core machine at load ~20-30; replace n = 300 measured at 1.2 s per seed per 50
generations and accumulate cost grows as n T^2. A first launch at 200/200/100 replace and
10/10 accumulate seeds was stopped before writing any output (projected > 1.5 h).
Output: raw.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib.util  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "EXP-MCR-01"))
import mcr_gmm as g  # noqa: E402
from common import REPO, Timer, environment, write_json  # noqa: E402

SMOKE = os.environ.get("MCR02_SMOKE") == "1"  # tiny end-to-end check of the analysis code only
T = 40 if SMOKE else 200
WS = (10, 20, 40)
FLOOR = 1 / (10 * g.K)
NZ = 2000
_zr = np.random.default_rng(424242)
Z = _zr.standard_normal((NZ, 2))


def _sketch():
    spec = importlib.util.spec_from_file_location(
        "mcr_sketch", REPO / "experiments/sketches/mcr_recursive_gaussian.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tail_crn(w, mu, cov):
    L = np.linalg.cholesky(cov + g.REG * np.eye(2))           # (S,K,2,2)
    x = mu[:, :, None, :] + np.einsum("skij,mj->skmi", L, Z)  # (S,K,m,2)
    p = (np.linalg.norm(x, axis=-1) > 3 * g.SIGMA0).mean(-1)  # (S,K)
    return (w * p).sum(1)


def chains(args):
    n, temp, mode, seeds = args
    S = len(seeds)
    rng = np.random.default_rng([20261127, n, int(temp * 1000), 0 if mode == "replace" else 1, seeds[0], S])
    w = np.tile(g.TRUE_W, (S, 1))
    mu = np.tile(g.TRUE_MU, (S, 1, 1))
    cov = np.tile(g.TRUE_SD ** 2 * np.eye(2), (S, g.K, 1, 1))
    data = g.sample_gmm(rng, w, mu, cov, n)
    sd, mw, tail = np.zeros((S, T + 1)), np.zeros((S, T + 1)), np.zeros((S, T + 1))
    for t in range(T + 1):
        w, mu, cov = g.em(data, w, mu, cov)
        sd[:, t], mw[:, t], tail[:, t] = g.overall_sd(w, mu, cov), w.min(1), tail_crn(w, mu, cov)
        synth = g.sample_gmm(rng, w, mu, cov, n, temp)
        data = synth if mode == "replace" else np.concatenate([data, synth], axis=1)
    return {"n": n, "temp": temp, "mode": mode, "seeds": list(seeds),
            "sd": sd.tolist(), "min_w": mw.tolist(), "tail": tail.tolist()}


def first_event(x, cond):
    idx = np.where(cond(x[1:]))[0]
    return int(idx[0] + 1) if len(idx) else None


def prefix_taus(sk, series, w):
    """Kendall tau (sketch definition) of every prefix of the rolling-var and rolling-AC1
    series: out[k][m] = kendall_tau(rolling[:m]). Equal to sk.ews(series[:E], w) at m = E - w + 1
    (rolling windows ending inside [0, E) are the first E - w + 1 values; asserted in main)."""
    out = {}
    for key, fn in (("tau_var", np.var), ("tau_ac1", sk.ac1)):
        r = sk.rolling(np.asarray(series), w, fn)
        L = len(r)
        sg = np.sign(r[None, :] - r[:, None])          # sg[i, j] = sign(r_j - r_i)
        col = np.triu(sg, 1).sum(0)                     # sum over i < j
        cum = np.concatenate([[0.0], np.cumsum(col)])   # cum[m] = sum over j < m
        m = np.arange(L + 1)
        den = np.where(m >= 3, m * (m - 1) / 2, 1.0)
        out[key] = np.where(m >= 3, cum / den, 0.0)
    return out


def auc_pairs(ev, nulls):
    """ev: list of (E, tau); nulls: dict E -> array of null taus. Returns AUC and pair arrays."""
    num, den = 0.0, 0
    for E, tau in ev:
        nz = nulls[E]
        if len(nz) == 0:
            continue
        num += (tau > nz).sum() + 0.5 * (tau == nz).sum()
        den += len(nz)
    return num / den if den else float("nan"), den


def main():
    sk = _sketch()
    jobs = []
    for temp in (1.0, 0.9):
        for n, S in (((50, 8), (100, 8), (300, 4)) if SMOKE else ((50, 100), (100, 100), (300, 50))):
            for s0 in range(0, S, 25):
                jobs.append((n, temp, "replace", list(range(s0, min(S, s0 + 25)))))
        for n in (50, 100):
            jobs.append((n, temp, "accumulate", [0, 1, 2]))
    jobs.sort(key=lambda j: -(j[0] * (4 if j[2] == "accumulate" else 1)))
    with Timer() as tm, Pool(2) as pool:
        parts = pool.map(chains, jobs, chunksize=1)
    res = {"experiment": "EXP-MCR-02", "environment": environment(), "sim_seconds": tm.seconds,
           "scale": "PILOT", "cells": []}
    rng = np.random.default_rng(7)
    checked = {}
    for temp in (1.0, 0.9):
        for n in (50, 100, 300):
            rep = [p for p in parts if p["n"] == n and p["temp"] == temp and p["mode"] == "replace"]
            acc = [p for p in parts if p["n"] == n and p["temp"] == temp and p["mode"] == "accumulate"]
            R = {k: np.concatenate([np.array(p[k]) for p in rep]) for k in ("sd", "min_w", "tail")}
            A = {k: (np.concatenate([np.array(p[k]) for p in acc]) if acc else np.zeros((0, T + 1)))
                 for k in ("sd", "min_w", "tail")}
            ev_defs = {"E1_minw": lambda d, i: first_event(d["min_w"][i], lambda x: x < FLOOR),
                       "E2_tail": lambda d, i: first_event(d["tail"][i], lambda x, t0=d["tail"][i][0]: x < 0.5 * t0)}
            for ev_name, evf in ev_defs.items():
                Er = [evf(R, i) for i in range(len(R["sd"]))]
                Ea = [evf(A, i) for i in range(len(A["sd"]))]
                cell = {"n": n, "temp": temp, "event": ev_name, "replace_runs": len(Er),
                        "accumulate_runs": len(Ea),
                        "replace_event_frac": float(np.mean([e is not None for e in Er])),
                        "accumulate_event_frac": float(np.mean([e is not None for e in Ea])) if Ea else None,
                        "E_quantiles": (np.percentile([e for e in Er if e is not None], [10, 50, 90]).tolist()
                                        if any(e is not None for e in Er) else None),
                        "stats": {}}
                for stat in ("sd", "min_w"):
                    for w in WS:
                        pre = {("r", j): prefix_taus(sk, R[stat][j], w) for j in range(len(Er))}
                        pre.update({("a", j): prefix_taus(sk, A[stat][j], w) for j in range(len(Ea))})
                        for tau_key in ("tau_var", "tau_ac1"):
                            def tau_of(src, j, E):
                                return pre[(src, j)][tau_key][E - w + 1]
                            ev = [(E, tau_of("r", i, E), i) for i, E in enumerate(Er)
                                  if E is not None and E - w + 1 >= 10]
                            Es = sorted({e[0] for e in ev})
                            nul, null_src = {}, {}
                            for E in Es:
                                src = [("r", j) for j, Ej in enumerate(Er) if Ej is None or Ej > E]
                                src += [("a", j) for j, Ej in enumerate(Ea) if Ej is None or Ej > E]
                                nul[E] = np.array([tau_of(a, j, E) for a, j in src])
                                null_src[E] = src
                            if ev and not checked.get((stat, w)):
                                E0, t0, i0 = ev[0]
                                ref = sk.ews(R[stat][i0][:E0], w)[tau_key]
                                assert abs(ref - t0) < 1e-12, (ref, t0)
                                checked[(stat, w)] = True
                            auc, pairs = auc_pairs([(E, t) for E, t, _ in ev], nul)
                            boots = []
                            if len(ev) >= 5 and pairs:
                                for _ in range(1000):
                                    bev = [ev[k] for k in rng.integers(0, len(ev), len(ev))]
                                    keep_r = set(rng.integers(0, len(Er), len(Er)).tolist())
                                    keep_a = set(rng.integers(0, len(Ea), len(Ea)).tolist()) if Ea else set()
                                    bn = {}
                                    for E in Es:
                                        m = np.array([(a == "r" and j in keep_r) or (a == "a" and j in keep_a)
                                                      for a, j in null_src[E]], bool)
                                        bn[E] = nul[E][m] if len(m) else np.array([])
                                    a_, _ = auc_pairs([(E, t) for E, t, _ in bev], bn)
                                    if not np.isnan(a_):
                                        boots.append(a_)
                            evt = np.array([t for _, t, _ in ev])
                            allnull = np.concatenate([nul[E] for E in Es]) if Es else np.array([])
                            cell["stats"][f"{stat}|w{w}|{tau_key}"] = {
                                "n_event_runs": len(ev), "n_pairs": int(pairs), "auc": auc,
                                "auc_ci95": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                                             if boots else None),
                                "event_tau_median": float(np.median(evt)) if len(evt) else None,
                                "event_tau_frac_pos": float((evt > 0).mean()) if len(evt) else None,
                                "null_tau_median": float(np.median(allnull)) if len(allnull) else None,
                                "n_null_windows_pooled": int(len(allnull))}
                res["cells"].append(cell)
                print(n, temp, ev_name, cell["replace_event_frac"],
                      {k: round(v["auc"], 3) if v["auc"] == v["auc"] else None
                       for k, v in cell["stats"].items() if k.startswith("sd|w20")}, flush=True)
    res["series_summary"] = [{"n": p["n"], "temp": p["temp"], "mode": p["mode"], "seeds": p["seeds"],
                              "sd_mean": np.array(p["sd"]).mean(0)[::10].tolist(),
                              "tail_mean": np.array(p["tail"]).mean(0)[::10].tolist()} for p in parts]
    write_json(Path("/tmp/mcr02_smoke.json") if SMOKE else HERE / "raw.json", res)


if __name__ == "__main__":
    main()
