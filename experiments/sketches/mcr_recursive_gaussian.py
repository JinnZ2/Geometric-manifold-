#!/usr/bin/env python3
"""SKETCH, not results. EXP-MCR-01 / EXP-MCR-02: recursive refitting of a 1-D Gaussian.

Status: PROPOSED. Written 2026-11-24. Only `--selftest` was run by the author.

Generation t fits (mu_t, sigma_t^2) by maximum likelihood to a training set, then samples
the next training set from N(mu_t, sigma_t^2). Two data regimes:

  replace     generation t trains only on n samples drawn from generation t-1's model
              (the setting analysed in Shumailov et al., Nature 2024)
  accumulate  generation t trains on the real data plus all synthetic sets so far
              (the setting of Gerstgrasser et al., arXiv 2404.01413)

DERIVED (textbook, not measured): with the MLE variance and replace, E[sigma_t^2 | sigma_{t-1}^2]
= (n-1)/n * sigma_{t-1}^2, so E[sigma_t^2] = sigma_0^2 ((n-1)/n)^t. The median decays faster
than the mean because log sigma_t^2 is a random walk with negative drift.

EXP-MCR-02 adds the early-warning-signal (EWS) statistics computed ACROSS generations on a
summary series (sigma_t): rolling variance and lag-1 autocorrelation, with Kendall tau trend.
This sketch only provides the functions; whether they lead any defined "collapse event"
is the experiment, with its null arm, in
experiments/model-collapse-variance-suppression-and-recursive-training.md.

Usage:
    python3 experiments/sketches/mcr_recursive_gaussian.py --selftest
    python3 experiments/sketches/mcr_recursive_gaussian.py --n 100 --T 200 --seeds 200
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def simulate(n, T, seed, mode="replace", mu0=0.0, sigma0=1.0, temperature=1.0):
    """Return sigma_t (MLE std) for t = 0..T. temperature < 1 shrinks the sampler's std."""
    rng = np.random.default_rng(seed)
    real = rng.normal(mu0, sigma0, size=n)
    pool = [real]
    data = real
    sig = []
    for _ in range(T + 1):
        mu, s = float(data.mean()), float(data.std())  # MLE (ddof=0)
        sig.append(s)
        synth = rng.normal(mu, temperature * s, size=n)
        if mode == "replace":
            data = synth
        elif mode == "accumulate":
            pool.append(synth)
            data = np.concatenate(pool)
        else:
            raise ValueError(mode)
    return np.array(sig)


def kendall_tau(x):
    x = np.asarray(x)
    n = len(x)
    if n < 3:
        return 0.0
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    return float(s / (n * (n - 1) / 2))


def rolling(x, w, fn):
    return np.array([fn(x[i - w:i]) for i in range(w, len(x) + 1)])


def ac1(x):
    x = np.asarray(x) - np.mean(x)
    d = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else 0.0


def ews(series, w):
    """Kendall tau of rolling variance and rolling lag-1 AC (standard EWS pair)."""
    rv = rolling(series, w, np.var)
    ra = rolling(series, w, ac1)
    return {"tau_var": kendall_tau(rv), "tau_ac1": kendall_tau(ra)}


def run(n, T, seeds):
    out = {"sketch": "EXP-MCR-01", "status": "SKETCH OUTPUT - not a pre-registered result",
           "n": n, "T": T, "seeds": seeds}
    for mode in ("replace", "accumulate"):
        S = np.array([simulate(n, T, s, mode) for s in range(seeds)]) ** 2
        out[mode] = {"mean_var_ratio_at": {str(t): float(S[:, t].mean() / S[:, 0].mean())
                                           for t in (0, T // 4, T // 2, T)},
                     "median_var_ratio_at": {str(t): float(np.median(S[:, t] / S[:, 0]))
                                             for t in (0, T // 4, T // 2, T)}}
    out["replace_analytic_mean_ratio_at"] = {str(t): ((n - 1) / n) ** t
                                             for t in (0, T // 4, T // 2, T)}
    return out


def selftest() -> int:
    fails = []
    n, T, seeds = 20, 10, 4000
    S = np.array([simulate(n, T, s, "replace") for s in range(seeds)]) ** 2
    # code check vs the derived mean recursion (E over seeds, relative to sigma0^2 = 1;
    # generation 0 is itself an MLE fit to real data so E[S0] = (n-1)/n)
    for t in (1, 5, 10):
        expect = ((n - 1) / n) ** (t + 1)
        got = S[:, t].mean()
        if abs(got - expect) / expect > 0.05:
            fails.append(f"t={t}: mean var {got:.4f} vs analytic {expect:.4f}")
    if abs(kendall_tau(np.arange(10)) - 1.0) > 1e-12:
        fails.append("kendall_tau(increasing) != 1")
    if abs(ac1(np.tile([1.0, -1.0], 50)) + 0.99) > 0.02:
        fails.append("ac1(alternating) not ~ -1")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failures": fails}))
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    json.dump(run(a.n, a.T, a.seeds), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
