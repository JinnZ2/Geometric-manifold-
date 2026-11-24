#!/usr/bin/env python3
"""SKETCH, not results. EXP-FIT-01: is the repo's "repair energy" a property of the path
or of how finely the path is stepped?

Status: PROPOSED. Written 2026-11-24. Only `--selftest` was run by the author.

`addon_thermodynamic_control/energy.py::RepairEnergyAccumulator` sums
    C = sum_k  delta_k^T G(theta_k) delta_k        (G = diagonal Fisher)
Mathematically (DERIVED, standard information geometry; not measured here):
  * for small delta, delta^T F delta ~= 2 KL(p_theta || p_theta+delta)  (nats)
  * for a fixed path cut into N equal steps, C_N ~= A / N, where A = int ||dtheta/ds||_F^2 ds
    is an action, while L_N = sum_k sqrt(delta_k^T F delta_k) -> L, the Fisher path length,
    independent of N. By Cauchy-Schwarz, A >= L^2 (equality for constant-speed paths).
So C is not a path invariant; it scales with step count / learning rate. This sketch
builds a softmax-regression model with an EXACT Fisher (numpy only), cuts the same path
into N steps, and reports C_N, N*C_N, L_N and sum of 2*KL per step, with full and diagonal
Fisher. It does not touch the repo's torch model; the full EXP-FIT-01 replays the repo's
repair trajectories (see experiments/fisher-information-and-thermodynamics-of-learning.md).

Usage:
    python3 experiments/sketches/fit_discretization.py --selftest
    python3 experiments/sketches/fit_discretization.py --seed 0 --Ns 5 10 20 40 80 160
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """p(y|x) = softmax(W x); theta = W.ravel(); W is (K, d)."""

    def __init__(self, X: np.ndarray, K: int):
        self.X, self.K, self.d = X, K, X.shape[1]

    def probs(self, theta):
        return softmax(self.X @ theta.reshape(self.K, self.d).T)

    def fisher(self, theta, diagonal=False):
        """Exact model Fisher averaged over X: mean_i (diag(p)-pp^T) kron x x^T."""
        P = self.probs(theta)
        n = self.X.shape[0]
        F = np.zeros((self.K * self.d, self.K * self.d))
        for i in range(n):
            p = P[i]
            A = np.diag(p) - np.outer(p, p)
            F += np.kron(A, np.outer(self.X[i], self.X[i]))
        F /= n
        return np.diag(np.diag(F)) if diagonal else F

    def mean_kl(self, t0, t1):
        p, q = self.probs(t0), self.probs(t1)
        return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def curved_path(t0, t1, bend, s):
    """theta(s) = straight line + a bend orthogonal-ish to it (fixed endpoints)."""
    return t0 + s * (t1 - t0) + np.sin(np.pi * s) * bend


def measure(model, t0, t1, bend, N, diagonal):
    s = np.linspace(0.0, 1.0, N + 1)
    pts = [curved_path(t0, t1, bend, si) for si in s]
    C = L = KL2 = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        G = model.fisher(a, diagonal=diagonal)
        q = float(d @ G @ d)
        C += q
        L += np.sqrt(max(q, 0.0))
        KL2 += 2.0 * model.mean_kl(a, b)
    return {"N": N, "C_N": C, "N_times_C_N": N * C, "L_N": L, "sum_2KL": KL2}


def build(seed, n=64, d=4, K=3, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    model = SoftmaxRegression(X, K)
    t0 = rng.normal(scale=scale, size=K * d)
    t1 = t0 + rng.normal(scale=scale, size=K * d)
    bend = rng.normal(scale=0.5 * scale, size=K * d)
    return model, t0, t1, bend


def run(seed, Ns):
    model, t0, t1, bend = build(seed)
    return {"sketch": "EXP-FIT-01", "status": "SKETCH OUTPUT - not a pre-registered result",
            "seed": seed,
            "full_fisher": [measure(model, t0, t1, bend, N, False) for N in Ns],
            "diag_fisher": [measure(model, t0, t1, bend, N, True) for N in Ns]}


def selftest() -> int:
    """Checks of the code against known mathematics, on one synthetic seed."""
    fails = []
    model, t0, _, _ = build(123)
    rng = np.random.default_rng(7)
    # (1) second-order identity: d^T F d ~= 2 KL for small d
    d = 1e-3 * rng.normal(size=t0.shape)
    q = float(d @ model.fisher(t0) @ d)
    kl2 = 2 * model.mean_kl(t0, t0 + d)
    if not abs(q - kl2) / max(q, 1e-30) < 1e-2:
        fails.append(f"dFd={q:.3e} vs 2KL={kl2:.3e}")
    # (2) Fisher is PSD
    if np.linalg.eigvalsh(model.fisher(t0)).min() < -1e-10:
        fails.append("Fisher not PSD")
    # (3) Riemann sums: L_N converges, N*C_N converges (code check of measure())
    model, t0, t1, bend = build(123)
    r = [measure(model, t0, t1, bend, N, False) for N in (40, 80, 160)]
    if abs(r[2]["L_N"] - r[1]["L_N"]) / r[2]["L_N"] > 0.02:
        fails.append("L_N not converging")
    if abs(r[2]["N_times_C_N"] - r[1]["N_times_C_N"]) / r[2]["N_times_C_N"] > 0.05:
        fails.append("N*C_N not converging")
    # (4) Cauchy-Schwarz: N*C_N >= L_N^2 (up to discretisation)
    if r[2]["N_times_C_N"] < r[2]["L_N"] ** 2 * (1 - 1e-6):
        fails.append("Cauchy-Schwarz violated")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failures": fails}))
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ns", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    json.dump(run(a.seed, a.Ns), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
