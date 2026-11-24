#!/usr/bin/env python3
"""SKETCH, not results. EXP-LLG-03: does kappa_eff read the function or the coordinates?

Status: PROPOSED. Written 2026-11-24. Only `--selftest` was run by the author.

ToyLLM (`simulation/environment.py`) is fc1 -> ReLU -> fc2. For any alpha > 0,
    (W1, b1, W2, b2) -> (alpha W1, alpha b1, W2 / alpha, b2)
leaves every logit unchanged (ReLU is positively homogeneous). Anything defined in function
space -- the KL basin B = {theta : KL(f_theta || f_ref) < eps} -- is therefore unchanged.
kappa_eff = v^T H v / v^T v (`addon_thermodynamic_control/stability.py::
SpectralCertificate.effective_repair_curvature`) is a Euclidean Rayleigh quotient, so it is
not expected to be invariant (cf. Dinh et al. 2017, arXiv 1703.04933, for sharpness).

The sketch reuses the repo's own `model_fn` and `effective_repair_curvature`, applies the
rescaling to theta, theta_ref and the drift direction (pushed forward by the same linear
map), and reports kappa_eff(alpha) and the max logit difference. It needs torch (a repo
dependency); without torch it exits 2 with a message.

Usage (from repo root):
    python3 experiments/sketches/llg_rescaling.py --selftest
    python3 experiments/sketches/llg_rescaling.py --alphas 0.25 0.5 1 2 4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import torch
except ImportError:  # pragma: no cover
    print(json.dumps({"skipped": "torch not installed"}))
    raise SystemExit(2)

from addon_thermodynamic_control.stability import SpectralCertificate  # noqa: E402
from simulation.environment import model_fn  # noqa: E402

D_IN, D_H, D_OUT = 32, 64, 16
S1 = D_IN * D_H
S2 = S1 + D_H
S3 = S2 + D_H * D_OUT


def rescale(theta: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear map R_alpha: scale fc1 (W1, b1) by alpha and W2 by 1/alpha; b2 unchanged."""
    t = theta.clone()
    t[:S2] = t[:S2] * alpha
    t[S2:S3] = t[S2:S3] / alpha
    return t


def setup(seed: int, drift: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    n_par = S3 + D_OUT
    theta_ref = 0.1 * torch.randn(n_par, generator=g)
    v = torch.randn(n_par, generator=g)
    theta = theta_ref + drift * v / v.norm()
    x = torch.randn(32, D_IN, generator=g)
    return theta_ref, theta, v, x


def run(seed: int, alphas):
    theta_ref, theta, v, x = setup(seed)
    cert = SpectralCertificate()
    rows = []
    base = model_fn(x, theta)
    for a in alphas:
        th_a, ref_a, v_a = rescale(theta, a), rescale(theta_ref, a), rescale(v, a)
        dlogit = float((model_fn(x, th_a) - base).abs().max())
        k = cert.effective_repair_curvature(model_fn, th_a, ref_a, x, v_a)
        rows.append({"alpha": a, "max_abs_logit_diff": dlogit, "kappa_eff": k})
    return {"sketch": "EXP-LLG-03", "status": "SKETCH OUTPUT - not a pre-registered result",
            "seed": seed, "rows": rows}


def selftest() -> int:
    fails = []
    theta_ref, theta, v, x = setup(0)
    for a in (0.5, 2.0):
        d = float((model_fn(x, rescale(theta, a)) - model_fn(x, theta)).abs().max())
        if d > 1e-4:
            fails.append(f"alpha={a}: function changed by {d:.2e}")
    if not torch.allclose(rescale(rescale(theta, 3.0), 1 / 3.0), theta, atol=1e-6):
        fails.append("rescale not invertible")
    k = SpectralCertificate().effective_repair_curvature(model_fn, theta, theta_ref, x, v)
    if not (k == k and k >= 0):
        fails.append("kappa_eff not a finite non-negative number")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failures": fails}))
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    json.dump(run(a.seed, a.alphas), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
