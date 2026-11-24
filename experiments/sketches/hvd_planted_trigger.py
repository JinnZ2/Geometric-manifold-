#!/usr/bin/env python3
"""SKETCH, not results. EXP-HVD-01: could the engine's hidden-variable trigger detect a
planted exogenous driver if the plumbing were fixed?

Status: PROPOSED. Written 2026-11-24. Only `--selftest` was run by the author.

The engine (scripts/hypothesis_engine.py, stage 6) uses
    residual_i = |beta_confidence_i - 0.5|,  beta = (1 + passed) / (2 + passed + failed)
and fires when mean|residual| >= 0.1 AND |pearson_r(residual, candidate)| > 0.5.
docs/hypothesis_engine.md records that no valid exogenous candidate exists today.

This sketch builds synthetic claim records whose pass/fail outcomes depend on a planted
exogenous series z (e.g. a per-date source-outage indicator) with effect size `beta_z`,
runs the engine's trigger with the engine's own `pearson_r`, and reports detection rate
(planted effect) and false-alarm rate (beta_z = 0) over many replicate topics. It also
reports the same for a signed residual (beta - 0.5) to expose the cost of the |.| fold.

Usage (from repo root):
    python3 experiments/sketches/hvd_planted_trigger.py --selftest
    python3 experiments/sketches/hvd_planted_trigger.py --n-claims 10 30 100 --effects 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "hypothesis_engine_readonly", REPO / "scripts" / "hypothesis_engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def synth_topic(rng: random.Random, n_claims: int, beta_z: float, tests_per_claim: int,
                base_logit: float, z_rate: float):
    """Per claim: z in {0,1}; each test passes with p = sigmoid(base + beta_z * z)."""
    z, beta = [], []
    for _ in range(n_claims):
        zi = 1.0 if rng.random() < z_rate else 0.0
        p = 1 / (1 + math.exp(-(base_logit + beta_z * zi)))
        passed = sum(rng.random() < p for _ in range(tests_per_claim))
        failed = tests_per_claim - passed
        z.append(zi)
        beta.append((1 + passed) / (2 + passed + failed))
    return z, beta


def trigger(eng, z, beta, signed=False):
    res = [(b - 0.5) if signed else abs(b - 0.5) for b in beta]
    mean_abs = sum(abs(r) for r in res) / len(res)
    r = eng.pearson_r(res, z)
    return (mean_abs >= 0.1 and abs(r) > 0.5), r


def run(n_claims_list, effects, reps, tests_per_claim, base_logit, z_rate, seed):
    eng = load_engine()
    rng = random.Random(seed)
    rows = []
    for n in n_claims_list:
        for e in effects:
            fired = {"folded": 0, "signed": 0}
            for _ in range(reps):
                z, beta = synth_topic(rng, n, e, tests_per_claim, base_logit, z_rate)
                for k in fired:
                    fired[k] += trigger(eng, z, beta, signed=(k == "signed"))[0]
            rows.append({"n_claims": n, "beta_z": e,
                         "fire_rate_folded": fired["folded"] / reps,
                         "fire_rate_signed": fired["signed"] / reps})
    return {"sketch": "EXP-HVD-01", "status": "SKETCH OUTPUT - not a pre-registered result",
            "reps": reps, "tests_per_claim": tests_per_claim, "base_logit": base_logit,
            "z_rate": z_rate, "seed": seed, "rows": rows}


def selftest() -> int:
    eng = load_engine()
    fails = []
    # engine pearson_r on a perfect line
    if abs(eng.pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) > 1e-12:
        fails.append("pearson_r(line) != 1")
    # degenerate candidate -> 0.0 (engine convention), so trigger cannot fire
    z, beta = [1.0] * 10, [0.9] * 10
    if trigger(eng, z, beta)[0]:
        fails.append("trigger fired on constant candidate")
    # synth_topic returns valid beta values
    _, b = synth_topic(random.Random(0), 50, 1.0, 5, 0.0, 0.5)
    if not all(0 < x < 1 for x in b):
        fails.append("beta outside (0,1)")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failures": fails}))
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-claims", type=int, nargs="+", default=[10, 30, 100])
    ap.add_argument("--effects", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--tests-per-claim", type=int, default=5)
    ap.add_argument("--base-logit", type=float, default=0.0)
    ap.add_argument("--z-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    json.dump(run(a.n_claims, a.effects, a.reps, a.tests_per_claim, a.base_logit,
                  a.z_rate, a.seed), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
