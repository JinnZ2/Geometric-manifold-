#!/usr/bin/env python3
"""EXP-HVD-01 run (2026-11-27): power and false-alarm rate of the engine's hidden-variable
trigger on planted data.

Full pre-registered grid: beta_z in {0, 0.5, 1, 2}, tests per claim in {3, 5, 10}, claims per
topic in {10, 30, 100, 300}, base pass logit in {-1, 0, 1}, z rate 0.3, 500 replicate topics
per cell. Uses experiments/sketches/hvd_planted_trigger.py (synth_topic, trigger) unchanged,
which calls the engine's own pearson_r (imported read-only). Each cell has its own seeded
random.Random (seed = cell index) so cells are reproducible independently.
Output: raw.json (fire counts and Wilson 95% CIs per cell, folded and signed residuals).
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, environment, write_json  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def main():
    sk = _load("hvd_planted_trigger_sketch", REPO / "experiments/sketches/hvd_planted_trigger.py")
    eng = sk.load_engine()
    REPS = 500
    grid = list(itertools.product([0.0, 0.5, 1.0, 2.0], [3, 5, 10], [10, 30, 100, 300],
                                  [-1.0, 0.0, 1.0]))
    rows = []
    with Timer() as t:
        for idx, (bz, tpc, nc, base) in enumerate(grid):
            rng = random.Random(idx)
            k = {"folded": 0, "signed": 0}
            for _ in range(REPS):
                z, beta = sk.synth_topic(rng, nc, bz, tpc, base, 0.3)
                for key in k:
                    k[key] += sk.trigger(eng, z, beta, signed=(key == "signed"))[0]
            rows.append({"cell": idx, "beta_z": bz, "tests_per_claim": tpc, "n_claims": nc,
                         "base_logit": base, "reps": REPS,
                         "fires_folded": k["folded"], "fires_signed": k["signed"],
                         "rate_folded": k["folded"] / REPS, "rate_signed": k["signed"] / REPS,
                         "ci_folded": wilson(k["folded"], REPS),
                         "ci_signed": wilson(k["signed"], REPS)})
    write_json(HERE / "raw.json", {"experiment": "EXP-HVD-01", "environment": environment(),
                                   "z_rate": 0.3, "runtime_seconds": t.seconds, "rows": rows})
    print("done", t.seconds)


if __name__ == "__main__":
    main()
