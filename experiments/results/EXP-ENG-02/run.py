#!/usr/bin/env python3
"""EXP-ENG-02 run (2026-11-27): ablation of the engine's corroboration heuristic.

Uses experiments/sketches/eng_lexical_audit.py unchanged (its `run()` scores conditions
A, A1, B, C, D with the engine's own corroboration(), imported read-only). Condition C is
the only seeded one; it is run for seeds 0-9 as pre-registered. Output: raw.json.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO, Timer, environment, write_json  # noqa: E402


def load_sketch():
    spec = importlib.util.spec_from_file_location(
        "eng_lexical_audit", REPO / "experiments" / "sketches" / "eng_lexical_audit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sk = load_sketch()
    out = {"experiment": "EXP-ENG-02", "environment": environment(), "runs": {}}
    with Timer() as t:
        for seed in range(10):
            r = sk.run(REPO / "data", seed, None)
            if seed == 0:
                out["deterministic_conditions"] = {k: v for k, v in r["conditions"].items()
                                                   if not k.startswith("C_")}
                out["n_claims"], out["n_findings"] = r["n_claims"], r["n_findings"]
            out["runs"][str(seed)] = {"cross_topic_map": r["cross_topic_map"],
                                      "C_cross_topic_null": r["conditions"]["C_cross_topic_null"]}
    out["runtime_seconds"] = t.seconds
    write_json(HERE / "raw.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
