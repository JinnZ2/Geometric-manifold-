#!/usr/bin/env python3
"""EXP-HVD-04 post-hoc check (2026-11-27, NOT pre-registered; written after seeing raw.json).

The pre-registered Spearman rho pools strength 0 (brace or tether absent) with strengths > 0
(present). A step at "present vs absent" can carry rho on its own. This recomputes rho over
strengths > 0 only, and the ratio of the median residual at s = 0.1 to s = 0.01.
Output: posthoc.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import write_json  # noqa: E402

raw = json.loads((HERE / "raw.json").read_text())
out = []
for c in raw["summary"]:
    sel = [r for r in raw["rows"] if r["kind"] == c["kind"] and r["loc"] == c["loc"] and r["k"] == c["k"] and r["s"] > 0]
    rho = spearmanr([r["s"] for r in sel], [r["unexplained"] for r in sel]).statistic
    m = c["median_by_s"]
    out.append({"kind": c["kind"], "loc": c["loc"], "k": c["k"], "rho_s_gt_0": float(rho),
                "median_ratio_s0.1_over_s0.01": m["0.1"] / m["0.01"],
                "median_ratio_s1_over_s0.01": m["1.0"] / m["0.01"]})
    print(out[-1])
write_json(HERE / "posthoc.json", out)
