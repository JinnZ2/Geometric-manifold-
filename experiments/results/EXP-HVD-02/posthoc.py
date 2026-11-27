#!/usr/bin/env python3
"""EXP-HVD-02 post-hoc check (2026-11-27, NOT pre-registered; written after seeing raw.json).

Within a topic, same_topic_at is a monotone function of retr_days (rank of retrieval time),
so their joint coefficients in the main model are not separately interpretable. This fits
each of the two alone (plus the other pre-registered covariates and topic fixed effects).
Output: posthoc.json.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("hvd02", HERE / "run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

d = m.standardise(m.load())
base = ["abs_tokens", "topic_tokens", "pub_days", "src_semantic_scholar", "lane_none"]
out = {}
for keep in ("same_topic_at", "retr_days"):
    rhs = " + ".join(base + [keep, "C(topic)"])
    lg = m.fit_logit(d, rhs)
    pc = m.fit_nb(d, rhs)
    out[keep] = {"rhs": rhs, "logit": m.coef_table(lg)[keep], "logit_converged": bool(lg.mle_retvals["converged"]),
                 "count_model": m.coef_table(pc)[keep], "count_fallback": pc.fallback}
    print(keep, out[keep]["logit"], out[keep]["count_model"])
write_json(HERE / "posthoc.json", out)
