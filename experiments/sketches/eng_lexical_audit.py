#!/usr/bin/env python3
"""SKETCH, not results. EXP-ENG-02: what does the engine's `corroboration()` respond to?

Status: PROPOSED. Written 2026-11-24 against main at 02320b51. Nothing here has been
run on data/ by the author of this file; only `--selftest` (synthetic strings) was run.

It imports `scripts/hypothesis_engine.py` read-only and re-scores every (claim, finding)
pair under four conditions, using the engine's own `corroboration()`:

  A  as_is         claim text as stored, findings from the claim's own topic (all lanes)
  A1 as_is_primary as A, findings restricted to lane != cross_domain
  B  no_prefix     claim text with the template prefix "On topic <topic>, " and the
                   " reports:" connector removed, own-topic findings
  C  cross_topic   claim text as stored, findings from a different topic (seeded
                   derangement of topics) -- a null: these findings were not retrieved
                   for the claim's topic
  D  prefix_only   claim text replaced by "On topic <topic>," alone, own-topic findings

Output: per topic and condition, counts of +1 / -1 / 0 verdicts over pairs, and the
number of claims that would collect >= 3 passes in a single pass over the findings
(the engine's "survived" threshold). Output is JSON on stdout. The script states no
verdict; the pre-registered reading is in experiments/engine-evidence-audit.md.

Usage (from repo root):
    python3 experiments/sketches/eng_lexical_audit.py --selftest
    python3 experiments/sketches/eng_lexical_audit.py --data-dir data --seed 0 > out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "hypothesis_engine_readonly", REPO / "scripts" / "hypothesis_engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


PREFIX_RE = re.compile(r"^On topic (?P<topic>.*?), (?P<body>.*)$", re.S)


def strip_prefix(text: str, topic: str) -> str:
    """Remove 'On topic <topic>, ' and the ' reports:' connector; keep title + sentence."""
    head = f"On topic {topic}, "
    body = text[len(head):] if text.startswith(head) else text
    return body.replace(" reports: ", " ", 1)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derange(items: list[str], rng: random.Random) -> dict[str, str]:
    """Map each topic to a different topic (seeded)."""
    if len(items) < 2:
        raise ValueError("need >= 2 topics for a cross-topic null")
    while True:
        perm = items[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def score(eng, claims, by_topic, text_fn, topic_fn, lane_filter=None):
    out = defaultdict(lambda: {"pos": 0, "neg": 0, "zero": 0, "pairs": 0,
                               "claims": 0, "claims_ge3_pass": 0})
    for c in claims:
        topic = c["scope"].get("topic", "")
        ctext = text_fn(c, topic)
        rows = by_topic.get(topic_fn(topic), [])
        passes = 0
        rec = out[topic]
        rec["claims"] += 1
        for r in rows:
            if r.get("hash") == c["id"]:
                continue
            if lane_filter and not lane_filter(r):
                continue
            v = eng.corroboration(ctext, f"{r.get('title', '')} {r.get('abstract', '')}")
            rec["pairs"] += 1
            if v > 0:
                rec["pos"] += 1
                passes += 1
            elif v < 0:
                rec["neg"] += 1
            else:
                rec["zero"] += 1
        if passes >= 3:
            rec["claims_ge3_pass"] += 1
    return dict(out)


def run(data_dir: Path, seed: int, max_claims: int | None) -> dict:
    eng = load_engine()
    claims = json.loads((data_dir / "claim_tree.json").read_text(encoding="utf-8"))["claims"]
    if max_claims:
        claims = claims[:max_claims]
    findings = read_jsonl(data_dir / "findings_log.jsonl")
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in findings:
        by_topic[r.get("topic", "")].append(r)
    topics = sorted(by_topic)
    cross = derange(topics, random.Random(seed))

    def same(t):
        return t

    conds = {
        "A_as_is": score(eng, claims, by_topic, lambda c, t: c["text"], same),
        "A1_as_is_primary_lane": score(eng, claims, by_topic, lambda c, t: c["text"], same,
                                       lane_filter=lambda r: r.get("lane") != "cross_domain"),
        "B_no_prefix": score(eng, claims, by_topic, lambda c, t: strip_prefix(c["text"], t), same),
        "C_cross_topic_null": score(eng, claims, by_topic, lambda c, t: c["text"],
                                    lambda t: cross.get(t, t)),
        "D_prefix_only": score(eng, claims, by_topic, lambda c, t: f"On topic {t},", same),
    }
    return {"sketch": "EXP-ENG-02", "status": "SKETCH OUTPUT - not a pre-registered result",
            "seed": seed, "n_claims": len(claims), "n_findings": len(findings),
            "cross_topic_map": cross, "conditions": conds}


def selftest() -> int:
    eng = load_engine()
    fails = []
    # 1. unrelated strings -> 0
    if eng.corroboration("graphene Mott transition heavy fermion",
                         "recursive synthetic data model collapse") != 0:
        fails.append("unrelated strings did not score 0")
    # 2. related, same polarity -> +1
    if eng.corroboration("loss landscape flatness generalization improves",
                         "flatness of loss landscape improves generalization") != 1:
        fails.append("related positive pair did not score +1")
    # 3. related, opposite polarity -> -1
    if eng.corroboration("loss landscape flatness generalization improves",
                         "loss landscape flatness does not improve generalization; fails") != -1:
        fails.append("related opposite pair did not score -1")
    # 4. prefix stripping
    t = "On topic X and Y, Some Title reports: First sentence."
    if strip_prefix(t, "X and Y") != "Some Title First sentence.":
        fails.append(f"strip_prefix wrong: {strip_prefix(t, 'X and Y')!r}")
    # 5. derangement has no fixed points
    m = derange(["a", "b", "c", "d"], random.Random(1))
    if any(k == v for k, v in m.items()):
        fails.append("derangement has a fixed point")
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failures": fails}))
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-claims", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    json.dump(run(Path(a.data_dir), a.seed, a.max_claims), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
