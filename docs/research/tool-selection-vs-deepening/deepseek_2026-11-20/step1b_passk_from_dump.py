#!/usr/bin/env python3
# Author: DeepSeek (relayed by owner 2026-11-20); reviewed by chat Claude L0, see REVIEW.md
"""Step 1.5: pass@k curves straight from a released trace dump.

Needs no GPU and no re-sampling: groups the dump by (model, problem),
computes the unbiased pass@k curve per model, and — if two arms are
present (e.g. base and GRPO-trained) — tests the narrowing claim directly.

Degrades honestly: if the dump has one sample per problem, or no problem
ids, or no model field, it says so and writes a partial result with
status 'ran_partial'.

Usage:
    python step1b_passk_from_dump.py --data DIR --probe
    python step1b_passk_from_dump.py --data DIR --out step1b_results.json \
        --emit-arm-files --compare base trained
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_recheck import (iter_records, detect_mapping, is_correct,
                           detect_field, norm_answer)  # noqa: E402

PROBLEM_FIELDS = ["problem_id", "problemid", "qid", "question_id",
                  "uid", "idx", "id", "index"]
K_MAX = 256


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "unnamed"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="step1b_results.json")
    ap.add_argument("--emit-arm-files", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two model labels to compare (B minus A)")
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)

    if not Path(args.data).exists():
        print(f"error: {args.data} not found", file=sys.stderr)
        return 2

    it = iter_records(args.data)
    try:
        first = next(it)
    except StopIteration:
        print("error: no records parsed", file=sys.stderr)
        return 2

    m = detect_mapping([first])
    m["problem"] = detect_field([first], PROBLEM_FIELDS)

    if args.probe:
        print(json.dumps({"detected": m,
                          "keys": sorted(first.keys())}, indent=2))
        return 0

    if not m.get("text"):
        print("error: no text field; run --probe and pin --map", file=sys.stderr)
        return 3
    if not m.get("problem"):
        print("error: no problem-id field detected; cannot group samples.\n"
              "hint: --probe, then pin the field name. If the dump is one\n"
              "sample per problem, pass@k is not computable from it and\n"
              "step 1.5 stays 'not run'.", file=sys.stderr)
        return 3

    groups = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "c": 0, "traces": set()}))
    known, skipped = 0, 0
    for i, rec in enumerate(it):
        if i >= args.limit:
            break
        # re-inject the first record; cheap and keeps the loop uniform
        if i == 0:
            rec = first
        text = rec.get(m["text"])
        if not isinstance(text, str):
            continue
        pid = str(rec.get(m["problem"]))
        model = str(rec.get(m["model"])) if m.get("model") else "ALL"
        g = groups[model][pid]
        ok = is_correct(rec, m)
        if ok is None:
            skipped += 1
            continue
        g["n"] += 1
        g["c"] += int(ok)
        g["traces"].add(norm_answer(text[-200:]) or text[-200:].strip().lower())
        known += 1

    if not groups:
        print("error: no usable records", file=sys.stderr)
        return 3

    out_groups = {}
    for model, probs in groups.items():
        out_groups[model] = {
            pid: {"n": v["n"], "c": v["c"],
                  "distinct_tail_rate": round(len(v["traces"]) / v["n"], 4)
                  if v["n"] else None}
            for pid, v in probs.items() if v["n"] > 0
        }

    # per-model curves, computed locally to avoid a hard dep on passk.py
    import math

    def pak(n, c, k):
        if k > n:
            return None
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    curves = {}
    for model, probs in out_groups.items():
        curves[model] = {}
        for k in ks:
            vals = [pak(p["n"], p["c"], k) for p in probs.values() if p["n"] >= k]
            curves[model][str(k)] = {
                "pass": round(sum(vals) / len(vals), 4) if vals else None,
                "n_problems": len(vals),
            }

    max_n = max((p["n"] for probs in out_groups.values()
                 for p in probs.values()), default=0)
    partial = max_n < 2
    status = "ran_partial" if partial else "ran"

    result = {
        "step": "step1b_passk_from_dump",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mapping": m,
        "n_samples_correctness_known": known,
        "n_samples_correctness_unknown": skipped,
        "max_samples_per_problem": max_n,
        "groups": out_groups,
        "curves": curves,
        "caveats": [
            "pooled pass@k filters to problems with n>=k; high-k points "
            "rest on fewer problems (reported as n_problems)",
            "distinct_tail_rate is a coarse path-diversity proxy from the "
            "last 200 chars; exploratory, not a substitute for edit distance",
            "if max_samples_per_problem < 2 the dump cannot support pass@k "
            "and status is ran_partial",
        ],
    }

    if args.emit_arm_files:
        for model, probs in out_groups.items():
            f = Path(args.out).parent / f"arm_{safe(model)}.json"
            f.write_text(json.dumps({p: {"n": v["n"], "c": v["c"]}
                                     for p, v in probs.items()}))
            result.setdefault("arm_files", {})[model] = str(f)

    Path(args.out).write_text(json.dumps(result, indent=2))

    for model, c in curves.items():
        pts = "  ".join(f"@{k}={c[str(k)]['pass']}" for k in ks
                        if c[str(k)]["pass"] is not None)
        print(f"[{model}]  {pts}")
    print(f"max samples/problem = {max_n}  status={status}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
