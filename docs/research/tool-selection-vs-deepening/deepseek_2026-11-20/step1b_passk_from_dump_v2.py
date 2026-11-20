#!/usr/bin/env python3
# v2 of step1b_passk_from_dump.py (DeepSeek, relayed by owner 2026-11-20). Changes by Claude
# Code, 2026-11-20, listed in REVIEW.md section 6. The original beside this file is unchanged.
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
    python step1b_passk_from_dump_v2.py --data DIR --map mapping.json
    python step1b_passk_from_dump_v2.py --self-test
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_recheck_v2 import (iter_records, detect_mapping, is_correct,
                           detect_field, norm_answer)  # noqa: E402

PROBLEM_FIELDS = ["problem_id", "problemid", "qid", "question_id",
                  "uid", "idx", "id", "index"]
K_MAX = 256


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "unnamed"


def collect(records, m, limit=None):
    """Group records by (model, problem). Every record handed in is read;
    none is replaced. Returns groups, known, skipped, records_seen,
    limit_applied."""
    groups = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "c": 0, "traces": set()}))
    known, skipped, seen = 0, 0, 0
    limit_applied = False
    for i, rec in enumerate(records):
        if limit and i >= limit:
            limit_applied = True
            break
        seen += 1
        text = rec.get(m["text"])
        if not isinstance(text, str):
            continue
        pid = str(rec.get(m["problem"]))
        model = str(rec.get(m["model"])) if m.get("model") else "ALL"
        ok = is_correct(rec, m)
        if ok is None:
            skipped += 1
            continue
        g = groups[model][pid]
        g["n"] += 1
        g["c"] += int(ok)
        g["traces"].add(norm_answer(text[-200:]) or text[-200:].strip().lower())
        known += 1
    return groups, known, skipped, seen, limit_applied


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "d.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for i in range(3):
                fh.write(json.dumps({"trace": "t%d answer 42" % i, "gold": "42",
                                     "pred": "42", "problem_id": "p1",
                                     "model": "m"}) + "\n")
        out = Path(d) / "out.json"
        rc = main(["--data", str(f), "--out", str(out)])
        assert rc == 0, rc
        res = json.loads(out.read_text())
        g = res["groups"]["m"]["p1"]
        assert g["n"] == 3 and g["c"] == 3, g          # 3 of 3 counted
        assert res["n_samples_correctness_known"] == 3
        assert res["records_seen"] == 3 and res["limit_applied"] is False
        assert res["max_samples_per_problem"] == 3
        rc = main(["--data", str(f), "--out", str(out), "--limit", "2"])
        res = json.loads(out.read_text())
        assert res["records_seen"] == 2 and res["limit_applied"] is True, res
        assert res["groups"]["m"]["p1"]["n"] == 2
    print(json.dumps({"self_test": "PASS", "counted": "3 of 3"}))
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--map", help="json file pinning field names "
                    "(overrides auto-detection key by key)")
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
    if args.map:
        m.update(json.loads(Path(args.map).read_text()))

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

    def chain(first_rec, rest):
        yield first_rec
        yield from rest

    groups, known, skipped, records_seen, limit_applied = collect(
        chain(first, it), m, limit=args.limit)

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
        "records_seen": records_seen,
        "limit": args.limit,
        "limit_applied": limit_applied,
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
    print(f"records seen = {records_seen}  limit={args.limit}  "
          f"limit_applied={limit_applied}")
    print(f"max samples/problem = {max_n}  status={status}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
