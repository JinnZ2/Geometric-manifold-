#!/usr/bin/env python3
"""Step 1.5: pass@k curves straight from the released trace dump.

Groups the dump by (model, task, step, temperature, problem), computes the
unbiased pass@k curve per config, and — because the release holds GRPO
checkpoints INCLUDING step 0 (the initialization) — contrasts step 0 vs the
final step per (model, task, temperature). That is the dump-supported form
of the narrowing question.

OBSERVED data limits (rows API, 2026-11-20):
  * G = 8 samples per (problem, step, temperature) -> k is data-capped at 8.
    pass@256 lives in step 2's own sampling, not here.
  * No separate base model: models are GRPO-1.5B / GRPO-7B / GRPO-Llama8B.
    "Base" here means step 0 (initialization under the RL prompt template).
  * pass@k over a config mixes no training states; pooling across steps
    would, so it is never done here.

Repair record (vs the first draft):
  R5  the draft silently dropped the second record (re-injected `first`
      over the iterator's head). Fixed with itertools.chain.
  R6  the draft grouped by (model, problem), pooling all steps and
      temperatures into one contaminated pass@k. Grouping is now per
      (model, task, step, temperature, problem).
  R7  the draft's PROBLEM_FIELDS missed the observed id fields (problem /
      example_id / enumeration); detection is now shared with
      step1_recheck.problem_id_of.

Usage:
    python step1b_passk_from_dump.py --self-test
    python step1b_passk_from_dump.py --data DIR --probe
    python step1b_passk_from_dump.py --data DIR --out step1b_results.json \
        --emit-arm-files
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_recheck import (iter_records, detect_layout, pass_records,  # noqa: E402
                           problem_id_of, rate)
from passk import pass_at_k, paired_delta  # noqa: E402

KS = [1, 2, 4, 8]  # data-capped: G = 8 samples per config (OBSERVED)


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "unnamed"


def build_groups(records, limit=0):
    """groups[model][task][step][temp][problem] = {"n","c"} from pass1."""
    groups = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "c": 0})))))
    known = skipped = n_rows = 0
    for i, rec in enumerate(records):
        if limit and i >= limit:
            break
        n_rows += 1
        passes = list(pass_records(rec))
        if not passes:
            skipped += 1
            continue
        p1 = passes[0]
        if p1["correct"] is None or p1["problem"] is None:
            skipped += 1
            continue
        g = groups[str(p1["model"])][str(p1["task"])][str(p1["step"])][
            str(p1["temperature"])][p1["problem"]]
        g["n"] += 1
        g["c"] += int(p1["correct"])
        known += 1
    return groups, n_rows, known, skipped


def curve_of(probs, ks):
    out = {}
    for k in ks:
        vals = [pass_at_k(p["n"], p["c"], k) for p in probs.values()
                if p["n"] >= k]
        out[str(k)] = {"pass": rate(sum(vals), len(vals)) if vals else None,
                       "n_problems": len(vals)}
    return out


def step0_vs_final(groups, ks):
    """Per (model, task, temp): paired delta pass@k(final) - pass@k(step 0),
    problem-level bootstrap CI."""
    contrasts = {}
    for model, tasks in sorted(groups.items()):
        for task, steps in sorted(tasks.items()):
            step_ids = sorted(steps.keys(), key=lambda s: float(s))
            if len(step_ids) < 2:
                continue
            s0, sf = step_ids[0], step_ids[-1]
            for temp in sorted(steps[s0].keys(), key=lambda t: float(t)):
                A = steps[s0].get(temp)
                B = steps[sf].get(temp)
                if not A or not B:
                    continue
                label = f"{model} | {task} | T={temp} | step{sf}-step0"
                contrasts[label] = {
                    "step0": s0, "step_final": sf,
                    "n_problems_step0": len(A), "n_problems_final": len(B),
                    "delta": {str(k): paired_delta(A, B, k)
                              for k in ks},
                }
    return contrasts


def self_test():
    """Planted narrowing: final step collapses each problem's solve
    probability to 0 or 1. pass@1 must rise, pass@8 must fall."""
    import random as _r

    rng = _r.Random(3)
    ps = [rng.uniform(0.1, 0.9) for _ in range(400)]  # one p per problem,
    rows = []                                         # shared across steps
    for step, collapse in ((0, False), (950, True)):
        for pi in range(400):
            p = ps[pi]
            if collapse:
                # collapsed tail: pass@1 0.50 -> 0.625, pass@8 ~0.9 -> 0.625
                p = 1.0 if p > 0.4 else 0.0
            for si in range(8):
                c = rng.random() < p
                rows.append({
                    "model": "GRPO-synthetic", "task": "math",
                    "step": step, "temperature": 0.7, "sample_idx": si,
                    "problem": f"q{pi}",
                    "pass1": json.dumps({
                        "output": "solution text",
                        "entropy": 1.0,
                        "is_correct_pred": c,
                        "has_reconsider_cue": False,
                        "_shift_prefilter_markers": [],
                        "shift_in_reasoning_v1": False,
                    }),
                    "pass2": None,
                })
    layout = detect_layout(rows[:50])
    assert layout == "passpair", layout
    groups, n_rows, known, skipped = build_groups(rows)
    assert n_rows == 6400 and known == 6400, (n_rows, known)
    contrasts = step0_vs_final(groups, KS)
    assert len(contrasts) == 1, list(contrasts)
    d = next(iter(contrasts.values()))["delta"]
    assert d["1"]["delta"] > 0 and d["1"]["ci95"][0] > 0, d["1"]
    assert d["8"]["delta"] < 0 and d["8"]["ci95"][1] < 0, d["8"]
    print(json.dumps({"self_test": "PASS",
                      "planted_narrowing": {
                          "d_pass1": d["1"]["delta"],
                          "d_pass8": d["8"]["delta"]}},
                     indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="step1b_results.json")
    ap.add_argument("--emit-arm-files", action="store_true",
                    help="write step0 / final-step arm files per "
                         "(model, task, temp) for passk.py")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.data:
        print("error: --data required (or --self-test)", file=sys.stderr)
        return 2
    if not Path(args.data).exists():
        print(f"error: {args.data} not found", file=sys.stderr)
        return 2

    it = iter_records(args.data)
    try:
        first = next(it)
    except StopIteration:
        print("error: no records parsed", file=sys.stderr)
        return 2

    if args.probe:
        sample = list(chain([first], (r for _, r in zip(range(199), it))))
        layout = detect_layout(sample)
        print(json.dumps({"layout": layout,
                          "problem_id": problem_id_of(first),
                          "keys": sorted(first.keys())}, indent=2))
        if layout != "passpair":
            print("note: step1b currently supports the passpair layout "
                  "(the OBSERVED dump schema)", file=sys.stderr)
            return 3
        return 0

    layout = detect_layout(list(chain([first], (r for _, r in
                                                zip(range(199), iter_records(args.data))))))
    if layout != "passpair":
        print("error: step1b supports the passpair layout; got "
              f"'{layout}'. If the raw artifacts are flat, pin fields and "
              "extend build_groups.", file=sys.stderr)
        return 3

    groups, n_rows, known, skipped = build_groups(iter_records(args.data),
                                                  limit=args.limit)
    if not groups:
        print("error: no usable records", file=sys.stderr)
        return 3

    max_n = max((p["n"] for tasks in groups.values()
                 for steps in tasks.values()
                 for temps in steps.values()
                 for probs in temps.values()
                 for p in probs.values()), default=0)
    partial = max_n < 2
    status = "ran_partial" if partial else "ran"

    curves = {}
    for model, tasks in sorted(groups.items()):
        for task, steps in sorted(tasks.items()):
            for step, temps in sorted(steps.items(), key=lambda kv: float(kv[0])):
                for temp, probs in sorted(temps.items(), key=lambda kv: float(kv[0])):
                    curves[f"{model} | {task} | step {step} | T={temp}"] = \
                        curve_of(probs, KS)

    contrasts = step0_vs_final(groups, KS)

    result = {
        "step": "step1b_passk_from_dump",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "n_pass1_used": known,
        "n_rows_skipped": skipped,
        "ks": KS,
        "k_cap_reason": "G=8 samples per (problem, step, temperature) in the "
                        "released dump (OBSERVED); pass@256 belongs to step 2",
        "max_samples_per_config": max_n,
        "base_arm_note": "no base model in the release; step 0 = "
                         "initialization under the RL prompt template",
        "curves": curves,
        "step0_vs_final": contrasts,
        "caveats": [
            "pass@k per (model, task, step, temperature); no pooling across "
            "training states or temperatures",
            "pooled pass@k filters to problems with n>=k; high-k points rest "
            "on fewer problems (reported as n_problems)",
            "if max_samples_per_config < 2 the dump cannot support pass@k "
            "and status is ran_partial",
        ],
    }

    if args.emit_arm_files:
        outdir = Path(args.out).parent
        result["arm_files"] = {}
        for model, tasks in sorted(groups.items()):
            for task, steps in sorted(tasks.items()):
                step_ids = sorted(steps.keys(), key=lambda s: float(s))
                if len(step_ids) < 2:
                    continue
                for tag, sid in (("step0", step_ids[0]),
                                 (f"step{step_ids[-1]}", step_ids[-1])):
                    for temp, probs in steps[sid].items():
                        f = outdir / (f"arm_{safe(model)}_{safe(task)}_"
                                      f"{tag}_T{safe(temp)}.json")
                        f.write_text(json.dumps(
                            {p: v for p, v in probs.items()}))
                        result["arm_files"][
                            f"{model}|{task}|{tag}|T={temp}"] = str(f)

    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"rows={n_rows} used={known} skipped={skipped} "
          f"max_n={max_n} status={status}")
    for label, c in contrasts.items():
        line = "  ".join(
            f"@{k}={c['delta'][k]['delta']}"
            for k in map(str, KS) if c["delta"][k]["delta"] is not None)
        print(f"[{label}]  pass@k(final-step0): {line}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
