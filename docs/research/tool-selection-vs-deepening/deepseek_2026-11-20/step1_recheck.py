#!/usr/bin/env python3
# Author: DeepSeek (relayed by owner 2026-11-20); reviewed by chat Claude L0, see REVIEW.md
"""Step 1 independent re-check of arXiv 2601.00514v2 (Princeton).

Re-derives shift rate, accuracy-given-shift, and entropy-gated gain from the
released trace dump, stdlib only, without the authors' code in the loop.

Usage:
    python step1_recheck.py --self-test
    python step1_recheck.py --data PATH --probe
    python step1_recheck.py --data PATH --out step1_results.json
    python step1_recheck.py --data PATH --map mapping.json --limit 5000

Exit codes: 0 ran, 2 input missing, 3 schema unmapped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HF_DATASET = "od2961/illusion-of-reasoning-main-traces"
GH_REPO = "humans-and-machines/Illusion-of-Reasoning"

# --- field detection -------------------------------------------------------

TEXT_FIELDS = ["trace", "reasoning", "reasoning_content", "response",
               "text", "output", "completion", "content"]
GOLD_FIELDS = ["gold", "answer", "gt", "target", "solution", "ground_truth"]
PRED_FIELDS = ["prediction", "pred", "extracted_answer", "model_answer",
               "final_answer"]
CORRECT_FIELDS = ["correct", "is_correct", "acc", "score"]
ENTROPY_FIELDS = ["entropies", "entropy", "token_entropies",
                  "token_entropy", "logprobs"]
MODEL_FIELDS = ["model", "model_name", "model_id"]
TASK_FIELDS = ["dataset", "task", "source", "benchmark"]

# --- shift markers ---------------------------------------------------------
# NARROW is the primary definition (conservative: fewer false positives).
NARROW = [r"\bwait\b", r"\bhmm+\b", r"\bactually\b",
          r"\bon second thought\b", r"\blet me reconsider\b",
          r"\blet me re-?check\b"]
WIDE = NARROW + [r"\boh\b", r"\bhold on\b", r"\bthat'?s wrong\b",
                 r"\bi made (?:a|an) (?:mistake|error)\b",
                 r"\bbut that'?s not right\b", r"\balternatively\b",
                 r"\breconsider\b", r"\bdouble[- ]check\b"]


def compile_markers(patterns):
    return re.compile("|".join(patterns), re.IGNORECASE)


# --- io --------------------------------------------------------------------

def _flatten(obj):
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        for k in ("rows", "data", "traces", "records", "examples"):
            if isinstance(obj.get(k), list):
                for x in obj[k]:
                    if isinstance(x, dict):
                        yield x
                return
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            for v in vals:
                yield v


def iter_records(path):
    p = Path(path)
    if not p.exists():
        return
    if p.is_dir():
        files = sorted(f for f in p.rglob("*")
                       if f.suffix in (".json", ".jsonl")
                       or f.name.endswith(".jsonl.gz"))
    else:
        files = [p]
    for f in files:
        opener = gzip.open if f.name.endswith(".gz") else open
        try:
            if f.suffix == ".jsonl" or f.name.endswith(".jsonl.gz"):
                with opener(f, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            elif f.suffix == ".json":
                with opener(f, "rt", encoding="utf-8") as fh:
                    obj = json.load(fh)
                for rec in _flatten(obj):
                    yield rec
            else:
                print(f"[skip] unsupported (parquet needs an extra dep): {f}",
                      file=sys.stderr)
        except OSError as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)


def detect_field(sample, candidates):
    if not sample:
        return None
    for k in candidates:
        hits = sum(1 for r in sample if r.get(k) is not None)
        if hits >= 0.8 * len(sample):
            return k
    return None


def detect_mapping(sample):
    return {
        "text": detect_field(sample, TEXT_FIELDS),
        "gold": detect_field(sample, GOLD_FIELDS),
        "pred": detect_field(sample, PRED_FIELDS),
        "correct": detect_field(sample, CORRECT_FIELDS),
        "entropy": detect_field(sample, ENTROPY_FIELDS),
        "model": detect_field(sample, MODEL_FIELDS),
        "task": detect_field(sample, TASK_FIELDS),
    }


# --- answer normalisation --------------------------------------------------

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_answer(s):
    if s is None:
        return None
    s = str(s).strip()
    m = _BOXED.search(s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return "num:" + repr(float(s))
    except ValueError:
        return "str:" + s.lower()


def is_correct(rec, m):
    ck = m.get("correct")
    if ck and rec.get(ck) is not None:
        v = rec[ck]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v) >= 0.5
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "correct")
        return None
    gk, pk = m.get("gold"), m.get("pred")
    if not gk or not pk:
        return None
    g, p = norm_answer(rec.get(gk)), norm_answer(rec.get(pk))
    if g is None or p is None:
        return None
    return g == p


def entropies_of(rec, m):
    ek = m.get("entropy")
    if not ek:
        return None
    v = rec.get(ek)
    if not isinstance(v, list) or not v:
        return None
    out = []
    for x in v:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, dict):
            for kk in ("entropy", "e", "value"):
                if isinstance(x.get(kk), (int, float)):
                    out.append(float(x[kk]))
                    break
    return out or None


# --- statistics ------------------------------------------------------------

def percentile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def boot_ci(flags, n_boot=2000, alpha=0.05, seed=0):
    flags = [bool(f) for f in flags if f is not None]
    if len(flags) < 2:
        return None
    rng = random.Random(seed)
    N = len(flags)
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(N)] for _ in range(N))
        means.append(s / N)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


# --- main computation ------------------------------------------------------

def run(records, mapping, limit=None):
    narrow = compile_markers(NARROW)
    wide = compile_markers(WIDE)

    kept = []
    for i, rec in enumerate(records):
        if limit and i >= limit:
            break
        kept.append(rec)

    if not kept:
        return None, "no records"

    m = dict(mapping)
    if not m.get("text"):
        m = detect_mapping(kept[:200])
    if not m.get("text"):
        return None, "schema unmapped: no text field detected"

    # pass 1: entropies -> global p90 threshold
    all_ent = []
    for rec in kept:
        e = entropies_of(rec, m)
        if e:
            all_ent.extend(e)
    gate = percentile(all_ent, 0.90) if all_ent else None

    # pass 2: metrics
    n = 0
    n_shift_narrow = n_shift_wide = 0
    by_shift = {True: [0, 0], False: [0, 0]}   # [correct, total]
    gated = {"spike_shift": [0, 0], "spike_noshift": [0, 0]}
    per_model = defaultdict(lambda: {"n": 0, "shifted": 0})
    shift_flags = []

    for rec in kept:
        text = rec.get(m["text"])
        if not isinstance(text, str):
            continue
        n += 1
        s_narrow = bool(narrow.search(text))
        s_wide = bool(wide.search(text))
        n_shift_narrow += s_narrow
        n_shift_wide += s_wide
        shift_flags.append(s_narrow)

        ok = is_correct(rec, m)
        if ok is not None:
            by_shift[s_narrow][0] += int(ok)
            by_shift[s_narrow][1] += 1

        if m.get("model"):
            mk = str(rec.get(m["model"]))
            per_model[mk]["n"] += 1
            per_model[mk]["shifted"] += int(s_narrow)

        if gate is not None and ok is not None:
            e = entropies_of(rec, m)
            if e and max(e) >= gate:
                key = "spike_shift" if s_narrow else "spike_noshift"
                gated[key][0] += int(ok)
                gated[key][1] += 1

    def rate(num, den):
        return round(num / den, 4) if den else None

    shift_rate = rate(n_shift_narrow, n)
    shift_rate_wide = rate(n_shift_wide, n)
    acc_shift = rate(*by_shift[True])
    acc_noshift = rate(*by_shift[False])
    gated_gain = None
    if gated["spike_shift"][1] and gated["spike_noshift"][1]:
        gated_gain = round(rate(*gated["spike_shift"])
                           - rate(*gated["spike_noshift"]), 4)

    metrics = {
        "shift_rate": {
            "status": "OBSERVED",
            "value": shift_rate,
            "ci95": boot_ci(shift_flags),
            "n": n,
            "markers": "narrow",
            "sensitivity_wide": shift_rate_wide,
            "source": "this script; primary definition = NARROW marker set",
        },
        "accuracy_given_shift": {
            "status": "OBSERVED" if acc_shift is not None else "not run",
            "with_shift": acc_shift,
            "without_shift": acc_noshift,
            "n_with": by_shift[True][1],
            "n_without": by_shift[False][1],
            "source": "this script",
        },
        "entropy_gated_gain": {
            "status": "OBSERVED" if gated_gain is not None else "not run",
            "value": gated_gain,
            "gate_p90": gate,
            "n_spike_shift": gated["spike_shift"][1],
            "n_spike_noshift": gated["spike_noshift"][1],
            "source": "this script; trace-level gate, positional alignment "
                      "NOT attempted",
        },
    }

    per_model_out = {
        k: {"n": v["n"],
            "shift_rate": round(v["shifted"] / v["n"], 4) if v["n"] else None}
        for k, v in sorted(per_model.items())
    }

    result = {
        "step": "step1_recheck",
        "status": "ran",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"hf": HF_DATASET, "github": GH_REPO},
        "n_records": n,
        "mapping": m,
        "markers": {"narrow": NARROW, "wide": WIDE},
        "metrics": metrics,
        "per_model": per_model_out,
        "caveats": [
            "shift detection is marker-based, not the authors' classifier",
            "trace-level entropy gate: 'spike somewhere and shift somewhere', "
            "not 'shift caused by spike'",
            "correctness falls back to normalized gold/pred string match "
            "when no `correct` field is present",
        ],
    }
    return result, None


# --- self-test -------------------------------------------------------------

def self_test():
    rng = random.Random(7)
    recs = []
    for i in range(400):
        shifted = rng.random() < 0.10
        body = "First attempt. " + ("Wait, that is wrong. " if shifted else "")
        body += "So the answer is 42."
        base_p = 0.55 + (0.10 if shifted else 0.0)
        correct = rng.random() < base_p
        rec = {
            "trace": body,
            "gold": "42",
            "pred": "42" if correct else "41",
            "model": "synthetic-A" if i % 2 else "synthetic-B",
            "entropies": [round(rng.random() * 3, 3) for _ in range(8)],
        }
        recs.append(rec)
    res, err = run(recs)
    assert err is None, err
    sr = res["metrics"]["shift_rate"]
    assert 0.05 <= sr["value"] <= 0.20, sr
    assert res["metrics"]["accuracy_given_shift"]["n_with"] > 10
    assert res["status"] == "ran"
    print(json.dumps({
        "self_test": "PASS",
        "shift_rate": sr["value"],
        "acc_given_shift": res["metrics"]["accuracy_given_shift"],
    }, indent=2))
    return 0


# --- cli -------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="file or directory of traces")
    ap.add_argument("--out", default="step1_results.json")
    ap.add_argument("--map", help="json file pinning field names")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--probe", action="store_true",
                    help="print schema of first record and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.data:
        print("error: --data required (or --self-test)", file=sys.stderr)
        return 2
    if not Path(args.data).exists():
        print(f"error: {args.data} does not exist.\n"
              f"fetch with:\n"
              f"  pip install huggingface_hub\n"
              f"  huggingface-cli download {HF_DATASET} --repo-type dataset "
              f"--local-dir {args.data}\n"
              f"or clone https://github.com/{GH_REPO}",
              file=sys.stderr)
        return 2

    it = iter_records(args.data)
    try:
        first = next(it)
    except StopIteration:
        print("error: no records parsed from --data", file=sys.stderr)
        return 2

    if args.probe:
        print(json.dumps({
            "first_record_keys": sorted(first.keys()),
            "detected_mapping": detect_mapping([first]),
            "sample_values": {k: str(v)[:120] for k, v in first.items()},
        }, indent=2))
        return 0

    def chain(first_rec, rest):
        yield first_rec
        yield from rest

    mapping = {}
    if args.map:
        mapping = json.loads(Path(args.map).read_text())

    res, err = run(chain(first, it), mapping, limit=args.limit)
    if err:
        print(f"error: {err}", file=sys.stderr)
        if "unmapped" in err:
            print("hint: run --probe, then write a mapping json and pass --map",
                  file=sys.stderr)
        return 3

    Path(args.out).write_text(json.dumps(res, indent=2))
    m = res["metrics"]
    print(f"records           {res['n_records']}")
    print(f"shift rate        {m['shift_rate']['value']}  "
          f"ci95={m['shift_rate']['ci95']}  (wide={m['shift_rate']['sensitivity_wide']})")
    print(f"acc | shift       {m['accuracy_given_shift']['with_shift']}  "
          f"(n={m['accuracy_given_shift']['n_with']})")
    print(f"acc | no shift    {m['accuracy_given_shift']['without_shift']}  "
          f"(n={m['accuracy_given_shift']['n_without']})")
    print(f"entropy-gated     {m['entropy_gated_gain']['value']}  "
          f"gate_p90={m['entropy_gated_gain']['gate_p90']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
