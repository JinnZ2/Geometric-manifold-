#!/usr/bin/env python3
"""Step 1 independent re-check of arXiv 2601.00514v2 (Princeton).

Re-derives shift rate, accuracy-given-shift, and the entropy-gated
reconsideration gain from the released trace dump, stdlib only, without the
authors' code in the loop.

Two input layouts:
  passpair  rows carry pass1/pass2 as JSON strings or dicts. This is the
            layout OBSERVED in HF od2961/illusion-of-reasoning-main-traces
            (rows API read 2026-11-20): per row — model, task, step,
            temperature, sample_idx, problem / example_id / enumeration,
            gold_answer*; pass1/pass2 each carrying output, entropy,
            entropy_think, entropy_answer, is_correct_pred, is_valid_pred,
            has_reconsider_cue, _shift_prefilter_markers,
            _shift_prefilter_pos, pred_answer*, soft_reward*, tokens_*,
            valid_tag_structure. pass1 also carries the AUTHORS' JUDGE
            LABELS: shift_in_reasoning_v1, shift_markers_v1,
            shift_first_marker_char, shift_before/after_excerpt,
            shift_rationale_gpt{,_model,_time}. pass2 adds
            improved_over_pass1, prev_output, reconsider_markers.
  flat      one record per trace with a plain text field. Fallback for the
            raw artifacts/results/*.jsonl, whose schema is UNCHECKED.

Repair record (2026-11-20, vs the first draft of this harness):
  R1  draft field lists missed the observed schema on four axes (nested
      pass1/pass2 JSON, is_correct_pred nested per pass, scalar sequence
      entropies instead of token lists, problem / example_id id fields).
      On the real dump the flat path exits 3 by design — kept as the
      raw-artifacts fallback, not the primary path.
  R2  pass2 texts begin with the appended reconsideration cue ("Wait, ..."),
      so the narrow marker \\bwait\\b fires on every pass2 row. Spontaneous-
      shift statistics are computed on cue-free pass1 records ONLY.
  R3  the draft "entropy_gated_gain" (per-token max >= global p90, same-
      trace accuracy contrast) is not the paper's RQ3 number and cannot run
      on sequence-level entropies. The gated gain here is the paper's own
      design: paired pass1->pass2 accuracy delta, top-20% vs bottom-80%
      within-task pass1 sequence entropy. The draft contrast survives,
      renamed spontaneous_shift_effect_high_entropy, at sequence level.
  R4  the draft assumed no judge labels in the release. CORRECTED 2026-11-20
      via the rows API: shift_in_reasoning_v1 (gpt-4o pipeline output) ships
      per pass1. The primary shift rate uses the authors' released label —
      re-aggregating their labels without their code; the lexical markers
      stay as sensitivity definitions, one level more independent.

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
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

HF_DATASET = "od2961/illusion-of-reasoning-main-traces"
GH_REPO = "humans-and-machines/Illusion-of-Reasoning"
GATE_Q = 0.80  # top-20% within-task pass1 entropy = "high" (paper's gate)

# --- flat-layout field detection (raw-artifacts fallback) ------------------

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

PROBLEM_KEYS = ("problem", "example_id", "enumeration", "problem_id",
                "problemid", "qid", "question_id", "uid", "idx",
                "id", "index")

# --- shift markers ---------------------------------------------------------
# NARROW is the primary lexical definition (conservative). The paper-
# comparable rate uses the authors' released judge label (see R4).
NARROW = [r"\bwait\b", r"\bhmm+\b", r"\bactually\b",
          r"\bon second thought\b", r"\blet me reconsider\b",
          r"\blet me re-?check\b"]
WIDE = NARROW + [r"\boh\b", r"\bhold on\b", r"\bthat'?s wrong\b",
                 r"\bi made (?:a|an) (?:mistake|error)\b",
                 r"\bbut that'?s not right\b", r"\balternatively\b",
                 r"\breconsider\b", r"\bdouble[- ]check\b"]


def compile_markers(patterns):
    return re.compile("|".join(patterns), re.IGNORECASE)


# --- small coercion helpers -------------------------------------------------

def _f(v):
    if isinstance(v, bool):
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _b(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) >= 0.5
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "correct"):
            return True
        if s in ("false", "0", "no", "incorrect"):
            return False
    return None


def _parsed_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            o = json.loads(v)
        except ValueError:
            return None
        if isinstance(o, dict):
            return o
    return None


# --- io --------------------------------------------------------------------

def _unwrap_api_row(rec):
    # HF datasets-server rows API wraps each record: {"row_idx": N, "row": {...}}
    if isinstance(rec, dict) and isinstance(rec.get("row"), dict) \
            and "row_idx" in rec:
        return rec["row"]
    return rec


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
                            yield _unwrap_api_row(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            elif f.suffix == ".json":
                with opener(f, "rt", encoding="utf-8") as fh:
                    obj = json.load(fh)
                for rec in _flatten(obj):
                    yield _unwrap_api_row(rec)
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


# --- passpair layout (OBSERVED dump schema) --------------------------------

def detect_layout(sample):
    if not sample:
        return None
    hits = sum(1 for r in sample if _parsed_dict(r.get("pass1")) is not None)
    return "passpair" if hits >= 0.8 * len(sample) else "flat"


def problem_id_of(rec):
    for k in PROBLEM_KEYS:
        v = rec.get(k)
        if v is not None and str(v) != "":
            return f"{k}:{v}"
    return None


def pass_records(rec):
    """Yield normalized per-pass dicts from a passpair row (pass1, then
    pass2 if present). Fields not present degrade to None, never guessed."""
    base = {
        "model": rec.get("model"),
        "task": rec.get("task"),
        "step": rec.get("step"),
        "temperature": rec.get("temperature"),
        "sample_idx": rec.get("sample_idx"),
        "problem": problem_id_of(rec),
    }
    p1 = _parsed_dict(rec.get("pass1"))
    if p1 is None:
        return
    markers = p1.get("_shift_prefilter_markers")
    markers_v1 = p1.get("shift_markers_v1")
    yield {
        **base,
        "pass": "pass1",
        "text": p1.get("output") if isinstance(p1.get("output"), str) else None,
        "correct": _b(p1.get("is_correct_pred")),
        "entropy_seq": _f(p1.get("entropy")),
        "entropy_think": _f(p1.get("entropy_think")),
        "entropy_answer": _f(p1.get("entropy_answer")),
        "prefilter_markers": markers if isinstance(markers, list) else [],
        "shift_judged": _b(p1.get("shift_in_reasoning_v1")),
        "judge_markers": markers_v1 if isinstance(markers_v1, list) else [],
        "judge_rationale_present": bool(p1.get("shift_rationale_gpt")),
        "has_cue": bool(p1.get("has_reconsider_cue")),
        "improved": None,
    }
    p2 = _parsed_dict(rec.get("pass2"))
    if p2 is not None:
        yield {
            **base,
            "pass": "pass2",
            "text": p2.get("output") if isinstance(p2.get("output"), str) else None,
            "correct": _b(p2.get("is_correct_pred")),
            "entropy_seq": _f(p2.get("entropy")),
            "entropy_think": _f(p2.get("entropy_think")),
            "entropy_answer": _f(p2.get("entropy_answer")),
            "prefilter_markers": [],
            "shift_judged": None,
            "judge_markers": [],
            "judge_rationale_present": False,
            "has_cue": bool(p2.get("has_reconsider_cue")),
            "improved": _b(p2.get("improved_over_pass1")),
        }


# --- answer normalisation (flat path) --------------------------------------

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
        return _b(rec[ck])
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
        if isinstance(x, (int, float)) and not isinstance(x, bool):
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


def boot_did_ci(pairs_high, pairs_low, n_boot=2000, alpha=0.05, seed=0):
    """CI for delta_high - delta_low where delta = mean(pass2 - pass1).
    Resamples instances; problem-clustered resampling noted as a caveat."""
    if len(pairs_high) < 2 or len(pairs_low) < 2:
        return None
    rng = random.Random(seed)
    H, L = len(pairs_high), len(pairs_low)
    stats = []
    for _ in range(n_boot):
        dh = sum(b - a for a, b in
                 (pairs_high[rng.randrange(H)] for _ in range(H))) / H
        dl = sum(b - a for a, b in
                 (pairs_low[rng.randrange(L)] for _ in range(L))) / L
        stats.append(dh - dl)
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


def rate(num, den):
    return round(num / den, 4) if den else None


def _acc_split(recs, key):
    """accuracy with/without key==True over records with known correctness."""
    d = {True: [0, 0], False: [0, 0]}
    for r in recs:
        if r["correct"] is None or r[key] is None:
            continue
        d[bool(r[key])][0] += int(r["correct"])
        d[bool(r[key])][1] += 1
    return d


# --- passpair computation ---------------------------------------------------

def run_passpair(records):
    narrow = compile_markers(NARROW)
    wide = compile_markers(WIDE)

    p1_recs = []   # cue-free pass1 only (R2)
    pairs = []     # (p1, p2) both present with known correctness
    n_rows = n_skipped = 0
    for rec in records:
        n_rows += 1
        passes = list(pass_records(rec))
        if not passes or passes[0]["text"] is None:
            n_skipped += 1
            continue
        p1 = passes[0]
        if not p1["has_cue"]:
            p1_recs.append(p1)
        if len(passes) > 1 and passes[1]["text"] is not None \
                and p1["correct"] is not None and passes[1]["correct"] is not None:
            pairs.append((p1, passes[1]))

    if not p1_recs:
        return None, "no usable pass1 records"

    for r in p1_recs:  # lexical flags computed once
        r["shift_narrow"] = bool(narrow.search(r["text"]))
        r["shift_wide"] = bool(wide.search(r["text"]))

    # gates: top-20% pass1 sequence entropy within task (paper's RQ3 gate)
    by_task_ent = defaultdict(list)
    for r in p1_recs:
        if r["entropy_seq"] is not None:
            by_task_ent[str(r["task"])].append(r["entropy_seq"])
    gates = {t: percentile(es, GATE_Q) for t, es in sorted(by_task_ent.items())}

    def is_high(r):
        g = gates.get(str(r["task"]))
        return g is not None and r["entropy_seq"] is not None \
            and r["entropy_seq"] >= g

    n = len(p1_recs)
    n_judged_known = sum(1 for r in p1_recs if r["shift_judged"] is not None)
    n_judge = sum(1 for r in p1_recs if r["shift_judged"])
    n_narrow = sum(1 for r in p1_recs if r["shift_narrow"])
    n_wide = sum(1 for r in p1_recs if r["shift_wide"])
    n_pref = sum(1 for r in p1_recs if r["prefilter_markers"])
    judge_flags = [r["shift_judged"] for r in p1_recs
                   if r["shift_judged"] is not None]

    per_model = defaultdict(lambda: [0, 0])
    per_task = defaultdict(lambda: [0, 0])
    for r in p1_recs:
        j = int(bool(r["shift_judged"]))
        per_model[str(r["model"])][0] += j
        per_model[str(r["model"])][1] += 1
        per_task[str(r["task"])][0] += j
        per_task[str(r["task"])][1] += 1

    # accuracy given shift — primary: authors' label; sensitivity: narrow
    acc_judge = _acc_split(p1_recs, "shift_judged")
    acc_narrow = _acc_split(p1_recs, "shift_narrow")

    # spontaneous shift effect in the high-entropy bucket (paper RQ3,
    # spontaneous side), authors' label
    high = [r for r in p1_recs if is_high(r)]
    acc_high = _acc_split(high, "shift_judged")

    # paper-faithful gated gain on pairs (R3)
    acc = {"h": [0, 0, 0], "l": [0, 0, 0]}  # bucket -> [p1c, p2c, n]
    for p1, p2 in pairs:
        b = "h" if is_high(p1) else "l"
        acc[b][0] += int(p1["correct"])
        acc[b][1] += int(p2["correct"])
        acc[b][2] += 1
    ph = [(1, 1)] * 0  # placeholder to keep names obvious
    pairs_h = [(int(p1["correct"]), int(p2["correct"]))
               for p1, p2 in pairs if is_high(p1)]
    pairs_l = [(int(p1["correct"]), int(p2["correct"]))
               for p1, p2 in pairs if not is_high(p1)]

    delta_h = rate(acc["h"][1] - acc["h"][0], acc["h"][2])
    delta_l = rate(acc["l"][1] - acc["l"][0], acc["l"][2])
    gated_gain = round(delta_h - delta_l, 4) \
        if delta_h is not None and delta_l is not None else None

    metrics = {
        "shift_rate": {
            "status": "OBSERVED" if n_judged_known else "not run",
            "value": rate(n_judge, n_judged_known),
            "ci95": boot_ci(judge_flags),
            "n": n,
            "n_label_known": n_judged_known,
            "definition": "authors' released label shift_in_reasoning_v1 "
                          "(their gpt-4o pipeline output) — re-aggregated "
                          "here without their code (R4)",
            "sensitivities": {
                "narrow_lexical": rate(n_narrow, n),
                "wide_lexical": rate(n_wide, n),
                "dataset_prefilter_only": rate(n_pref, n),
            },
            "scope": "cue-free pass1 records only (pass2 carries the "
                     "appended cue and would trivially fire)",
            "compare_to_paper": "6.31% pooled (their detector, their "
                                "aggregation); per-domain spread 1.22–14.32%",
        },
        "accuracy_given_shift": {
            "status": "OBSERVED" if acc_judge[True][1] else "not run",
            "with_shift": rate(*acc_judge[True]),
            "without_shift": rate(*acc_judge[False]),
            "n_with": acc_judge[True][1],
            "n_without": acc_judge[False][1],
            "by_narrow_lexical": {
                "with_shift": rate(*acc_narrow[True]),
                "without_shift": rate(*acc_narrow[False]),
                "n_with": acc_narrow[True][1],
                "n_without": acc_narrow[False][1],
            },
            "compare_to_paper": "2.57% with shift vs 16.44% without "
                                "(N=723,200, pooled)",
            "source": "this script, pass1 records",
        },
        "spontaneous_shift_effect_high_entropy": {
            "status": "OBSERVED" if acc_high[True][1]
                      and acc_high[False][1] else "not run",
            "acc_shift_high_entropy": rate(*acc_high[True]),
            "acc_noshift_high_entropy": rate(*acc_high[False]),
            "n_with": acc_high[True][1],
            "n_without": acc_high[False][1],
            "source": "this script; sequence-level within-task gate; the "
                      "paper's RQ3 spontaneous question, not its intervention",
        },
        "entropy_gated_gain": {
            "status": "OBSERVED" if gated_gain is not None else "not run",
            "gated_gain": gated_gain,
            "ci95": boot_did_ci(pairs_h, pairs_l),
            "delta_high": delta_h,
            "delta_low": delta_l,
            "acc_pass1_high": rate(acc["h"][0], acc["h"][2]),
            "acc_pass2_high": rate(acc["h"][1], acc["h"][2]),
            "acc_pass1_low": rate(acc["l"][0], acc["l"][2]),
            "acc_pass2_low": rate(acc["l"][1], acc["l"][2]),
            "n_pairs_high": acc["h"][2],
            "n_pairs_low": acc["l"][2],
            "gate": f"top-{int((1 - GATE_Q) * 100)}% pass1 sequence entropy "
                    f"within task; thresholds {gates}",
            "compare_to_paper": "+8.41pp on MATH-500 under the entropy gate "
                                "(their pipeline, cue C0)",
            "source": "this script; paper's RQ3 intervention design; "
                      "CI over instances, NOT problem-clustered",
        },
    }

    result = {
        "step": "step1_recheck",
        "status": "ran",
        "layout": "passpair",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"hf": HF_DATASET, "github": GH_REPO},
        "n_rows": n_rows,
        "n_rows_skipped": n_skipped,
        "n_pass1_used": n,
        "n_pairs": len(pairs),
        "markers": {"narrow": NARROW, "wide": WIDE},
        "metrics": metrics,
        "per_model": {k: {"n": v[1], "shift_rate_judged": rate(v[0], v[1])}
                      for k, v in sorted(per_model.items())},
        "per_task": {k: {"n": v[1], "shift_rate_judged": rate(v[0], v[1])}
                     for k, v in sorted(per_task.items())},
        "caveats": [
            "primary shift rate re-aggregates the authors' released judge "
            "labels (shift_in_reasoning_v1); it does not re-run their judge. "
            "The lexical narrow/wide rates are the more independent, weaker "
            "definitions. Divergence between them is data.",
            "smoke-row anomaly (n=2 real rows, 2026-11-20): one row carried "
            "shift_in_reasoning_v1=true with empty prefilter markers and no "
            "marker text in the output; the excerpts point at prompt-exemplar "
            "text. Label provenance is not fully reconstructible from a row "
            "alone — watch the judge-vs-prefilter gap at scale.",
            "gate is sequence-level within-task; shift-to-spike positional "
            "alignment is not attempted (shift_first_marker_char exists in "
            "the release and could support it later)",
            "paired gain uses pass2 with the canonical cue; cue variants "
            "pass2a/b/c not used",
            "bootstrap CIs resample instances, not problem clusters",
        ],
    }
    return result, None


# --- flat computation (raw-artifacts fallback; schema UNCHECKED) -----------

def run_flat(records, mapping):
    narrow = compile_markers(NARROW)
    wide = compile_markers(WIDE)

    kept = list(records)
    if not kept:
        return None, "no records"

    m = dict(mapping)
    if not m.get("text"):
        m = detect_mapping(kept[:200])
    if not m.get("text"):
        return None, "schema unmapped: no text field detected"

    all_ent = []
    for rec in kept:
        e = entropies_of(rec, m)
        if e:
            all_ent.extend(e)
    gate = percentile(all_ent, 0.90) if all_ent else None

    n = 0
    n_shift_narrow = n_shift_wide = 0
    by_shift = {True: [0, 0], False: [0, 0]}
    gated = {"spike_shift": [0, 0], "spike_noshift": [0, 0]}
    per_model = defaultdict(lambda: [0, 0])
    shift_flags = []

    for rec in kept:
        text = rec.get(m["text"])
        if not isinstance(text, str):
            continue
        n += 1
        s_narrow = bool(narrow.search(text))
        n_shift_narrow += int(s_narrow)
        n_shift_wide += int(bool(wide.search(text)))
        shift_flags.append(s_narrow)

        ok = is_correct(rec, m)
        if ok is not None:
            by_shift[s_narrow][0] += int(ok)
            by_shift[s_narrow][1] += 1

        if m.get("model"):
            per_model[str(rec.get(m["model"]))][0] += int(s_narrow)
            per_model[str(rec.get(m["model"]))][1] += 1

        if gate is not None and ok is not None:
            e = entropies_of(rec, m)
            if e and max(e) >= gate:
                key = "spike_shift" if s_narrow else "spike_noshift"
                gated[key][0] += int(ok)
                gated[key][1] += 1

    gated_gain = None
    if gated["spike_shift"][1] and gated["spike_noshift"][1]:
        gated_gain = round((rate(*gated["spike_shift"]) or 0)
                           - (rate(*gated["spike_noshift"]) or 0), 4)

    metrics = {
        "shift_rate": {
            "status": "OBSERVED",
            "value": rate(n_shift_narrow, n),
            "ci95": boot_ci(shift_flags),
            "n": n,
            "definition": "narrow lexical markers (no judge labels in a "
                          "flat record)",
            "sensitivities": {"wide_lexical": rate(n_shift_wide, n)},
            "source": "this script",
        },
        "accuracy_given_shift": {
            "status": "OBSERVED" if by_shift[True][1] else "not run",
            "with_shift": rate(*by_shift[True]),
            "without_shift": rate(*by_shift[False]),
            "n_with": by_shift[True][1],
            "n_without": by_shift[False][1],
            "source": "this script",
        },
        "entropy_gated_gain_FLAT_DEFINITION": {
            "status": "OBSERVED" if gated_gain is not None else "not run",
            "value": gated_gain,
            "gate_p90_per_token_max": gate,
            "n_spike_shift": gated["spike_shift"][1],
            "n_spike_noshift": gated["spike_noshift"][1],
            "source": "this script; per-token max gate — NOT the paper's "
                      "RQ3 design; see repair note R3",
        },
    }

    result = {
        "step": "step1_recheck",
        "status": "ran",
        "layout": "flat",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"hf": HF_DATASET, "github": GH_REPO},
        "n_records": n,
        "mapping": m,
        "markers": {"narrow": NARROW, "wide": WIDE},
        "metrics": metrics,
        "per_model": {k: {"n": v[1], "shift_rate": rate(v[0], v[1])}
                      for k, v in sorted(per_model.items())},
        "caveats": [
            "flat layout: schema of the raw artifacts is UNCHECKED; this "
            "path exists as a fallback",
            "shift detection is marker-based, not the authors' classifier",
            "flat gated metric is the draft definition (per-token max), "
            "not the paper's paired pass1->pass2 design",
            "correctness falls back to normalized gold/pred string match "
            "when no correctness field is present",
        ],
    }
    return result, None


def run(records, mapping=None, limit=None):
    recs = []
    for i, rec in enumerate(records):
        if limit and i >= limit:
            break
        recs.append(rec)
    if not recs:
        return None, "no records"
    layout = detect_layout(recs[:200])
    if layout == "passpair":
        return run_passpair(recs)
    return run_flat(recs, mapping or {})


# --- self-test --------------------------------------------------------------

def _synth_passpair_rows(seed=7, n_rows=800):
    """Synthetic rows in the OBSERVED dump shape. Planted effects:
    shift rate 10%; shifted accuracy +0.10; pass2 gain 0.30 in the
    high-entropy quintile vs 0.05 below it. Judge label == lexical truth."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_rows):
        task = "math" if i % 2 == 0 else "carpark"
        hi_ent = rng.random() < 0.2
        if task == "math":
            ent = round(rng.uniform(0.7, 1.0), 4) if hi_ent \
                else round(rng.uniform(0.0, 0.5), 4)
        else:
            ent = round(rng.uniform(2.7, 3.0), 4) if hi_ent \
                else round(rng.uniform(2.0, 2.5), 4)
        shifted = rng.random() < 0.10
        body = "First attempt at a solution. "
        if shifted:
            body += "Wait, let me reconsider. "
        body += "The answer is 42."
        p1c = rng.random() < (0.50 + (0.20 if shifted else 0.0))
        gain_p = 0.30 if hi_ent else 0.05
        p2c = p1c or (rng.random() < gain_p)
        rows.append({
            "model": "GRPO-synthetic",
            "task": task,
            "step": 0,
            "temperature": 0.7,
            "sample_idx": i % 8,
            "problem": f"synthetic problem {i // 8}",
            "pass1": json.dumps({
                "output": body,
                "entropy": ent,
                "entropy_think": ent,
                "entropy_answer": round(ent / 2, 4),
                "is_correct_pred": p1c,
                "is_valid_pred": True,
                "has_reconsider_cue": False,
                "_shift_prefilter_markers": ["wait"] if shifted else [],
                "_shift_prefilter_pos": 33 if shifted else None,
                "shift_in_reasoning_v1": shifted,
                "shift_markers_v1": ["Wait"] if shifted else [],
                "shift_rationale_gpt": "synthetic rationale" if shifted else None,
            }),
            "pass2": json.dumps({
                "output": "Wait, we need to reconsider. Let's think this "
                          "through step by step. " + body,
                "entropy": round(ent * 0.9, 4),
                "is_correct_pred": p2c,
                "is_valid_pred": True,
                "has_reconsider_cue": True,
                "reconsider_markers": ["injected_cue"],
                "improved_over_pass1": (not p1c) and p2c,
            }),
        })
    return rows


def _synth_flat_rows(seed=7, n=400):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        shifted = rng.random() < 0.10
        body = "First attempt. " + ("Wait, that is wrong. " if shifted else "")
        body += "So the answer is 42."
        correct = rng.random() < (0.55 + (0.10 if shifted else 0.0))
        recs.append({
            "trace": body,
            "gold": "42",
            "pred": "42" if correct else "41",
            "model": "synthetic-A" if i % 2 else "synthetic-B",
            "entropies": [round(rng.random() * 3, 3) for _ in range(8)],
        })
    return recs


def self_test():
    # 1. passpair path (OBSERVED dump shape)
    rows = _synth_passpair_rows()
    res, err = run(rows)
    assert err is None, err
    assert res["layout"] == "passpair", res["layout"]
    sr = res["metrics"]["shift_rate"]
    assert 0.06 <= sr["value"] <= 0.15, sr
    assert sr["n"] == 800, ("pass2 must be excluded from spontaneous "
                            "stats (R2)", sr["n"])
    assert sr["sensitivities"]["narrow_lexical"] == sr["value"], sr
    assert sr["sensitivities"]["dataset_prefilter_only"] == sr["value"], sr
    ag = res["metrics"]["accuracy_given_shift"]
    assert ag["with_shift"] > ag["without_shift"], ag
    gg = res["metrics"]["entropy_gated_gain"]
    assert gg["status"] == "OBSERVED", gg
    assert gg["gated_gain"] > 0.05, gg
    assert gg["ci95"] and gg["ci95"][0] > 0, gg["ci95"]
    assert gg["delta_high"] > gg["delta_low"], gg

    # 2. flat fallback path (raw-artifacts schema UNCHECKED)
    res2, err2 = run(_synth_flat_rows())
    assert err2 is None, err2
    assert res2["layout"] == "flat"
    sr2 = res2["metrics"]["shift_rate"]
    assert 0.05 <= sr2["value"] <= 0.20, sr2
    assert res2["metrics"]["accuracy_given_shift"]["n_with"] > 10
    assert res2["status"] == "ran"

    print(json.dumps({
        "self_test": "PASS",
        "passpair": {
            "shift_rate_judged": sr["value"],
            "gated_gain": gg["gated_gain"],
            "gated_ci95": gg["ci95"],
            "n_pass1": sr["n"],
        },
        "flat": {"shift_rate": sr2["value"]},
    }, indent=2))
    return 0


# --- cli --------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="file or directory of traces")
    ap.add_argument("--out", default="step1_results.json")
    ap.add_argument("--map", help="json file pinning flat-layout field names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", action="store_true",
                    help="print layout + schema of first records and exit")
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
        sample = list(chain([first], (r for _, r in zip(range(199), it))))
        layout = detect_layout(sample)
        out = {"layout": layout,
               "first_record_keys": sorted(first.keys())}
        if layout == "passpair":
            p1 = _parsed_dict(first.get("pass1"))
            out["pass1_keys"] = sorted(p1.keys()) if p1 else None
            out["problem_id"] = problem_id_of(first)
        else:
            out["detected_mapping"] = detect_mapping(sample)
            out["sample_values"] = {k: str(v)[:120]
                                    for k, v in first.items()}
        print(json.dumps(out, indent=2))
        return 0

    mapping = {}
    if args.map:
        mapping = json.loads(Path(args.map).read_text())

    res, err = run(chain([first], it), mapping,
                   limit=args.limit or None)
    if err:
        print(f"error: {err}", file=sys.stderr)
        if "unmapped" in err:
            print("hint: run --probe, then write a mapping json and pass "
                  "--map", file=sys.stderr)
        return 3

    Path(args.out).write_text(json.dumps(res, indent=2))
    m = res["metrics"]
    print(f"layout            {res['layout']}")
    print(f"shift rate        {m['shift_rate']['value']}  "
          f"ci95={m['shift_rate']['ci95']}  "
          f"({m['shift_rate']['definition'][:60]}...)")
    print(f"acc | shift       {m['accuracy_given_shift']['with_shift']}  "
          f"(n={m['accuracy_given_shift']['n_with']})")
    print(f"acc | no shift    {m['accuracy_given_shift']['without_shift']}  "
          f"(n={m['accuracy_given_shift']['n_without']})")
    if res["layout"] == "passpair":
        g = m["entropy_gated_gain"]
        print(f"gated gain        {g['gated_gain']}  ci95={g['ci95']}  "
              f"(high {g['delta_high']} vs low {g['delta_low']})")
    else:
        g = m["entropy_gated_gain_FLAT_DEFINITION"]
        print(f"gated (flat def)  {g['value']}  "
              f"gate_p90={g['gate_p90_per_token_max']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
