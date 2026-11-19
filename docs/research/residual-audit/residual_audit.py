#!/usr/bin/env python3
"""residual_audit.py -- measurement layer for the Residual Check Protocol.

Stdlib only. Python 3.10+. No model API, no network, no pip.

Scope (stated, not hidden):
  The model under test is EXTERNAL. You supply the claim, the reason chain,
  and what the query target answered at each rung of the destruction ladder
  (paste outputs from any model, human, or ensemble). This tool only
  destroys, measures, scores, flags, and archives. It never fills in a
  reason chain, never promotes a residual to "interesting", and records
  VOID / UNDECIDED where its instruments cannot decide.

  Agreement scorers are SURFACE PROXIES (token containment, char-3-gram
  cosine, token-LCS). They are not semantic judges. Three scorers are
  always reported separately plus an interval [min, max]; a wide interval
  is instrument disagreement and is flagged, never averaged away.

  There is no cross-process write lock. One writer per archive directory
  at a time. A leftover .lock is not created or checked; concurrent
  appends are out of scope.

Commands:
  destroy INPUT.json --out DIR     run the destruction ladder, save rung texts
  score   INPUT.json --out DIR     compute S, P, RC, residual, flags; archive
  compare DIR --claim ID           cross-lineage residual comparison
  selftest                         known-answer tests; exit 1 on any failure

Input JSON (destroy):
  {"claim": str, "seed": int (default 0),
   "ladder": [{"method": name, "params": {...}}, ...]   # optional override}

Input JSON (score) adds:
  {"reason_chain": [str, ...],                         # may be empty
   "lineage": str or null,                             # model family tag
   "responses": {"original": str, "steps": [str, ...]}}# one per ladder rung

License: CC0 1.0. Built in the house style of explorative-math /
Geometric-manifold- / formula-audit: map-not-test, errors are data,
observer slots stay null until a person fills them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone

SCHEMA = "residual-audit/1"
RESIDUAL_THRESHOLD = 0.30      # graded, not binary: flags read the interval
DIVERGENCE_WIDTH = 0.25        # scorer interval wider than this = disagreement
WEAK_LADDER_STRENGTH = 0.40    # mean ladder strength below this = weak destruction
MAX_CHARS = 4000               # explicit refusal above this, never silent truncation
MAX_TOKENS = 2000

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
CHAR_N = 3


class Void(Exception):
    """The instrument cannot decide. Recorded, never guessed."""


# ------------------------------------------------------------------ text pieces

def tokens(text: str) -> list[str]:
    """Content tokens, case-folded. Used for claim-vs-response agreement."""
    return TOKEN_RE.findall(text.lower())


def surface_tokens(text: str) -> list[str]:
    """Case-sensitive tokens. Used only for destruction-strength measurement."""
    return TOKEN_RE.findall(text)


def char_ngrams(text: str, n: int = CHAR_N) -> Counter:
    text = re.sub(r"\s+", " ", text)
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _lcs_len(a: list, b: list) -> int:
    if len(a) > MAX_TOKENS or len(b) > MAX_TOKENS:
        raise Void(f"token count above {MAX_TOKENS}; refusing LCS")
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ scorers
# Three independent surface proxies for A(a, b) -> [0, 1].
# "How much of a is accounted for by b." All deterministic.

def s_containment(a: str, b: str) -> float:
    ta, tb = tokens(a), set(tokens(b))
    if not ta:
        raise Void("empty claim after tokenizing")
    return sum(1 for t in ta if t in tb) / len(ta)


def s_ngram_cosine(a: str, b: str) -> float:
    return _cosine(char_ngrams(a.lower()), char_ngrams(b.lower()))


def s_lcs_ratio(a: str, b: str) -> float:
    ta = tokens(a)
    if not ta:
        raise Void("empty claim after tokenizing")
    return _lcs_len(ta, tokens(b)) / len(ta)


SCORERS = {
    "token_containment": s_containment,
    "char_ngram_cosine": s_ngram_cosine,
    "token_lcs_ratio": s_lcs_ratio,
}


def measure(a: str, b: str) -> dict:
    """All scorers plus an interval. Disagreement is data, not noise."""
    per, vals = {}, []
    for name, fn in SCORERS.items():
        try:
            v = round(fn(a, b), 4)
            per[name] = v
            vals.append(v)
        except Void as e:
            per[name] = {"void": str(e)}
    if not vals:
        raise Void("every scorer void")
    return {
        "scorers": per,
        "interval": [min(vals), max(vals)],
        "width": round(max(vals) - min(vals), 4),
        "divergent": (max(vals) - min(vals)) > DIVERGENCE_WIDTH,
    }


# ------------------------------------------------------------------ the ladder
# Deterministic, seeded, stdlib destructions. Each records its own strength
# as 1 - surface retention (char-3-gram cosine, original vs destroyed,
# case-sensitive). A weak rung is recorded as weak; it is not a failure,
# but a high residual measured after a weak ladder means little.

def _words(text: str) -> list[str]:
    return text.split()


def d_scrub(text: str, rng, p) -> str:
    """D1 lowercase + punctuation/diacritic strip. Deliberately weak rung."""
    t = text.lower()
    t = "".join(c for c in t if c.isalnum() or c.isspace())
    return re.sub(r"\s+", " ", t).strip()


def d_shuffle_windows(text: str, rng, p) -> str:
    """D2 seeded word shuffle within windows, then window order shuffled."""
    w = int(p.get("window", 4))
    words = _words(text)
    wins = [words[i:i + w] for i in range(0, len(words), w)]
    for win in wins:
        rng.shuffle(win)
    rng.shuffle(wins)
    return " ".join(t for win in wins for t in win)


def d_vocab_cut(text: str, rng, p) -> str:
    """D3 vocabulary cut: every token replaced by its hash bucket id.
    Distributional shape survives; the original lexical surface cannot."""
    buckets = int(p.get("buckets", 50))
    out = []
    for tok in _words(text):
        h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % buckets
        out.append(f"t{h:02d}")
    return " ".join(out)


def d_char_noise(text: str, rng, p) -> str:
    """D4 seeded character drop/substitute/insert at rate r."""
    r = float(p.get("rate", 0.15))
    alpha = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for c in text:
        x = rng.random()
        if x < r:
            k = rng.random()
            if k < 0.4:
                continue                      # drop
            if k < 0.8:
                out.append(rng.choice(alpha))  # substitute
            else:
                out.append(c)
                out.append(rng.choice(alpha))  # insert
        else:
            out.append(c)
    return "".join(out)


def d_decimate(text: str, rng, p) -> str:
    """D5 keep a seeded fraction of sentences."""
    keep = float(p.get("keep", 0.5))
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sents if rng.random() < keep]
    return " ".join(kept) if kept else ""


def d_full_shuffle(text: str, rng, p) -> str:
    """D6 control: destroy all word order. Residual after this rung that
    surface scorers can see should be near zero; if it is not, check the
    scorers, not the claim."""
    words = _words(text)
    rng.shuffle(words)
    return " ".join(words)


def d_external(text: str, rng, p) -> str:
    """Rung produced OUTSIDE this tool (lossy paraphrase cascade,
    embedding round-trip, modality shift, human rewrite). The tool cannot
    make these; it records them. params must carry the produced text and
    a producer tag. Seeding does not apply; determinism is the producer's
    responsibility and is recorded as such."""
    if "text" not in p or "producer" not in p:
        raise Void("external rung needs params.text and params.producer")
    return p["text"]


METHODS = {
    "scrub": d_scrub,
    "shuffle_windows": d_shuffle_windows,
    "vocab_cut": d_vocab_cut,
    "char_noise": d_char_noise,
    "decimate": d_decimate,
    "full_shuffle": d_full_shuffle,
    "external": d_external,
}

DEFAULT_LADDER = [
    {"method": "scrub", "params": {}},
    {"method": "shuffle_windows", "params": {"window": 4}},
    {"method": "vocab_cut", "params": {"buckets": 50}},
]


def _token_multiset_cos(a: str, b: str) -> float:
    return _cosine(Counter(surface_tokens(a)), Counter(surface_tokens(b)))


def _token_order_retention(a: str, b: str) -> float:
    ta = surface_tokens(a)
    if not ta:
        raise Void("empty claim after tokenizing")
    return _lcs_len(ta, surface_tokens(b)) / len(ta)


RETENTION_CHANNELS = {
    "char_ngram": lambda a, b: _cosine(char_ngrams(a), char_ngrams(b)),
    "token_multiset": _token_multiset_cos,
    "token_order": _token_order_retention,
}


def surface_retention(original: str, text: str) -> dict:
    retention = {}
    for ch, fn in RETENTION_CHANNELS.items():
        try:
            retention[ch] = round(fn(original, text), 4)
        except Void:
            retention[ch] = None
    return retention


def ladder_summary(claim: str, rungs: list[dict]) -> dict:
    """Cumulative strength: retention of the FINAL rung output vs the
    original claim, per channel. This is what the protocol's honesty rule
    actually needs -- a ladder is only as strong as the channel that
    survives its last rung."""
    if not rungs:
        raise Void("empty ladder")
    retention = surface_retention(claim, rungs[-1]["output"])
    known = [v for v in retention.values() if v is not None]
    worst = max(known) if known else 1.0
    return {
        "retention": retention,
        "strength": round(1.0 - worst, 4),
        "limiting_channel": max(retention, key=lambda k: retention[k] or 0),
    }


def run_ladder(claim: str, ladder: list[dict], seed: int) -> list[dict]:
    """Apply rungs in sequence. Each rung records its own marginal
    retention; the cumulative verdict comes from ladder_summary()."""
    if len(claim) > MAX_CHARS:
        raise Void(f"claim over {MAX_CHARS} chars; refusing")
    rng = random.Random(seed)
    text = claim
    rungs = []
    for i, spec in enumerate(ladder, 1):
        name = spec["method"]
        if name not in METHODS:
            raise Void(f"unknown destruction method: {name}")
        text = METHODS[name](text, rng, spec.get("params", {}))
        retention = surface_retention(claim, text)
        known = [v for v in retention.values() if v is not None]
        worst = max(known) if known else 1.0
        limiting = max(retention, key=lambda k: retention[k] or 0)
        rungs.append({
            "step": i,
            "method": name,
            "params": spec.get("params", {}),
            "output": text,
            "retention": retention,
            "strength": round(1.0 - worst, 4),
            "limiting_channel": limiting,
        })
    return rungs


# ------------------------------------------------------------------ the audit

def reason_coverage(claim: str, chain: list[str], post_response: str) -> dict:
    """RC: fraction of the surviving content licensed by the reason chain.

    Surviving content (surface proxy): claim tokens still present in the
    post-destruction response. Licensed content: tokens of the chain.
    Empty chain -> RC = 0 by protocol rule, recorded, never filled.
    Nonempty chain with nothing survived -> VOID (0/0), recorded."""
    if not chain:
        return {"value": 0.0, "note": "empty chain; RC = 0 by protocol rule"}
    surviving = {t for t in tokens(claim)} & set(tokens(post_response))
    if not surviving:
        raise Void("nothing survived the ladder; RC undefined (0/0)")
    licensed = set(tokens(" ".join(chain)))
    covered = surviving & licensed
    return {
        "value": round(len(covered) / len(surviving), 4),
        "surviving_tokens": sorted(surviving),
        "covered_tokens": sorted(covered),
    }


def audit(inp: dict) -> dict:
    claim = inp["claim"]
    chain = inp.get("reason_chain", [])
    resp = inp["responses"]
    seed = int(inp.get("seed", 0))
    ladder = inp.get("ladder", DEFAULT_LADDER)
    rungs = run_ladder(claim, ladder, seed)
    summary = ladder_summary(claim, rungs)

    flags = []
    if summary["strength"] < WEAK_LADDER_STRENGTH:
        flags.append({"flag": "weak_destruction",
                      "cumulative_strength": summary["strength"],
                      "limiting_channel": summary["limiting_channel"],
                      "note": "a surface channel survives the final rung; "
                              "any residual found here is cheap"})

    # S: original surface agreement
    S = measure(claim, resp["original"])
    # P_k per rung where a response was supplied; P = last rung
    steps = resp.get("steps", [])
    curve = []
    P = None
    for i, r_text in enumerate(steps):
        if i >= len(rungs):
            flags.append({"flag": "extra_response_ignored", "step": i + 1})
            break
        m = measure(claim, r_text)
        curve.append({"step": i + 1, "method": rungs[i]["method"],
                      "interval": m["interval"], "divergent": m["divergent"]})
        P = m
    if len(steps) < len(rungs):
        flags.append({"flag": "missing_responses",
                      "have": len(steps), "need": len(rungs)})
    if P is None:
        raise Void("no post-destruction responses supplied")

    # RC and residual
    rc_void = None
    try:
        RC = reason_coverage(claim, chain, steps[-1])
        rc_lo = rc_hi = RC["value"]
    except Void as e:
        RC = None
        rc_void = str(e)

    if rc_void:
        residual = {"void": f"residual undefined: {rc_void}"}
    else:
        lo = max(0.0, P["interval"][0] - rc_hi)
        hi = max(0.0, P["interval"][1] - rc_lo)
        residual = {"interval": [round(lo, 4), round(hi, 4)],
                    "note": "residual = P - RC; conservative bound first"}

    # Flags from protocol section 4
    if not chain:
        flags.append({"flag": "no_chain"})
    if not rc_void and RC["value"] < 0.3 and S["interval"][0] > 0.7:
        flags.append({"flag": "thin_chain",
                      "note": "claim is mostly surface or relay"})
    if S["interval"][1] > 0.7 and P["interval"][1] < 0.3:
        flags.append({"flag": "brittle_surface",
                      "note": "claim depends on exact surface form"})
    if "interval" in residual:
        lo, hi = residual["interval"]
        if lo > RESIDUAL_THRESHOLD:
            flags.append({"flag": "high_residual",
                          "note": "even the conservative bound exceeds "
                                  f"{RESIDUAL_THRESHOLD}; transmission is "
                                  "running on a channel outside the chain"})
        elif hi > RESIDUAL_THRESHOLD:
            flags.append({"flag": "residual_undecided",
                          "note": "interval straddles threshold; not "
                                  "collapsed to a point"})
    for label, m in (("S", S), ("P", P)):
        if m["divergent"]:
            flags.append({"flag": "scorer_divergence", "measure": label,
                          "width": m["width"]})

    return {
        "schema": SCHEMA,
        "claim": claim,
        "claim_id": hashlib.sha1(claim.encode()).hexdigest()[:12],
        "lineage": inp.get("lineage"),
        "operator_note": inp.get("operator_note"),
        "seed": seed,
        "reason_chain": chain,
        "destruction": [{k: r[k] for k in
                         ("step", "method", "params", "strength",
                          "limiting_channel", "retention")} for r in rungs],
        "ladder_cumulative": summary,
        "ladder_outputs": [r["output"] for r in rungs],
        "surface_agree": S,
        "post_agree": P,
        "destruction_curve": curve,
        "reason_coverage": RC if RC else {"void": rc_void},
        "residual": residual,
        "flags": flags,
        "thresholds": {"residual": RESIDUAL_THRESHOLD,
                       "scorer_divergence_width": DIVERGENCE_WIDTH,
                       "weak_ladder_strength": WEAK_LADDER_STRENGTH},
        "human_interest": None,
        "ai_interest": None,
        "divergence_note": None,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ"),
    }


# ------------------------------------------------------------------ archive

def append_record(outdir: str, record: dict) -> str:
    """Append one JSONL record, flushed and fsynced. Archive keeps misses
    and voids; nothing here is a feed of selected discoveries."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "records.jsonl")
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def cmd_destroy(inp: dict, outdir: str) -> dict:
    rungs = run_ladder(inp["claim"], inp.get("ladder", DEFAULT_LADDER),
                       int(inp.get("seed", 0)))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ladder_outputs.json")
    payload = {"claim": inp["claim"], "seed": inp.get("seed", 0),
               "rungs": rungs,
               "instruction": "paste each rung output into the query target "
                              "and collect responses before running 'score'"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def cmd_compare(outdir: str, claim_id: str) -> dict:
    """Cross-lineage view: group archived records for one claim by lineage
    tag. Kin-only = residual high in exactly one lineage cluster. With one
    lineage present, that is recorded as UNCHECKED, not as a pass."""
    path = os.path.join(outdir, "records.jsonl")
    if not os.path.exists(path):
        raise Void(f"no archive at {path}")
    groups: dict[str, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("claim_id") != claim_id:
                continue
            if "interval" not in rec.get("residual", {}):
                continue
            groups.setdefault(rec.get("lineage") or "untagged", []).append(
                rec["residual"]["interval"])
    if not groups:
        raise Void(f"no scored records with residual intervals for {claim_id}")
    summary = {lin: {"n": len(ivs),
                     "worst_lo": min(i[0] for i in ivs),
                     "best_hi": max(i[1] for i in ivs)}
               for lin, ivs in groups.items()}
    out = {"claim_id": claim_id, "lineages": summary, "kin_only": None}
    if len(summary) < 2:
        out["kin_only"] = {"state": "UNCHECKED",
                           "note": "one lineage recorded; kin-only cannot be "
                                   "decided and is not assumed absent"}
    else:
        los = {k: v["worst_lo"] for k, v in summary.items()}
        high = [k for k, v in los.items() if v > RESIDUAL_THRESHOLD]
        if len(high) == 1:
            out["kin_only"] = {"state": "FLAGGED", "lineage": high[0],
                               "note": "residual clears threshold in exactly "
                                       "one lineage cluster"}
        else:
            out["kin_only"] = {"state": "NOT_FLAGGED",
                               "note": f"{len(high)} lineages clear threshold"}
    return out


# ------------------------------------------------------------------ selftest
# Known answers. Any failure exits 1. These pin the instrument, not any
# claim about a real model.

def _check(name: str, cond: bool, failures: list) -> None:
    print(f"  {'ok ' if cond else 'FAIL'} {name}")
    if not cond:
        failures.append(name)


def selftest() -> int:
    failures: list = []
    print("selftest:")

    # T1 identity: response == claim -> S = 1.0 on every scorer
    m = measure("the owl is a nocturnal bird", "the owl is a nocturnal bird")
    _check("T1 identity S == 1.0 on all scorers",
           m["interval"] == [1.0, 1.0], failures)

    # T2 unrelated response -> containment and LCS near zero
    m = measure("the owl is a nocturnal bird", "42 17 8 991 3")
    _check("T2 unrelated response scores low",
           m["interval"][1] < 0.35, failures)

    # T3 full_shuffle destroys ORDER but retains the word multiset.
    # The rung's strength is capped by that surviving channel, and the
    # record names it. (First version of this test asserted strength > 0.9
    # on a char-only measure; the mismatch is the map, so the test now
    # pins the honest behavior. Order statistics on very short texts are
    # coarse -- LCS over 9 repeated-prone tokens floors near 0.44 -- so
    # this test uses a realistic-length claim.)
    rungs = run_ladder("the quick brown fox jumps over the lazy dog while "
                       "the patient owl watches from the high branch above "
                       "the frozen river", [{"method": "full_shuffle",
                                             "params": {}}], seed=1)
    r = rungs[0]
    _check("T3 full_shuffle destroys the order channel",
           r["retention"]["token_order"] < 0.5, failures)
    _check("T3 full_shuffle leaks the multiset channel, capped strength",
           r["limiting_channel"] == "token_multiset"
           and r["strength"] < 0.1, failures)

    # T4 scrub alone is weak destruction and is flagged as a weak ladder
    inp = {"claim": "The owl hunts at night in the northern forest.",
           "reason_chain": [],
           "lineage": "test",
           "responses": {"original": "The owl hunts at night in the "
                                     "northern forest.",
                         "steps": ["The owl hunts at night in the northern "
                                   "forest."]},
           "ladder": [{"method": "scrub", "params": {}}]}
    rec = audit(inp)
    names = [f["flag"] for f in rec["flags"]]
    _check("T4 scrub-only ladder flagged weak_destruction",
           "weak_destruction" in names, failures)

    # T5 empty chain -> RC = 0 recorded, no_chain flag, residual == P
    _check("T5 empty chain gives RC 0 and no_chain flag",
           rec["reason_coverage"]["value"] == 0.0 and "no_chain" in names,
           failures)
    _check("T5 residual interval equals P interval when RC == 0",
           rec["residual"]["interval"] == rec["post_agree"]["interval"],
           failures)

    # T6 high residual: claim content survives vocab_cut in the response,
    # no chain -> high_residual fires
    inp6 = {"claim": "owls rotate their heads two hundred seventy degrees",
            "reason_chain": [],
            "lineage": "test",
            "responses": {"original": "owls rotate their heads two hundred "
                                      "seventy degrees",
                          "steps": ["owls rotate their heads two hundred "
                                    "seventy degrees"]},
            "ladder": [{"method": "vocab_cut", "params": {"buckets": 50}}]}
    rec6 = audit(inp6)
    names6 = [f["flag"] for f in rec6["flags"]]
    _check("T6 surviving content with no chain -> high_residual",
           "high_residual" in names6, failures)

    # T7 brittle surface: full response to original, empty after shuffle
    inp7 = {"claim": "the hidden trait travels through number sequences",
            "reason_chain": ["numbers carry the signal"],
            "lineage": "test",
            "responses": {"original": "the hidden trait travels through "
                                      "number sequences",
                          "steps": ["unrelated tokens only"]},
            "ladder": [{"method": "full_shuffle", "params": {}}]}
    rec7 = audit(inp7)
    names7 = [f["flag"] for f in rec7["flags"]]
    _check("T7 collapse after full_shuffle -> brittle_surface",
           "brittle_surface" in names7, failures)

    # T8 RC is VOID (not zero) when a chain exists but nothing survived
    _check("T8 reason_coverage VOID when nothing survives",
           "void" in rec7["reason_coverage"], failures)

    # T9 unknown method refuses, never silently skipped
    try:
        run_ladder("x", [{"method": "teleport", "params": {}}], 0)
        _check("T9 unknown destruction method refuses", False, failures)
    except Void:
        _check("T9 unknown destruction method refuses", True, failures)

    # T10 compare with one lineage -> kin_only UNCHECKED, not absent
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        append_record(d, rec6)
        out = cmd_compare(d, rec6["claim_id"])
        _check("T10 single-lineage compare is UNCHECKED",
               out["kin_only"]["state"] == "UNCHECKED", failures)

    # T11 external rung: records producer text verbatim; refuses without it
    r11 = run_ladder("a claim", [{"method": "external",
                                  "params": {"text": "a rewritten claim",
                                             "producer": "test-paraphraser"}}],
                     0)
    _check("T11 external rung carries producer text",
           r11[0]["output"] == "a rewritten claim", failures)
    try:
        run_ladder("a claim", [{"method": "external", "params": {}}], 0)
        _check("T11 external rung refuses missing params", False, failures)
    except Void:
        _check("T11 external rung refuses missing params", True, failures)

    print(f"selftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ------------------------------------------------------------------ cli

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="residual_audit",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("destroy", "score"):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("archive_dir")
    p.add_argument("--claim", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "selftest":
            return selftest()
        if args.cmd == "destroy":
            out = cmd_destroy(_load(args.input), args.out)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "score":
            rec = audit(_load(args.input))
            path = append_record(args.out, rec)
            print(json.dumps(rec, indent=2, ensure_ascii=False))
            print(f"archived -> {path}", file=sys.stderr)
            return 0
        if args.cmd == "compare":
            print(json.dumps(cmd_compare(args.archive_dir, args.claim),
                             indent=2, ensure_ascii=False))
            return 0
    except Void as e:
        print(f"VOID: {e}", file=sys.stderr)
        return 3
    except (KeyError, json.JSONDecodeError) as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
