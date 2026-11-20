#!/usr/bin/env python3
"""Explore / audit harness for residual-tower (see EXPLORE_HARNESS_SPEC.md).

Built to the spec at sha256 fada7cb0fa8726a6b03bad5a5ae13866ab9cb12893701b45aca6ed73317e4f7c.
The harness records what was run, by whom, at what stage, and what came out.
It gives no verdict on whether residual-tower is right.

Commands:
    python3 explore_harness.py self-test
    python3 explore_harness.py run --mode BLIND  --runner-family F --runner-version V \\
            --independence L1 --exposure-file exposure.json
    python3 explore_harness.py commit-expected --reveal-out /path/outside/the/repo.json
    python3 explore_harness.py run --mode REVIEW --runner-family F --runner-version V \\
            --independence L1 --exposure-file exposure.json
    python3 explore_harness.py validate          # every record in runs/records.jsonl
    python3 explore_harness.py report            # missing first, then supported

What is built: the record schema and its validator (all seven modes of the
spec are known to the validator; IMPLEMENT is rejected), the append-only
log, the probes the spec marks RUNNABLE, and the run paths for BLIND and
REVIEW. Not built: run paths for AUDIT, SUGGEST, REVIEW-2, AUDIT-2; the
linear-delta and snapshot-delta-* probes (recorded as BLOCKED / NOT_RUN, no
code path); exposure detection (VOID in the spec).

Expected outputs are kept as COMMITMENTS: expected/commitments.json holds only
sha256(raw output) per probe per stack pin. The plain outputs go to a file
outside the repository (`commit-expected --reveal-out`).

Exit codes: 0 ok; 1 a check failed or a record was rejected; 2 usage.
Stdlib only. Read-only on source files; writes only runs/ and expected/.
Co-authored (JinnZ2 + Claude).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
TOWER = HERE / "residual_tower.py"
DEMO = HERE / "demo_record.jsonl"
RUNS_DIR = HERE / "runs"
RUNS_FILE = RUNS_DIR / "records.jsonl"
EXPECTED_DIR = HERE / "expected"
COMMITMENTS = EXPECTED_DIR / "commitments.json"

SPEC_SHA256 = "fada7cb0fa8726a6b03bad5a5ae13866ab9cb12893701b45aca6ed73317e4f7c"
TOWER_PIN = "94fcd1e8679a6f65375f4034b00cd8a88f93c0bd14a39128eb47f6e85ece92d9"
CANONICAL_STACK = {"top_range": 4, "levels": 5, "codes_per_level": 16}
DEMO_STACK = {"top_range": 4, "levels": 4, "codes_per_level": 16}   # v0 demo, residual_tower.py

VOID = "void"

# --- schema (spec: "Modes" and "Record format") ----------------------------

MODE_TAGS = {
    "BLIND": {"OBSERVED", "VOID"},
    "REVIEW": {"OBSERVED", "VOID"},
    "AUDIT": {"OBSERVED", "DERIVED", "PROPOSED", "VOID"},
    "SUGGEST": {"PROPOSED"},
    "REVIEW-2": {"OBSERVED", "VOID"},
    "AUDIT-2": {"OBSERVED", "DERIVED", "VOID"},
}
REJECTED_MODES = {"IMPLEMENT"}
DIFF_MODES = {"REVIEW", "REVIEW-2", "AUDIT-2"}
FIELDS = [
    "run_id", "mode", "runner_family", "runner_version", "stack_level", "stack_spec",
    "independence_level", "recorded_at", "harness_sha256", "input_sha256s", "commit",
    "probe", "raw_output", "expected", "diff", "status_tag", "exposure", "notes", "observer",
]
LEVELS = {"L0", "L1", "L2", "L3", "L4"}
OBSERVER_VOID = {"human_interest": VOID, "ai_interest": VOID, "divergence_note": VOID}
STACK_KEYS = {"top_range", "levels", "codes_per_level", "residual_tower_sha256"}


class SchemaError(Exception):
    """A record the harness refuses to write."""


class BlindViolation(Exception):
    """BLIND mode touched expected/."""


def canonical(obj):
    # [CHOICE] the spec says "canonical JSON (sort_keys)"; separators and
    # ASCII escaping are fixed here.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_run_id(rec):
    body = {k: v for k, v in rec.items() if k != "run_id"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def validate_record(rec):
    """Raise SchemaError naming the first violation; return None when valid."""
    if not isinstance(rec, dict):
        raise SchemaError("record is not an object")
    mode = rec.get("mode")
    if mode in REJECTED_MODES:
        raise SchemaError("mode %s is schema-invalid: the harness implements nothing" % mode)
    missing = [f for f in FIELDS if f not in rec]
    if missing:
        raise SchemaError("missing field(s): %s (void is a value, not an omission)"
                          % ", ".join(missing))
    extra = [k for k in rec if k not in FIELDS]
    if extra:
        raise SchemaError("unknown field(s): %s" % ", ".join(sorted(extra)))
    if mode not in MODE_TAGS:
        raise SchemaError("unknown mode %r" % (mode,))
    if rec["status_tag"] not in MODE_TAGS[mode]:
        raise SchemaError("status_tag %r not allowed in mode %s (allowed: %s)"
                          % (rec["status_tag"], mode, ", ".join(sorted(MODE_TAGS[mode]))))
    for f in ("runner_family", "runner_version", "recorded_at", "harness_sha256", "commit",
              "probe", "raw_output", "expected", "notes"):
        if not isinstance(rec[f], str) or rec[f] == "":
            raise SchemaError("field %s must be a non-empty string (use \"void\")" % f)
    if rec["stack_level"] != VOID and rec["stack_level"] not in LEVELS:
        raise SchemaError("stack_level must be L0-L4 or void")
    if rec["independence_level"] not in LEVELS:
        raise SchemaError("independence_level must be L0-L4")
    if rec["independence_level"] in ("L3", "L4"):
        raise SchemaError("independence_level %s cannot be claimed: exposure detection is "
                          "VOID in the spec, so exposure cannot be checked"
                          % rec["independence_level"])
    ss = rec["stack_spec"]
    if ss != VOID and not (isinstance(ss, dict) and set(ss) == STACK_KEYS):
        raise SchemaError("stack_spec must be void or {top_range, levels, codes_per_level, "
                          "residual_tower_sha256}")
    if not isinstance(rec["input_sha256s"], dict):
        raise SchemaError("input_sha256s must be a map")
    if not isinstance(rec["exposure"], list) or \
            not all(isinstance(x, str) for x in rec["exposure"]):
        raise SchemaError("exposure must be a list of strings")
    if rec["observer"] != OBSERVER_VOID:
        raise SchemaError("observer slots stay void until an external party fills them")
    if mode == "BLIND" and rec["expected"] != VOID:
        raise SchemaError("BLIND: expected must be the string void")
    if mode in DIFF_MODES:
        if rec["diff"] != VOID and not isinstance(rec["diff"], list):
            raise SchemaError("diff must be a list or void")
    elif rec["diff"] != VOID:
        raise SchemaError("diff must be void in mode %s" % mode)
    if rec["run_id"] != compute_run_id(rec):
        raise SchemaError("run_id does not match the record")


def append_record(rec, path=None):
    """Validate, then append one line and fsync. Nothing is written if invalid."""
    validate_record(rec)
    path = Path(path) if path else RUNS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_records(path=None):
    path = Path(path) if path else RUNS_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- BLIND guard -------------------------------------------------------------

_BLIND_ACTIVE = False
_HOOK_INSTALLED = False
_EXPECTED_TOUCHES = []          # paths under expected/ opened while BLIND was active


def _under_expected(p):
    try:
        rp = os.path.realpath(os.fsdecode(p))
    except (TypeError, ValueError):
        return False
    root = os.path.realpath(str(EXPECTED_DIR))
    return rp == root or rp.startswith(root + os.sep)


def _audit(event, args):
    if not _BLIND_ACTIVE:
        return
    if event in ("open", "os.listdir", "os.scandir") and args and \
            isinstance(args[0], (str, bytes, os.PathLike)) and _under_expected(args[0]):
        _EXPECTED_TOUCHES.append(os.fsdecode(args[0]))
        raise BlindViolation("BLIND mode must not read expected/: %s" % (args[0],))


def _blind(on):
    global _BLIND_ACTIVE, _HOOK_INSTALLED
    if on and not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    _BLIND_ACTIVE = on


def load_commitments():
    if _BLIND_ACTIVE:
        raise BlindViolation("BLIND mode must not read expected/")
    if not COMMITMENTS.exists():
        return {}
    return json.loads(COMMITMENTS.read_text(encoding="utf-8"))


# --- inputs ------------------------------------------------------------------

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pin_key(tower_sha, stack):
    return "%s|top_range=%s|levels=%s|codes_per_level=%s" % (
        tower_sha, stack["top_range"], stack["levels"], stack["codes_per_level"])


def repo_commit():
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else VOID
    except (OSError, subprocess.SubprocessError):
        return VOID


def _load_tower():
    import importlib.util
    spec = importlib.util.spec_from_file_location("residual_tower_under_test", str(TOWER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- probes (spec: "Probe start set") ----------------------------------------

def _cli(args):
    out = subprocess.run([sys.executable, "-B", str(TOWER)] + args, cwd=str(HERE),
                         capture_output=True, text=True, timeout=300)
    return out.stdout + ("\n[exit %d]" % out.returncode)


def _encode(rt, value):
    books = rt.dyadic_stack(**CANONICAL_STACK)
    t = rt.encode_tower(value, books)
    return {"input": str(value),
            "codes": [c if c is not rt.ESCAPE else "ESCAPE" for c in t.codes],
            "residual": str(t.residual),
            "value_equals_input": t.value() == value,
            "per_level": t.per_level()}


def _values(rt, values):
    return json.dumps([_encode(rt, v) for v in values], sort_keys=True, indent=1)


RUNNABLE = {
    # id: (kind, payload, stack, input files, note)
    "selftest": ("cli", ["selftest"], DEMO_STACK, [TOWER], VOID),
    "check-demo": ("cli", ["check", "demo_record.jsonl"], DEMO_STACK, [TOWER, DEMO], VOID),
    "pi-approx": ("values", [Fraction(355, 113)], CANONICAL_STACK, [TOWER], VOID),
    "quarter-tie": ("values", [Fraction(1, 4)], CANONICAL_STACK, [TOWER], VOID),
    "third-nondyadic": ("values", [Fraction(1, 3)], CANONICAL_STACK, [TOWER], VOID),
    "gray-unit-step": ("values", [Fraction(1), Fraction(3, 2)], CANONICAL_STACK, [TOWER],
                       VOID),
    "carry-ripple": ("values", [Fraction(2) - Fraction(1, 7232), Fraction(2)],
                     CANONICAL_STACK, [TOWER], "input under review"),
}
NOT_RUNNABLE = {
    "linear-delta": "BLOCKED - PROPOSED: no incremental path exists in v0; not run; "
                    "no code path in the harness",
    "snapshot-delta-*": "NOT_RUN - PROPOSED: the snapshot-delta artifact is not pinned "
                        "here by sha256; not run; no code path in the harness",
}
PROBE_ORDER = list(RUNNABLE) + list(NOT_RUNNABLE)


def run_probe(probe, rt):
    kind, payload, _stack, _files, _note = RUNNABLE[probe]
    if kind == "cli":
        return _cli(payload)
    return _values(rt, payload)


# --- record building ---------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _base(mode, runner, probe):
    return {
        "mode": mode,
        "runner_family": runner["family"],
        "runner_version": runner["version"],
        "independence_level": runner["independence"],
        "recorded_at": _now(),
        "harness_sha256": sha256_file(__file__),
        "commit": repo_commit(),
        "probe": probe,
        "exposure": list(runner["exposure"]),
        "observer": dict(OBSERVER_VOID),
    }


def _finish(rec):
    rec["run_id"] = compute_run_id(rec)
    return rec


def _stack_fields(probe, tower_sha):
    stack = RUNNABLE[probe][2]
    level = "L%d" % (stack["levels"] - 1)        # whole-stack run: finest level reached
    return level, dict(stack, residual_tower_sha256=tower_sha)


def blind_records(runner):
    """Run every probe with no expected values in reach. Returns records."""
    _blind(True)
    try:
        tower_sha = sha256_file(TOWER)
        pinned = tower_sha == TOWER_PIN
        rt = _load_tower()
        recs = []
        for probe in PROBE_ORDER:
            rec = _base("BLIND", runner, probe)
            rec["expected"] = VOID
            rec["diff"] = VOID
            if probe in NOT_RUNNABLE:
                rec.update(stack_level=VOID, stack_spec=VOID, input_sha256s={},
                           raw_output=VOID, status_tag="VOID", notes=NOT_RUNNABLE[probe])
            else:
                level, spec = _stack_fields(probe, tower_sha)
                files = RUNNABLE[probe][3]
                note = RUNNABLE[probe][4]
                rec.update(stack_level=level, stack_spec=spec,
                           input_sha256s={p.name: sha256_file(p) for p in files},
                           raw_output=run_probe(probe, rt))
                if pinned:
                    rec.update(status_tag="OBSERVED", notes=note)
                else:           # spec rule 1: unpinned input -> VOID, not wrong
                    rec.update(status_tag="VOID",
                               notes="unpinned input: residual_tower.py sha256 differs "
                                     "from the pin" + ("" if note == VOID else "; " + note))
            recs.append(_finish(rec))
        if _EXPECTED_TOUCHES:
            raise BlindViolation("expected/ was touched: %r" % (_EXPECTED_TOUCHES,))
        return recs
    finally:
        _blind(False)


def compare(raw_output, commitment):
    """REVIEW comparison. Returns the diff list; no verdict key anywhere."""
    if sha256_text(raw_output) == commitment:
        return []
    return ["hash mismatch; reveal pending"]


def review_records(runner, blind):
    """Compare the given BLIND records to the commitments. Lists diffs only."""
    commitments = load_commitments()
    cm_sha = sha256_file(COMMITMENTS) if COMMITMENTS.exists() else VOID
    recs = []
    for b in blind:
        rec = _base("REVIEW", runner, b["probe"])
        rec.update(stack_level=b["stack_level"], stack_spec=b["stack_spec"],
                   raw_output=b["raw_output"])
        inputs = dict(b["input_sha256s"])
        inputs["runs/records.jsonl#" + b["run_id"]] = sha256_text(canonical(b))
        if cm_sha != VOID:
            inputs["expected/commitments.json"] = cm_sha
        rec["input_sha256s"] = inputs
        key = None
        if isinstance(b["stack_spec"], dict):
            key = pin_key(b["stack_spec"]["residual_tower_sha256"], b["stack_spec"])
        commitment = commitments.get(key, {}).get(b["probe"]) if key else None
        if b["status_tag"] != "OBSERVED":
            rec.update(expected=VOID, diff=VOID, status_tag="VOID",
                       notes="BLIND record %s is VOID; nothing to compare" % b["run_id"][:12])
        elif commitment is None:
            rec.update(expected=VOID, diff=VOID, status_tag="VOID",
                       notes="no commitment for this probe at this stack pin; reviews BLIND "
                             "record %s" % b["run_id"][:12])
        else:
            note = "reviews BLIND record %s" % b["run_id"]
            if b["notes"] != VOID:
                note += "; " + b["notes"]
            rec.update(expected=commitment, diff=compare(b["raw_output"], commitment),
                       status_tag="OBSERVED", notes=note)
        recs.append(_finish(rec))
    return recs


def latest_blind(records):
    """The most recent BLIND record per probe (file order)."""
    last = {}
    for r in records:
        if r["mode"] == "BLIND":
            last[r["probe"]] = r
    return [last[p] for p in PROBE_ORDER if p in last]


# --- commands ----------------------------------------------------------------

def cmd_run(args):
    exposure = json.loads(Path(args.exposure_file).read_text(encoding="utf-8"))
    if not isinstance(exposure, list):
        print("error: exposure file must hold a JSON list of strings", file=sys.stderr)
        return 2
    runner = {"family": args.runner_family, "version": args.runner_version,
              "independence": args.independence, "exposure": exposure}
    if args.mode == "BLIND":
        recs = blind_records(runner)
    elif args.mode == "REVIEW":
        blind = latest_blind(read_records())
        if not blind:
            print("error: no BLIND records in runs/ to review", file=sys.stderr)
            return 1
        recs = review_records(runner, blind)
    else:
        print("error: the harness has a run path for BLIND and REVIEW only; mode %s is "
              "not run by the harness" % args.mode, file=sys.stderr)
        return 1 if args.mode in REJECTED_MODES else 2
    for r in recs:
        validate_record(r)          # all or nothing
    for r in recs:
        append_record(r)
    tags = {}
    for r in recs:
        tags[r["status_tag"]] = tags.get(r["status_tag"], 0) + 1
    print("%s: %d records appended to %s  %s" % (args.mode, len(recs),
                                                 RUNS_FILE.relative_to(HERE), tags))
    if args.mode == "REVIEW":
        for r in recs:
            d = r["diff"]
            print("  %-18s diff=%s" % (r["probe"], "void" if d == VOID else
                                       ("[] (hash match)" if not d else d)))
    return 0


def cmd_commit_expected(args):
    out = Path(args.reveal_out).resolve()
    try:
        top = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        top = ""
    roots = [str(HERE)] + ([os.path.realpath(top)] if top else [])
    if any(str(out) == r or str(out).startswith(r + os.sep) for r in roots):
        print("error: --reveal-out must be outside the repository", file=sys.stderr)
        return 2
    tower_sha = sha256_file(TOWER)
    rt = _load_tower()
    commitments = load_commitments()
    plain = {}
    for probe in RUNNABLE:
        stack = RUNNABLE[probe][2]
        key = pin_key(tower_sha, stack)
        text = run_probe(probe, rt)
        commitments.setdefault(key, {})[probe] = sha256_text(text)
        plain.setdefault(key, {})[probe] = text
    EXPECTED_DIR.mkdir(exist_ok=True)
    COMMITMENTS.write_text(json.dumps(commitments, sort_keys=True, indent=1) + "\n",
                           encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plain, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print("commitments: %d probes -> %s (sha256 %s)"
          % (len(RUNNABLE), COMMITMENTS.relative_to(HERE), sha256_file(COMMITMENTS)))
    print("plain outputs (not committed) -> %s (sha256 %s)" % (out, sha256_file(out)))
    return 0


def cmd_validate(_args):
    recs = read_records()
    bad = 0
    for i, r in enumerate(recs, 1):
        try:
            validate_record(r)
        except SchemaError as e:
            bad += 1
            print("record %d: %s" % (i, e))
    print("records %d   invalid %d" % (len(recs), bad))
    return 1 if bad else 0


def cmd_report(_args):
    recs = read_records()
    print("Missing first (VOID / not run):")
    for r in recs:
        if r["status_tag"] == "VOID":
            print("  %-7s %-18s not run   source: run_id %s   %s"
                  % (r["mode"], r["probe"], r["run_id"][:12], r["notes"]))
    print("Recorded:")
    for r in recs:
        if r["status_tag"] != "VOID":
            extra = ""
            if r["mode"] in DIFF_MODES:
                extra = "   diff=%s" % ("[]" if r["diff"] == [] else r["diff"])
            if r["notes"] != VOID:
                extra += "   note: %s" % r["notes"]
            print("  %-7s %-18s %s   source: run_id %s%s"
                  % (r["mode"], r["probe"], r["status_tag"], r["run_id"][:12], extra))
    print("No score and no verdict on residual-tower.")
    return 0


# --- self-test ---------------------------------------------------------------

def _valid_example():
    rec = {
        "mode": "BLIND", "runner_family": "self-test", "runner_version": "void",
        "stack_level": "L4",
        "stack_spec": dict(CANONICAL_STACK, residual_tower_sha256=TOWER_PIN),
        "independence_level": "L0", "recorded_at": "2026-11-20T00:00:00+00:00",
        "harness_sha256": "0" * 64, "input_sha256s": {"residual_tower.py": TOWER_PIN},
        "commit": VOID, "probe": "pi-approx", "raw_output": "x", "expected": VOID,
        "diff": VOID, "status_tag": "OBSERVED", "exposure": [], "notes": VOID,
        "observer": dict(OBSERVER_VOID),
    }
    return _finish(rec)


def _rejected(rec):
    try:
        validate_record(rec)
    except SchemaError as e:
        return str(e)
    return None


def self_test():
    import tempfile
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    good = _valid_example()
    check("valid record accepted", _rejected(good) is None)

    # planted schema violations
    bad = dict(good, mode="IMPLEMENT")
    bad["run_id"] = compute_run_id(bad)
    msg = _rejected(bad)
    check("planted IMPLEMENT record rejected", msg is not None and "IMPLEMENT" in msg, msg)

    bad = {k: v for k, v in good.items() if k != "exposure"}
    bad["run_id"] = compute_run_id(bad)
    msg = _rejected(bad)
    check("planted missing field rejected", msg is not None and "exposure" in msg, msg)

    bad = dict(good, status_tag="DERIVED")          # not allowed in BLIND
    bad["run_id"] = compute_run_id(bad)
    msg = _rejected(bad)
    check("planted wrong status_tag rejected", msg is not None and "status_tag" in msg, msg)

    bad = dict(good, mode="SUGGEST", status_tag="OBSERVED")
    bad["run_id"] = compute_run_id(bad)
    check("SUGGEST allows PROPOSED only", _rejected(bad) is not None)

    bad = dict(good, expected="0" * 64)
    bad["run_id"] = compute_run_id(bad)
    check("BLIND with an expected value rejected", _rejected(bad) is not None)

    bad = dict(good, verdict="pass")
    bad["run_id"] = compute_run_id(bad)
    check("a verdict key is rejected", _rejected(bad) is not None)

    bad = dict(good, independence_level="L3")
    bad["run_id"] = compute_run_id(bad)
    check("independence L3 rejected while exposure detection is VOID",
          _rejected(bad) is not None)

    bad = dict(good, raw_output="edited after the fact")
    check("edited record: run_id no longer matches", _rejected(bad) is not None)

    # rejected records are not written; valid ones are appended, one line each
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs" / "records.jsonl"
        try:
            append_record(dict(good, mode="IMPLEMENT"), p)
            wrote = True
        except SchemaError:
            wrote = p.exists()
        check("rejected record leaves no file", not wrote)
        append_record(good, p)
        again = dict(good, recorded_at="2026-11-20T00:00:01+00:00")
        again["run_id"] = compute_run_id(again)
        append_record(again, p)
        got = read_records(p)
        check("append-only: two records, in order",
              len(got) == 2 and got[0] == good and got[1] == again)
        check("identical repeat runs get distinct run_ids", good["run_id"] != again["run_id"])

    # planted wrong output must give a mismatch; the right one must not
    text = "some probe output"
    check("matching output: empty diff", compare(text, sha256_text(text)) == [])
    check("planted wrong output: mismatch",
          compare(text + " tampered", sha256_text(text)) == ["hash mismatch; reveal pending"])

    # BLIND never reads expected/
    runner = {"family": "self-test", "version": "void", "independence": "L0", "exposure": []}
    touched_before = len(_EXPECTED_TOUCHES)
    recs = blind_records(runner)
    check("BLIND run opened nothing under expected/", len(_EXPECTED_TOUCHES) == touched_before)
    check("BLIND records: expected is void in all",
          all(r["expected"] == VOID and r["diff"] == VOID for r in recs))
    check("BLIND records all valid", all(_rejected(r) is None for r in recs))
    _blind(True)
    try:
        try:
            load_commitments()
            raised = False
        except BlindViolation:
            raised = True
        try:
            open(EXPECTED_DIR / "commitments.json").close()
            opened = True
        except BlindViolation:
            opened = False
        except OSError:
            opened = True
    finally:
        _blind(False)
        del _EXPECTED_TOUCHES[touched_before:]
    check("BLIND guard: load_commitments() raises", raised)
    check("BLIND guard: opening a file under expected/ raises (the guard can fire)",
          not opened)

    # probe table matches the spec's start set
    check("probe set", PROBE_ORDER == ["selftest", "check-demo", "pi-approx", "quarter-tie",
                                       "third-nondyadic", "gray-unit-step", "carry-ripple",
                                       "linear-delta", "snapshot-delta-*"])
    by = {r["probe"]: r for r in recs}
    check("linear-delta and snapshot-delta-* recorded VOID with no output",
          all(by[p]["status_tag"] == "VOID" and by[p]["raw_output"] == VOID
              for p in NOT_RUNNABLE))
    check("carry-ripple tagged input under review",
          "input under review" in by["carry-ripple"]["notes"])
    check("record has exactly the spec's 19 fields",
          all(sorted(r) == sorted(FIELDS) for r in recs) and len(FIELDS) == 19)

    fails = [r for r in results if not r["ok"]]
    print(json.dumps({"self_test": "PASS" if not fails else "FAIL", "checks": len(results),
                      "failed": fails}, indent=2))
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("self-test")
    r = sub.add_parser("run")
    r.add_argument("--mode", required=True)
    r.add_argument("--runner-family", required=True)
    r.add_argument("--runner-version", required=True)
    r.add_argument("--independence", required=True)
    r.add_argument("--exposure-file", required=True)
    c = sub.add_parser("commit-expected")
    c.add_argument("--reveal-out", required=True)
    sub.add_parser("validate")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "self-test":
        return self_test()
    if args.cmd == "run":
        try:
            return cmd_run(args)
        except SchemaError as e:
            print("rejected: %s" % e, file=sys.stderr)
            return 1
    if args.cmd == "commit-expected":
        return cmd_commit_expected(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "report":
        return cmd_report(args)
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
