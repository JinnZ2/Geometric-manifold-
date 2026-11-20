"""explore_harness.py: schema rejections, hash commitments, and BLIND never reads expected/."""
import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "docs" / "research" / "residual-tower" / "explore_harness.py"


@pytest.fixture(scope="module")
def eh():
    spec = importlib.util.spec_from_file_location("explore_harness", str(HARNESS))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["explore_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


RUNNER = {"family": "pytest", "version": "void", "independence": "L0", "exposure": []}


def test_self_test_passes(eh):
    assert eh.self_test() == 0


def test_blind_never_reads_expected(eh):
    opened = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            opened.append(str(args[0]))

    sys.addaudithook(hook)
    start = len(opened)
    recs = eh.blind_records(RUNNER)
    seen = opened[start:]
    root = str(eh.EXPECTED_DIR)
    assert not [p for p in seen if p.startswith(root)]
    assert all(r["expected"] == "void" for r in recs)


def test_blind_guard_fires(eh):
    eh._blind(True)
    try:
        with pytest.raises(eh.BlindViolation):
            eh.load_commitments()
        with pytest.raises(eh.BlindViolation):
            open(eh.EXPECTED_DIR / "commitments.json")
    finally:
        eh._blind(False)
        eh._EXPECTED_TOUCHES.clear()


@pytest.mark.parametrize("change", [
    {"mode": "IMPLEMENT"},
    {"status_tag": "DERIVED"},
    {"exposure": None},
])
def test_planted_schema_violations_rejected(eh, change):
    rec = eh._valid_example()
    if change.get("exposure", 0) is None:
        del rec["exposure"]
    else:
        rec.update(change)
    rec["run_id"] = eh.compute_run_id(rec)
    with pytest.raises(eh.SchemaError):
        eh.validate_record(rec)


def test_planted_wrong_output_is_a_mismatch(eh):
    c = eh.sha256_text("out")
    assert eh.compare("out", c) == []
    assert eh.compare("out!", c) == ["hash mismatch; reveal pending"]


def test_committed_records_valid(eh):
    recs = eh.read_records()
    for r in recs:
        eh.validate_record(r)
    assert {r["mode"] for r in recs} <= {"BLIND", "REVIEW"}
