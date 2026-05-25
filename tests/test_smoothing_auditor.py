"""
Tests for repair/smoothing_auditor.py.
"""

import json

import pytest

from repair.smoothing_auditor import (
    DEFAULT_PATTERNS,
    audit_path,
    to_claim_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def py_file(tmp_path):
    """A .py file containing several smoothing patterns."""
    src = tmp_path / "sample.py"
    src.write_text(
        "x = x.clip(0, 1)\ny = min(a, b)\nz = max(a, b)\nw = normalize(v)\nno_match_here = 42\n"
    )
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# audit_path
# ─────────────────────────────────────────────────────────────────────────────


def test_audit_detects_clip(py_file):
    hits = audit_path(str(py_file))
    names = {h.pattern_name for h in hits}
    assert "hard_clamp" in names


def test_audit_detects_min_max(py_file):
    hits = audit_path(str(py_file))
    names = {h.pattern_name for h in hits}
    assert "threshold_min" in names
    assert "threshold_max" in names


def test_audit_detects_normalize(py_file):
    hits = audit_path(str(py_file))
    names = {h.pattern_name for h in hits}
    assert "normalization" in names


def test_audit_no_false_positives_on_clean_file(tmp_path):
    """A file with no smoothing patterns produces no hits."""
    src = tmp_path / "clean.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    hits = audit_path(str(tmp_path))
    assert hits == []


def test_audit_empty_dir_returns_empty(empty_dir):
    assert audit_path(str(empty_dir)) == []


def test_audit_skips_non_source_files(tmp_path):
    """Non-source extensions (.txt, .md) are not scanned."""
    (tmp_path / "notes.txt").write_text("x = x.clip(0, 1)\n")
    (tmp_path / "README.md").write_text("normalize()\n")
    hits = audit_path(str(tmp_path))
    assert hits == []


def test_audit_hit_fields(py_file):
    hits = audit_path(str(py_file))
    assert len(hits) > 0
    h = hits[0]
    assert isinstance(h.file, str)
    assert isinstance(h.line, int) and h.line >= 1
    assert isinstance(h.pattern_name, str)
    assert isinstance(h.pattern, str)
    assert isinstance(h.context, str)


def test_audit_hit_line_numbers_are_correct(tmp_path):
    src = tmp_path / "lines.py"
    src.write_text("a = 1\nb = 2\nc = c.clip(0, 1)\nd = 4\n")
    hits = audit_path(str(tmp_path))
    clip_hits = [h for h in hits if h.pattern_name == "hard_clamp"]
    assert len(clip_hits) == 1
    assert clip_hits[0].line == 3


def test_audit_custom_patterns(tmp_path):
    src = tmp_path / "custom.py"
    src.write_text("result = magic_fn(x)\n")
    custom = {"magic": r"\bmagic_fn\b"}
    hits = audit_path(str(tmp_path), patterns=custom)
    assert len(hits) == 1
    assert hits[0].pattern_name == "magic"


def test_audit_custom_extensions(tmp_path):
    """Only scan the specified extensions."""
    (tmp_path / "code.rb").write_text("x = x.clip(0, 1)\n")
    (tmp_path / "code.py").write_text("x = x.clip(0, 1)\n")
    hits_default = audit_path(str(tmp_path))
    hits_rb = audit_path(str(tmp_path), extensions=frozenset({".rb"}))
    assert len(hits_default) == 1  # only .py
    assert len(hits_rb) == 1  # only .rb


def test_smoothing_hit_as_dict(py_file):
    hits = audit_path(str(py_file))
    d = hits[0].as_dict()
    for key in ("file", "line", "pattern_name", "pattern", "context"):
        assert key in d


# ─────────────────────────────────────────────────────────────────────────────
# to_claim_table
# ─────────────────────────────────────────────────────────────────────────────


def test_claim_table_keys(py_file, tmp_path):
    hits = audit_path(str(py_file))
    table = to_claim_table(hits, source_id="test", path=str(tmp_path / "ct.json"))
    for key in ("source_id", "total_hits", "total_claims", "claims", "note"):
        assert key in table


def test_claim_table_hit_count(py_file, tmp_path):
    hits = audit_path(str(py_file))
    table = to_claim_table(hits, path=str(tmp_path / "ct.json"))
    assert table["total_hits"] == len(hits)


def test_claim_table_claims_have_required_keys(py_file, tmp_path):
    hits = audit_path(str(py_file))
    table = to_claim_table(hits, path=str(tmp_path / "ct.json"))
    for claim in table["claims"]:
        for key in ("claim_id", "claim", "falsification_condition", "hits", "locations", "status"):
            assert key in claim


def test_claim_table_written_to_file(py_file, tmp_path):
    hits = audit_path(str(py_file))
    path = str(tmp_path / "out.json")
    to_claim_table(hits, source_id="test", path=path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["source_id"] == "test"
    assert loaded["total_hits"] == len(hits)


def test_claim_table_empty_hits(tmp_path):
    table = to_claim_table([], source_id="empty", path=str(tmp_path / "ct.json"))
    assert table["total_hits"] == 0
    assert table["claims"] == []


def test_claim_status_is_candidate(py_file, tmp_path):
    hits = audit_path(str(py_file))
    table = to_claim_table(hits, path=str(tmp_path / "ct.json"))
    assert all(c["status"] == "CANDIDATE" for c in table["claims"])


def test_locations_capped_at_ten(tmp_path):
    """locations list is capped — unbounded output would be noisy."""
    src = tmp_path / "many.py"
    src.write_text("\n".join(f"x = x.clip({i}, {i + 1})" for i in range(20)))
    hits = audit_path(str(tmp_path))
    table = to_claim_table(hits, path=str(tmp_path / "ct.json"))
    for claim in table["claims"]:
        assert len(claim["locations"]) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT_PATTERNS completeness
# ─────────────────────────────────────────────────────────────────────────────


def test_default_patterns_is_dict():
    assert isinstance(DEFAULT_PATTERNS, dict)
    assert len(DEFAULT_PATTERNS) >= 6


def test_default_patterns_are_valid_regex():
    import re

    for name, pat in DEFAULT_PATTERNS.items():
        try:
            re.compile(pat)
        except re.error as e:
            pytest.fail(f"Pattern '{name}' is invalid regex: {e}")
