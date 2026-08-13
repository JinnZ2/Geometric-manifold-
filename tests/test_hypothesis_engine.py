"""Offline tests for scripts/hypothesis_engine.py (dry-run / sample data only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hypothesis_engine as he  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "scripts" / "sample_findings.json"


@pytest.fixture()
def topics():
    return [
        {"name": "calibration and falsifiability of LLM agents",
         "queries": ["calibration"], "sources": []},
        {"name": "hidden variable detection / causal discovery from residuals",
         "queries": ["residual"], "sources": []},
    ]


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    return tmp_path


def run_explore(topics):
    return he.stage_explore(topics, max_per_topic=5, dry_run=True, sample_path=SAMPLE)


def test_explore_dry_run_uses_sample(topics):
    findings = run_explore(topics)
    assert len(findings) == 5
    assert {f.topic for f in findings} == {t["name"] for t in topics}


def test_dedup_idempotency(topics, workspace):
    log_path = workspace / "data" / "findings_log.jsonl"
    findings = run_explore(topics)
    new1, skipped1 = he.stage_log(findings, log_path)
    assert len(new1) == 5 and skipped1 == 0
    # second run with identical findings changes nothing
    new2, skipped2 = he.stage_log(run_explore(topics), log_path)
    assert new2 == [] and skipped2 == 5
    assert len(he.read_jsonl(log_path)) == 5


def test_claim_creation_and_falsifiability_routing(topics, workspace):
    log_path = workspace / "data" / "findings_log.jsonl"
    unknown = workspace / "data" / "unknown_journal.jsonl"
    new, _ = he.stage_log(run_explore(topics), log_path)
    tree = he.DependencyTree()
    made, unknown_count = he.stage_claim(new, tree, unknown)
    # the hedged "might/perhaps" sample entry routes to unknown journal
    assert unknown_count == 1
    assert len(made) == 4
    rows = he.read_jsonl(unknown)
    assert rows[0]["flag"] == "unfalsifiable"
    for c in made:
        assert c.falsification
        assert c.scope["topic"]


def test_reformulation_escape_hatch(workspace):
    unknown = workspace / "data" / "unknown_journal.jsonl"
    reform = workspace / "data" / "reformulations.jsonl"
    tree = he.DependencyTree()
    claim = he.Claim(text="test claim", falsification="replication contradicts",
                     scope={"topic": "t"})
    tree.add_claim(claim)
    for i in range(3):
        claim.failed = 3  # force falsified
        stats = he.stage_modify(tree, unknown, reform)
    assert stats["escape_hatched"] == 1
    assert claim.reformulation_count == 3
    assert claim.id not in tree.claims
    rows = he.read_jsonl(unknown)
    assert rows[-1]["flag"] == "escape-hatch"


def test_hidden_variable_scan_triggers(workspace):
    hidden = workspace / "data" / "hidden_variables.jsonl"
    tree = he.DependencyTree()
    # synthetic claims whose residuals correlate with findings rate
    for i, (p, f) in enumerate([(5, 0), (0, 5), (5, 0), (0, 5), (5, 0), (0, 5)]):
        tree.add_claim(he.Claim(text=f"claim {i}", falsification="x",
                                passed=p, failed=f, scope={"topic": "t"}))
    findings = [{"date": f"2024-0{(i % 6) + 1}-01", "topic": "t",
                 "source": "arxiv"} for i in range(12)]
    suggestions = he.stage_hidden(tree, findings, hidden)
    assert suggestions, "expected hidden-variable suggestion on correlated series"
    assert all(s["type"] == "hidden_variable_suggestion" for s in suggestions)
    assert he.read_jsonl(hidden)


def test_hidden_variable_scan_no_trigger_on_flat(workspace):
    hidden = workspace / "data" / "hidden_variables.jsonl"
    tree = he.DependencyTree()
    for i in range(5):
        tree.add_claim(he.Claim(text=f"c{i}", falsification="x",
                                passed=1, failed=1, scope={"topic": "t"}))
    suggestions = he.stage_hidden(tree, [{"date": "2024-01-01", "source": "arxiv"}],
                                  hidden)
    assert suggestions == []  # mean|residual| = |0.5-0.5| = 0 < 0.1


def test_consolidation_writes_hypothesis_md(topics, workspace):
    log_path = workspace / "data" / "findings_log.jsonl"
    unknown = workspace / "data" / "unknown_journal.jsonl"
    hidden = workspace / "data" / "hidden_variables.jsonl"
    new, _ = he.stage_log(run_explore(topics), log_path)
    tree = he.DependencyTree()
    he.stage_claim(new, tree, unknown)
    result = he.stage_consolidate(tree, topics, unknown, hidden,
                                  workspace / "hypotheses")
    files = list((workspace / "hypotheses").glob("*.md"))
    assert files and result["hypothesis_files"] == len(files)
    body = files[0].read_text()
    for section in ("## Supporting claims", "## Contradicted/refuted claims",
                    "## Hidden-variable suspects", "## Open unknowns"):
        assert section in body


def test_claim_tree_save_load_roundtrip(tmp_path):
    tree = he.DependencyTree()
    tree.add_claim(he.Claim(text="a", falsification="f", passed=2, failed=1,
                            scope={"topic": "t"}, source_url="http://x"))
    tree.add_claim(he.Claim(text="b", falsification="f2", reformulation_count=1,
                            scope={"topic": "u", "restrictions": ["narrow"]}))
    path = tmp_path / "claim_tree.json"
    he.save_tree(tree, path)
    loaded = he.load_tree(path)
    assert set(loaded.claims) == set(tree.claims)
    a = next(c for c in loaded.claims.values() if c.text == "a")
    assert (a.passed, a.failed, a.source_url) == (2, 1, "http://x")
    b = next(c for c in loaded.claims.values() if c.text == "b")
    assert b.reformulation_count == 1
    # missing file -> fresh tree
    assert len(he.load_tree(tmp_path / "nope.json").claims) == 0


def test_corroboration_heuristic():
    pos = "We demonstrate that calibration improves accuracy by 18% on agent benchmarks"
    cor = "We confirm and validate that calibration improves accuracy on agent benchmarks"
    con = "Calibration fails to improve accuracy and agents underperform benchmarks"
    unrel = "Quantum chromodynamics lattice results for meson spectra"
    assert he.corroboration(pos, cor) == 1
    assert he.corroboration(pos, con) == -1
    assert he.corroboration(pos, unrel) == 0


def test_full_dry_run_main(topics, workspace, monkeypatch):
    cfg = workspace / "topics.json"
    cfg.write_text(json.dumps({"topics": topics}))
    rc = he.main(["--config", str(cfg), "--dry-run", "--max-per-topic", "3",
                  "--data-dir", str(workspace / "data"),
                  "--hypotheses-dir", str(workspace / "hypotheses"),
                  "--sample", str(SAMPLE)])
    assert rc == 0
    assert (workspace / "data" / "engine_report.md").exists()
    assert (workspace / "data" / "claim_tree.json").exists()
    # idempotent second run: no new findings
    rc = he.main(["--config", str(cfg), "--dry-run",
                  "--data-dir", str(workspace / "data"),
                  "--hypotheses-dir", str(workspace / "hypotheses"),
                  "--sample", str(SAMPLE)])
    assert rc == 0
    assert len(he.read_jsonl(workspace / "data" / "findings_log.jsonl")) == 5
