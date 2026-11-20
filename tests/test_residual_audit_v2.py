"""residual_audit_v2.py: ra.manifest hash chain and channel_scope (CHANNEL_SCOPE.md section 4)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "docs" / "research" / "residual-audit"
PATH = DIR / "residual_audit_v2.py"
ORIGINAL_SHA256 = "53ccbfed03b40054d10b4e385dd3dd1c5f736e072f2dc8947bf04ff68c1dd4be"


@pytest.fixture(scope="module")
def ra():
    spec = importlib.util.spec_from_file_location("residual_audit_v2", str(PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def inp():
    return json.loads((DIR / "examples" / "owl_claim.json").read_text(encoding="utf-8"))


def test_original_file_is_unchanged():
    got = hashlib.sha256((DIR / "residual_audit.py").read_bytes()).hexdigest()
    assert got == ORIGINAL_SHA256


def test_selftest_passes_t1_to_t12_and_planted_cases(ra, capsys):
    assert ra.selftest() == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    for n in range(1, 13):
        assert f"T{n} " in out


def test_body_is_always_blind(ra, inp):
    for regime in ra.REGIMES:
        assert "body" in ra.audit(inp, regime=regime)["channel_scope"]["blind_channels"]


@pytest.mark.parametrize("blind", [[], ["semantic"], ["semantic", "activation"]])
def test_body_cannot_be_removed_by_an_argument(ra, inp, blind):
    with pytest.raises(ra.ChannelScopeError):
        ra.audit(inp, blind_channels=blind)


def test_regime_outside_the_enum_raises(ra, inp):
    with pytest.raises(ra.ChannelScopeError):
        ra.audit(inp, regime="surface")


def test_out_of_regime_is_void_and_names_the_regime(ra, inp):
    for regime in ra.REGIMES_OUT_OF_REACH:
        rec = ra.audit(inp, regime=regime)
        assert rec["status"] == "VOID"
        assert regime in rec["residual"]["void"]
        assert "interval" not in rec["residual"]


def test_planted_break_in_a_middle_record_is_detected(ra, inp, tmp_path):
    d = str(tmp_path)
    for seed in range(4):
        ra.append_record(d, ra.audit(dict(inp, seed=seed)))
    assert ra.verify_archive(d)["state"] == "CLEAN"
    path = tmp_path / "records.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    mid = len(lines) // 2
    rec = json.loads(lines[mid])
    rec["claim"] = rec.get("claim", "") + " (edited)"
    lines[mid] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v = ra.verify_archive(d)
    assert v["state"] == "MISMATCH"
    assert v["first_mismatch"]["line"] == mid + 1


def test_exp1_archive_reads_unchained_and_is_not_rewritten(ra):
    arch = DIR / "runs" / "exp1" / "archive"
    before = (arch / "records.jsonl").read_bytes()
    v = ra.verify_archive(str(arch))
    assert v["state"] == "UNCHAINED"
    assert v["unchained"] == v["records"] == 12
    assert (arch / "records.jsonl").read_bytes() == before


def test_old_lines_kept_when_a_chained_record_follows(ra, inp, tmp_path):
    src = (DIR / "runs" / "exp1" / "archive" / "records.jsonl").read_bytes()
    (tmp_path / "records.jsonl").write_bytes(src)
    ra.append_record(str(tmp_path), ra.audit(inp))
    assert (tmp_path / "records.jsonl").read_bytes().startswith(src)
    v = ra.verify_archive(str(tmp_path))
    assert v["state"] == "CLEAN" and v["unchained"] == 12


def test_regime_mismatch_in_compare(ra, inp, tmp_path):
    d = str(tmp_path)
    a = ra.audit(dict(inp, lineage="a"), regime="unknown")
    ra.append_record(d, a)
    ra.append_record(d, ra.audit(dict(inp, lineage="b"), regime="vocab_geometry"))
    out = ra.cmd_compare(d, a["claim_id"])
    assert out["regime_mismatch"]["state"] == "FLAGGED"
    assert out["kin_only"]["state"] == "UNCHECKED"
