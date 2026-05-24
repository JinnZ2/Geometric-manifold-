"""
Tests for science_constraint_bridge.py — the bridge to science_constraint_layers.
"""

import pytest
from repair.science_constraint_bridge import to_constraint_state, to_coupling_vector, export_trajectory


SAMPLE_METRICS = {
    "task_loss": 0.4,
    "safety_loss": 0.05,
    "curvature": 1.2,
    "confidence": 0.75,
    "dist_to_ref": 0.03,
}


def test_constraint_state_has_required_keys():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9, step=5)
    for key in ("time", "domain", "state_vector", "constraint_mask", "violated",
                "mathematics", "thermodynamics", "biology"):
        assert key in cs, f"Missing key: {key}"


def test_constraint_state_time_matches_step():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9, step=42)
    assert cs["time"] == 42


def test_state_vector_length():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    assert len(cs["state_vector"]) == 14


def test_constraint_mask_is_booleans():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    assert all(isinstance(v, bool) for v in cs["constraint_mask"])


def test_no_violations_when_healthy():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    assert cs["violated"] == [], f"Expected no violations, got: {cs['violated']}"


def test_violation_flagged_when_confidence_low():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.2, data_conf=0.9)
    # policy_conf=0.2 < threshold 0.4; not directly in constraint list but param_conf matters
    bad = to_constraint_state(
        {**SAMPLE_METRICS, "confidence": 0.1},
        policy_conf=0.2, data_conf=0.9
    )
    assert "confidence_above_threshold" in bad["violated"]


def test_coupling_vector_has_five_entries():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    couplings = to_coupling_vector(cs)
    assert len(couplings) == 5


def test_coupling_types_present():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    types = {c["type"] for c in to_coupling_vector(cs)}
    assert "mathematical_physical" in types
    assert "thermodynamic_biological" in types
    assert "thermodynamic_physical" in types


def test_coupling_strengths_bounded():
    cs = to_constraint_state(SAMPLE_METRICS, policy_conf=0.8, data_conf=0.9)
    for c in to_coupling_vector(cs):
        assert 0.0 <= c["strength"] <= 1.0, f"Coupling {c['type']} strength out of [0,1]"


def test_export_trajectory_length():
    history = [
        {**SAMPLE_METRICS, "policy_confidence": 0.8, "data_confidence": 0.9}
        for _ in range(10)
    ]
    exported = export_trajectory(history)
    assert len(exported) == 10


def test_export_trajectory_has_couplings():
    history = [{**SAMPLE_METRICS, "policy_confidence": 0.7, "data_confidence": 0.85}]
    exported = export_trajectory(history)
    assert "couplings" in exported[0]
    assert len(exported[0]["couplings"]) == 5


def test_high_curvature_raises_math_physical_coupling():
    low_curv = to_constraint_state({**SAMPLE_METRICS, "curvature": 0.1}, policy_conf=0.8, data_conf=0.9)
    high_curv = to_constraint_state({**SAMPLE_METRICS, "curvature": 15.0}, policy_conf=0.8, data_conf=0.9)
    low_c = next(c for c in to_coupling_vector(low_curv) if c["type"] == "mathematical_physical")
    high_c = next(c for c in to_coupling_vector(high_curv) if c["type"] == "mathematical_physical")
    assert high_c["strength"] > low_c["strength"], "Higher curvature must raise math-physical coupling"
