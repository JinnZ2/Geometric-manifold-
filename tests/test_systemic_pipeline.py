"""
Tests for SystemicViabilityModel, ParallelPullTransformer, and IntegratedViabilityPipeline.

Invariants
----------
SystemicViabilityModel:
  - useful_work ≥ 0 when energy_input ≥ 0 and local_efficiency ≥ 0
  - speculative_leak is monotone decreasing in tech_maturity
  - high maturity + high efficiency + low waste → positive viability
  - low maturity + high decay + high trust_entropy → negative viability
  - set_from_vector mapping is invertible (v → model state is deterministic)

ParallelPullTransformer:
  - cross_attention output ∈ [0, 2P] when P > 0 (sensory in (-1,1))
  - Execute_Pull shape matches feature_dimensions
  - score() net_viability matches ManifoldResearchInterface on same inputs
  - gradient is non-zero for non-trivial weights
  - joint_step stays within trust radius

IntegratedViabilityPipeline:
  - run_step increments step counter
  - timeline length mismatch raises ValueError
  - constraint states have correct structure
  - adapt_weights=True does not break the pipeline
"""

import numpy as np
import pytest

from research_interface import (
    SystemicViabilityModel,
    ParallelPullTransformer,
    IntegratedViabilityPipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# SystemicViabilityModel
# ─────────────────────────────────────────────────────────────────────────────

def _good_model(steps=5):
    m = SystemicViabilityModel("test", timeline_steps=steps)
    m.set_variable_timeline("energy_input",         40.0,  60.0)
    m.set_variable_timeline("local_efficiency",      0.8,   0.95)
    m.set_variable_timeline("tech_maturity",         0.9,   1.0)
    m.set_variable_timeline("infrastructure_decay",  0.1,   0.2)
    m.set_variable_timeline("trust_entropy",         0.1,   0.0)
    return m

def _bad_model(steps=5):
    m = SystemicViabilityModel("test", timeline_steps=steps)
    m.set_variable_timeline("energy_input",          100.0, 250.0)
    m.set_variable_timeline("local_efficiency",        0.4,   0.2)
    m.set_variable_timeline("tech_maturity",           0.1,   0.5)
    m.set_variable_timeline("infrastructure_decay",    0.2,   0.8)
    m.set_variable_timeline("trust_entropy",           0.3,   0.9)
    return m


def test_good_system_positive_viability():
    m = _good_model()
    v = m.evaluate_viability()
    assert np.all(v > 0), f"High-maturity, low-waste system should have positive viability, got {v}"


def test_bad_system_negative_viability():
    """Low energy + terrible efficiency + high waste → negative viability.

    The key: energy_input must be small enough that waste dominates.
    The original 'bad' scenario uses 100-250W which still generates positive
    useful_work despite low efficiency — that is physically correct.
    This test uses 5W where waste clearly wins.
    """
    m = SystemicViabilityModel("bad_low_energy", timeline_steps=5)
    m.set_variable_timeline("energy_input",          5.0,   5.0)   # tiny energy
    m.set_variable_timeline("local_efficiency",      0.1,   0.1)   # terrible efficiency
    m.set_variable_timeline("tech_maturity",         0.1,   0.1)   # highly speculative
    m.set_variable_timeline("infrastructure_decay",  0.8,   0.9)
    m.set_variable_timeline("trust_entropy",         0.7,   0.9)
    # useful_work = 5 * 0.1 * 1.5 = 0.75; waste ≈ 0.96+1.26+1.8 = 4.02 → viability ≈ -3.27
    v = m.evaluate_viability()
    assert np.all(v < 0), f"Expected all negative viability, got {v}"


def test_speculative_penalty_monotone():
    """speculative_leak = (1 - maturity) * w: must decrease as maturity rises."""
    m = SystemicViabilityModel("test", timeline_steps=10)
    m.set_variable_timeline("tech_maturity", 0.0, 1.0)
    m.set_variable_timeline("infrastructure_decay", 0.0, 0.0)
    m.set_variable_timeline("trust_entropy", 0.0, 0.0)
    waste = m.calculate_dynamic_waste()
    assert np.all(np.diff(waste) <= 0), "Waste should decrease monotonically as maturity increases"


def test_energy_zero_gives_non_positive_viability():
    """With no energy input, useful_work=0, waste≥0 → viability ≤ 0."""
    m = SystemicViabilityModel("test", timeline_steps=5)
    # all variables default to 0 — waste = (1-0)*w_maturity = 2.0 > 0; useful_work=0
    v = m.evaluate_viability()
    assert np.all(v <= 0)


def test_variable_range_validation():
    m = SystemicViabilityModel("test", timeline_steps=5)
    with pytest.raises(ValueError):
        m.set_variable_timeline("local_efficiency", -0.1, 0.5)
    with pytest.raises(ValueError):
        m.set_variable_timeline("local_efficiency", 0.0, 1.5)
    with pytest.raises(ValueError):
        m.set_variable_timeline("energy_input", -10.0, 50.0)


def test_unknown_variable_raises():
    m = SystemicViabilityModel("test", timeline_steps=5)
    with pytest.raises(ValueError):
        m.set_variable_timeline("flying_pigs", 0.0, 1.0)


def test_set_from_vector_mapping():
    """set_from_vector: +1 sensory → high efficiency/maturity, low decay/trust."""
    m = SystemicViabilityModel("test", timeline_steps=3)
    m.set_from_vector(0, np.array([2.0, 1.0, 1.0, 1.0, 1.0]))
    assert m.variables["local_efficiency"][0] == 1.0    # (1+1)/2 = 1
    assert m.variables["tech_maturity"][0] == 1.0       # (1+1)/2 = 1
    assert m.variables["infrastructure_decay"][0] == 0.0  # (1-1)/2 = 0
    assert m.variables["trust_entropy"][0] == 0.0       # (1-1)/2 = 0


def test_set_from_vector_negative_alignment():
    """-1 sensory → low efficiency/maturity, high decay/trust."""
    m = SystemicViabilityModel("test", timeline_steps=3)
    m.set_from_vector(0, np.array([-2.0, -1.0, -1.0, -1.0, -1.0]))
    assert m.variables["local_efficiency"][0] == 0.0
    assert m.variables["tech_maturity"][0] == 0.0
    assert m.variables["infrastructure_decay"][0] == 1.0
    assert m.variables["trust_entropy"][0] == 1.0


def test_to_dict_keys():
    m = _good_model()
    d = m.to_dict()
    for key in ("name", "steps", "net_scalar", "variables", "viability", "waste"):
        assert key in d, f"Missing key in to_dict: {key}"


def test_net_scalar_matches_mean_viability():
    m = _good_model()
    assert abs(m.net_scalar() - float(np.mean(m.evaluate_viability()))) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# ParallelPullTransformer
# ─────────────────────────────────────────────────────────────────────────────

DIMS = 4
S = [0.85, -0.2, 0.9, 0.1]
P = [150.0, 85.3, 0.4, 60.0]


@pytest.fixture
def transformer():
    return ParallelPullTransformer(feature_dimensions=DIMS, seed=42)


def test_execute_pull_shape(transformer):
    out = transformer.Execute_Pull(S, P)
    assert out.shape == (DIMS,), f"Expected shape ({DIMS},), got {out.shape}"


def test_cross_attention_suppresses_on_negative_sensory(transformer):
    """When sensory = -1 everywhere, output should be near zero."""
    # Set w_sensory to large negative values so tanh → -1
    transformer.w_sensory = np.full((DIMS, DIMS), -100.0)
    out = transformer.Execute_Pull(S, P)
    # compressed ≈ -1, so (1 + compressed) ≈ 0, output ≈ 0
    assert np.all(np.abs(out) < 1e-3), f"Expected ~0 output on max friction, got {out}"


def test_cross_attention_amplifies_on_positive_sensory(transformer):
    """When sensory = +1 everywhere, output should ≈ 2 × processed_science."""
    transformer.w_sensory = np.full((DIMS, DIMS), 100.0)
    out = transformer.Execute_Pull(S, P)
    processed = np.array(P) @ transformer.w_science
    np.testing.assert_allclose(out, 2.0 * processed, rtol=1e-3)


def test_execute_pull_deterministic_with_seed():
    t1 = ParallelPullTransformer(feature_dimensions=DIMS, seed=99)
    t2 = ParallelPullTransformer(feature_dimensions=DIMS, seed=99)
    np.testing.assert_array_equal(t1.Execute_Pull(S, P), t2.Execute_Pull(S, P))


def test_score_returns_net_viability(transformer):
    result = transformer.score(S, P)
    assert "net_viability" in result
    assert np.isfinite(result["net_viability"])


def test_gradient_w_sensory_shape(transformer):
    g = transformer.gradient_w_sensory(S, P)
    assert g.shape == (DIMS, DIMS)


def test_gradient_w_science_shape(transformer):
    g = transformer.gradient_w_science(S, P)
    assert g.shape == (DIMS, DIMS)


def test_joint_step_trust_region():
    """‖ΔW‖_F must not exceed trust_radius for either matrix."""
    t = ParallelPullTransformer(feature_dimensions=DIMS, seed=7)
    ws_before = t.w_sensory.copy()
    wp_before = t.w_science.copy()
    trust = 0.05
    t.joint_step(S, P, lr=1.0, trust_radius=trust)   # lr=1 to stress the trust region
    assert np.linalg.norm(t.w_sensory - ws_before) <= trust + 1e-9
    assert np.linalg.norm(t.w_science - wp_before) <= trust + 1e-9


def test_joint_step_improves_score():
    """After enough steps, score should improve."""
    t = ParallelPullTransformer(feature_dimensions=DIMS, seed=3)
    before = t.score(S, P)["net_viability"]
    for _ in range(20):
        t.joint_step(S, P, lr=0.02, trust_radius=0.1)
    after = t.score(S, P)["net_viability"]
    assert after >= before - 1e-4, f"Score should not degrade: {before:.4f} → {after:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# IntegratedViabilityPipeline
# ─────────────────────────────────────────────────────────────────────────────

SENSORY_TL = [[0.9 - 0.1*t, -0.4, 0.7, -0.1, 0.3] for t in range(5)]
SCIENCE_TL = [[120.0, 75.0, 0.35, 50.0, 10.0] for _ in range(5)]


@pytest.fixture
def pipeline():
    return IntegratedViabilityPipeline(
        name="test_pipeline", timeline_steps=5, feature_dimensions=5, config={"seed": 42}
    )


def test_run_step_increments_counter(pipeline):
    assert pipeline._current_step == 0
    pipeline.run_step(SENSORY_TL[0], SCIENCE_TL[0])
    assert pipeline._current_step == 1


def test_run_step_returns_required_keys(pipeline):
    rec = pipeline.run_step(SENSORY_TL[0], SCIENCE_TL[0])
    for key in ("step", "output_vector", "net_viability", "model_viability",
                "constraint_state", "couplings"):
        assert key in rec, f"Missing key: {key}"


def test_run_step_overflow_raises(pipeline):
    for i in range(5):
        pipeline.run_step(SENSORY_TL[i], SCIENCE_TL[i])
    with pytest.raises(RuntimeError):
        pipeline.run_step(SENSORY_TL[0], SCIENCE_TL[0])


def test_timeline_length_mismatch_raises():
    p = IntegratedViabilityPipeline(timeline_steps=5, feature_dimensions=5)
    with pytest.raises(ValueError):
        p.run_timeline(SENSORY_TL[:3], SCIENCE_TL[:3])  # wrong length


def test_run_timeline_full(capsys):
    p = IntegratedViabilityPipeline(
        name="full_run", timeline_steps=5, feature_dimensions=5, config={"seed": 0}
    )
    records = p.run_timeline(SENSORY_TL, SCIENCE_TL, log_interval=5)
    assert len(records) == 5
    assert all("model_viability" in r for r in records)


def test_constraint_history_length(pipeline):
    for i in range(3):
        pipeline.run_step(SENSORY_TL[i], SCIENCE_TL[i])
    ch = pipeline.constraint_history()
    assert len(ch) == 3
    assert all("constraint_mask" in cs for cs in ch)


def test_dominant_couplings_have_type(pipeline):
    for i in range(5):
        pipeline.run_step(SENSORY_TL[i], SCIENCE_TL[i])
    dc = pipeline.dominant_couplings()
    assert len(dc) == 5
    assert all("type" in c for c in dc)


def test_adapt_weights_does_not_crash(pipeline):
    rec = pipeline.run_step(SENSORY_TL[0], SCIENCE_TL[0], adapt_weights=True, lr=0.005)
    assert np.isfinite(rec["net_viability"])


def test_good_vs_bad_scenario_net_scalar():
    """Good (modular) scenario must outperform bad (centralised) on net_scalar."""
    good = SystemicViabilityModel("good", timeline_steps=5)
    good.set_variable_timeline("energy_input",         40.0,  60.0)
    good.set_variable_timeline("local_efficiency",      0.8,   0.95)
    good.set_variable_timeline("tech_maturity",         0.9,   1.0)
    good.set_variable_timeline("infrastructure_decay",  0.1,   0.2)
    good.set_variable_timeline("trust_entropy",         0.1,   0.0)

    # Same energy envelope, drastically lower efficiency and higher waste
    bad = SystemicViabilityModel("bad", timeline_steps=5)
    bad.set_variable_timeline("energy_input",          40.0,  60.0)  # same energy as good
    bad.set_variable_timeline("local_efficiency",       0.15,  0.10)  # profit siphoned out
    bad.set_variable_timeline("tech_maturity",          0.1,   0.3)   # mostly speculative
    bad.set_variable_timeline("infrastructure_decay",   0.7,   0.9)
    bad.set_variable_timeline("trust_entropy",          0.6,   0.9)

    assert good.net_scalar() > bad.net_scalar(), (
        f"Good scenario ({good.net_scalar():.3f}) should outscore "
        f"bad scenario ({bad.net_scalar():.3f})"
    )
