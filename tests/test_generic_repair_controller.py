"""
Tests for GenericRepairController and stdlib math primitives.

Invariants verified
-------------------
1. Trust region: delta_norm <= trust_radius after every step
2. Confidence bounded: confidence in [0, 1] for all inputs
3. Finite outputs: no NaN or inf in theta after any step
4. Saddle-point sign: saddle_objective = task - lambda * safety (minus intentional)
5. KL basin boundary: in_basin iff KL(softmax(theta) || softmax(ref)) < epsilon_basin
6. FD-HVP linearity: fd_hvp(f, x, alpha*v) = alpha * fd_hvp(f, x, v)
7. Claim table structure: 6 claims, required keys present
8. Convergence on quadratic: theta moves toward reference over many steps
9. RepairState keys: all required fields present after step()
10. Constraint violations recorded when constraint_fn triggers
"""

import math
import pytest

from repair.generic_repair_controller import (
    GenericRepairController,
    RepairState,
    _fd_gradient,
    _fd_hvp,
    _kl,
    _norm,
    _softmax,
    _kappa_eff,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

REF = [0.0, 0.0, 0.0, 0.0]
TASK_FN = lambda x: sum(xi**2 for xi in x) / 2.0
SAFE_FN = lambda x: sum((xi - ri)**2 for xi, ri in zip(x, REF)) / 2.0

BASE_CONFIG = {
    "lr": 0.02,
    "lambda_safety": 10.0,
    "trust_radius": 0.05,
    "epsilon_basin": 0.5,
    "repair_budget": 50.0,
    "spectral_C_bound": 10.0,
    "fd_epsilon": 1e-4,
    "mu_repair": 0.1,
    "mu_max": 5.0,
    "curvature_weight": 2.0,
    "confidence_dist_scale": 0.1,
}


@pytest.fixture
def ctrl():
    return GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="test")


@pytest.fixture
def theta_start():
    return [1.0, -0.5, 0.8, -1.2]


# ─────────────────────────────────────────────────────────────────────────────
# Math primitives
# ─────────────────────────────────────────────────────────────────────────────

def test_fd_gradient_quadratic():
    """Gradient of f(x) = 0.5 * sum(x^2) should be x."""
    f = lambda x: sum(xi**2 for xi in x) / 2.0
    x = [1.0, -2.0, 3.0]
    g = _fd_gradient(f, x, eps=1e-5)
    for gi, xi in zip(g, x):
        assert abs(gi - xi) < 1e-4, f"Expected grad={xi}, got {gi}"


def test_fd_hvp_linear_quadratic():
    """For f(x) = 0.5 x^T A x with A = diag(2,3,4), H v = A v."""
    a = [2.0, 3.0, 4.0]
    f = lambda x: sum(a[i] * x[i]**2 for i in range(3)) / 2.0
    x = [0.5, -1.0, 0.3]
    v = [1.0, 0.0, 0.0]
    hvp = _fd_hvp(f, x, v, eps=1e-4)
    # H v = diag(2,3,4) * (1,0,0) = (2,0,0)
    assert abs(hvp[0] - 2.0) < 1e-3
    assert abs(hvp[1]) < 1e-3
    assert abs(hvp[2]) < 1e-3


def test_fd_hvp_scales_linearly_with_v():
    """fd_hvp(f, x, alpha*v) == alpha * fd_hvp(f, x, v)."""
    f = lambda x: sum(xi**2 for xi in x) / 2.0
    x = [0.5, -0.3, 0.7, 0.1]
    v = [1.0, -1.0, 0.5, 2.0]
    alpha = 3.0
    av = [alpha * vi for vi in v]
    hvp_v  = _fd_hvp(f, x, v, eps=1e-4)
    hvp_av = _fd_hvp(f, x, av, eps=1e-4)
    for h1, h2 in zip(hvp_v, hvp_av):
        assert abs(h2 - alpha * h1) < 1e-3, f"Linearity broken: {h2} != {alpha}*{h1}"


def test_fd_hvp_zero_v_returns_zeros():
    """Zero v → zero HVP (early exit guard)."""
    f = lambda x: sum(xi**2 for xi in x)
    x = [1.0, 2.0]
    hvp = _fd_hvp(f, x, [0.0, 0.0])
    assert hvp == [0.0, 0.0]


def test_softmax_sums_to_one():
    x = [1.0, 2.0, -0.5, 3.0]
    s = _softmax(x)
    assert abs(sum(s) - 1.0) < 1e-12
    assert all(si > 0 for si in s)


def test_kl_self_is_zero():
    p = [0.25, 0.25, 0.25, 0.25]
    assert _kl(p, p) < 1e-12


def test_kl_non_negative():
    p = [0.1, 0.4, 0.5]
    q = [0.3, 0.3, 0.4]
    assert _kl(p, q) >= 0.0


def test_norm_known_value():
    assert abs(_norm([3.0, 4.0]) - 5.0) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Trust region invariant
# ─────────────────────────────────────────────────────────────────────────────

def test_trust_region_single_step(ctrl, theta_start):
    """delta_norm <= trust_radius after a single step."""
    _, state = ctrl.step(theta_start)
    assert state.delta_norm <= ctrl.trust_radius + 1e-9, (
        f"Trust region violated: delta_norm={state.delta_norm} > trust_radius={ctrl.trust_radius}"
    )


def test_trust_region_all_steps(theta_start):
    """delta_norm <= trust_radius for every step over a 20-step run."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="test")
    theta = theta_start[:]
    for _ in range(20):
        theta, state = ctrl.step(theta)
        assert state.delta_norm <= ctrl.trust_radius + 1e-9, (
            f"Trust region violated at step {state.step}: "
            f"delta_norm={state.delta_norm} > {ctrl.trust_radius}"
        )


def test_trust_region_large_lr():
    """Even with lr=100 (stress test), trust region must hold."""
    cfg = {**BASE_CONFIG, "lr": 100.0, "mu_repair": 0.0}
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, cfg, domain="stress")
    theta = [5.0, -3.0, 2.0, 1.0]
    _, state = ctrl.step(theta)
    assert state.delta_norm <= ctrl.trust_radius + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bounded [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_bounded_at_reference(ctrl):
    """Confidence at reference should be 1.0 (KL=0, dist=0)."""
    conf = ctrl._confidence(REF, kl=0.0)
    assert 0.0 <= conf <= 1.0

def test_confidence_bounded_far_away(ctrl):
    """Confidence far from reference is in [0, 1]."""
    far = [100.0, -100.0, 100.0, -100.0]
    kl = ctrl._kl_from_reference(far)
    conf = ctrl._confidence(far, kl)
    assert 0.0 <= conf <= 1.0


def test_confidence_monotone_with_kl(ctrl):
    """Higher KL → lower or equal confidence (monotone penalty)."""
    c0 = ctrl._confidence(REF, kl=0.0)
    c1 = ctrl._confidence(REF, kl=0.5)
    c2 = ctrl._confidence(REF, kl=2.0)
    assert c0 >= c1 >= c2


def test_confidence_bounded_all_steps(theta_start):
    """confidence in [0, 1] for all 20 steps."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="test")
    theta = theta_start[:]
    for _ in range(20):
        theta, state = ctrl.step(theta)
        assert 0.0 <= state.confidence <= 1.0, (
            f"Confidence out of bounds at step {state.step}: {state.confidence}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Finite outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_finite_outputs_normal(ctrl, theta_start):
    """All theta values are finite after a normal step."""
    theta_new, state = ctrl.step(theta_start)
    assert all(math.isfinite(t) for t in theta_new)
    assert math.isfinite(state.task_loss)
    assert math.isfinite(state.safety_loss)
    assert math.isfinite(state.kl_from_reference)


def test_finite_outputs_extreme_start():
    """Even from an extreme starting point, output theta is finite."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="extreme")
    theta = [1e6, -1e6, 1e6, -1e6]
    theta_new, state = ctrl.step(theta)
    assert all(math.isfinite(t) for t in theta_new)


# ─────────────────────────────────────────────────────────────────────────────
# Saddle-point sign
# ─────────────────────────────────────────────────────────────────────────────

def test_saddle_objective_sign(ctrl, theta_start):
    """saddle_objective == task_loss - lambda * safety_loss (minus sign)."""
    _, state = ctrl.step(theta_start)
    expected = round(state.task_loss - ctrl.lambda_safety * state.safety_loss, 6)
    assert abs(state.saddle_objective - expected) < 1e-9, (
        f"Saddle sign wrong: got {state.saddle_objective}, expected {expected}"
    )


def test_saddle_objective_direct():
    """Direct check: _saddle_objective returns task - lambda * safe."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="sign_check")
    theta = [1.0, 0.0, 0.0, 0.0]
    task = TASK_FN(theta)
    safe = SAFE_FN(theta)
    expected = task - ctrl.lambda_safety * safe
    got = ctrl._saddle_objective(theta)
    assert abs(got - expected) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# KL basin boundary
# ─────────────────────────────────────────────────────────────────────────────

def test_kl_basin_at_reference(ctrl):
    """theta == ref → KL = 0, in_basin = True."""
    assert ctrl._kl_from_reference(REF) < 1e-12
    assert ctrl._in_basin(REF)


def test_kl_basin_far_away(ctrl):
    """Far from reference: KL > epsilon_basin, in_basin = False."""
    far = [50.0, -50.0, 50.0, -50.0]
    kl = ctrl._kl_from_reference(far)
    assert kl > ctrl.epsilon_basin
    assert not ctrl._in_basin(far)


def test_kl_basin_matches_state(ctrl, theta_start):
    """state.in_basin is consistent with state.kl_from_reference."""
    _, state = ctrl.step(theta_start)
    expected_in_basin = state.kl_from_reference < ctrl.epsilon_basin
    assert state.in_basin == expected_in_basin


# ─────────────────────────────────────────────────────────────────────────────
# Convergence on quadratic
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_reduces_distance():
    """Over 30 steps on a quadratic, distance to reference should decrease."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="convergence")
    theta = [1.0, -0.5, 0.8, -1.2]
    dist_start = _norm([t - r for t, r in zip(theta, REF)])
    for _ in range(30):
        theta, _ = ctrl.step(theta)
    dist_end = _norm([t - r for t, r in zip(theta, REF)])
    assert dist_end < dist_start, (
        f"Distance did not decrease: {dist_start:.4f} → {dist_end:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RepairState keys
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_STATE_KEYS = {
    "step", "theta", "task_loss", "safety_loss", "saddle_objective",
    "delta_norm", "trust_radius", "kl_from_reference", "in_basin",
    "repair_energy", "cumulative_repair", "kappa_eff_value", "trend",
    "phase", "confidence", "constraint_violations", "ISS_proof_pending",
}

def test_repair_state_keys(ctrl, theta_start):
    """RepairState.as_dict() has all required keys."""
    _, state = ctrl.step(theta_start)
    d = state.as_dict()
    for key in REQUIRED_STATE_KEYS:
        assert key in d, f"Missing key in RepairState: {key}"


def test_iss_proof_pending_always_true(ctrl, theta_start):
    """ISS_proof_pending is always True — open problem, never falsified."""
    _, state = ctrl.step(theta_start)
    assert state.ISS_proof_pending is True


def test_phase_is_valid_string(ctrl, theta_start):
    """phase is one of 'stable', 'threshold', 'critical'."""
    theta = theta_start[:]
    for _ in range(10):
        theta, state = ctrl.step(theta)
        assert state.phase in ("stable", "threshold", "critical")


# ─────────────────────────────────────────────────────────────────────────────
# Constraint violations
# ─────────────────────────────────────────────────────────────────────────────

def test_constraint_violations_recorded():
    """When constraint_fn reports failure, it appears in state.constraint_violations."""
    def always_fail(x):
        return [("always_fails", False, "this always fails")]

    ctrl = GenericRepairController(
        REF, TASK_FN, SAFE_FN, BASE_CONFIG,
        constraint_fn=always_fail, domain="violations",
    )
    _, state = ctrl.step([1.0, 0.0, 0.0, 0.0])
    assert "always_fails" in state.constraint_violations


def test_constraint_no_violations_when_satisfied():
    """When constraint_fn always passes, violations list is empty."""
    def always_pass(x):
        return [("always_passes", True, "ok")]

    ctrl = GenericRepairController(
        REF, TASK_FN, SAFE_FN, BASE_CONFIG,
        constraint_fn=always_pass, domain="no_violations",
    )
    _, state = ctrl.step([0.1, 0.0, 0.0, 0.0])
    assert state.constraint_violations == []


# ─────────────────────────────────────────────────────────────────────────────
# Claim table structure
# ─────────────────────────────────────────────────────────────────────────────

def test_claim_table_structure(ctrl, theta_start, tmp_path):
    """to_claim_table() returns dict with 6 claims and required top-level keys."""
    ctrl.step(theta_start)
    path = str(tmp_path / "claims.json")
    table = ctrl.to_claim_table(source_id="test", path=path)

    assert "claims" in table
    assert len(table["claims"]) == 6
    assert "source_id" in table
    assert "domain" in table
    assert "total_claims" in table
    assert "summary" in table

    for claim in table["claims"]:
        assert "claim_id" in claim
        assert "claim" in claim
        assert "status" in claim


def test_claim_table_written_to_file(ctrl, theta_start, tmp_path):
    """to_claim_table() writes valid JSON to the given path."""
    import json
    ctrl.step(theta_start)
    path = str(tmp_path / "claims.json")
    ctrl.to_claim_table(path=path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["total_claims"] == 6


# ─────────────────────────────────────────────────────────────────────────────
# Summary keys
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_keys(ctrl, theta_start):
    """summary() returns all expected keys."""
    ctrl.run(theta_start, n_steps=5, verbose=False)
    s = ctrl.summary()
    for key in ("domain", "total_steps", "final_phase", "final_kl",
                "in_basin_final", "final_confidence", "cumulative_repair",
                "peak_kappa_eff", "violations_observed", "ISS_proof_pending",
                "calibration_note"):
        assert key in s, f"Missing key in summary: {key}"
    assert s["ISS_proof_pending"] is True


def test_summary_empty_before_run():
    """summary() on fresh controller returns empty dict."""
    ctrl = GenericRepairController(REF, TASK_FN, SAFE_FN, BASE_CONFIG, domain="empty")
    assert ctrl.summary() == {}


# ─────────────────────────────────────────────────────────────────────────────
# Configurable confidence_dist_scale
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_dist_scale_configurable():
    """confidence_dist_scale from config is used; higher scale → lower confidence at distance."""
    theta_far = [2.0, 0.0, 0.0, 0.0]

    ctrl_low = GenericRepairController(
        REF, TASK_FN, SAFE_FN, {**BASE_CONFIG, "confidence_dist_scale": 0.01}, domain="low"
    )
    ctrl_high = GenericRepairController(
        REF, TASK_FN, SAFE_FN, {**BASE_CONFIG, "confidence_dist_scale": 10.0}, domain="high"
    )

    kl = ctrl_low._kl_from_reference(theta_far)
    c_low  = ctrl_low._confidence(theta_far, kl)
    c_high = ctrl_high._confidence(theta_far, kl)
    assert c_low > c_high, (
        f"Higher dist_scale should reduce confidence: low={c_low:.4f}, high={c_high:.4f}"
    )
