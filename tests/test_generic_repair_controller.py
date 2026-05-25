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

import json
import math

import pytest

from repair.generic_repair_controller import (
    GenericRepairController,
    _fd_gradient,
    _fd_hvp,
    _kl,
    _norm,
    _softmax,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

REF = [0.0, 0.0, 0.0, 0.0]

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


def _task(x):
    return sum(xi**2 for xi in x) / 2.0


def _safe(x):
    return sum((xi - ri) ** 2 for xi, ri in zip(x, REF)) / 2.0


@pytest.fixture
def ctrl():
    return GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="test")


@pytest.fixture
def theta_start():
    return [1.0, -0.5, 0.8, -1.2]


# ─────────────────────────────────────────────────────────────────────────────
# Math primitives
# ─────────────────────────────────────────────────────────────────────────────


def test_fd_gradient_quadratic():
    """Gradient of 0.5·||x||² is x."""

    def f(x):
        return sum(xi**2 for xi in x) / 2.0

    x = [1.0, -2.0, 3.0]
    g = _fd_gradient(f, x, eps=1e-5)
    for gi, xi in zip(g, x):
        assert abs(gi - xi) < 1e-4


def test_fd_hvp_linear_quadratic():
    """For f = 0.5·xᵀ diag(a) x, Hv = diag(a)·v."""
    a = [2.0, 3.0, 4.0]

    def f(x):
        return sum(a[i] * x[i] ** 2 for i in range(3)) / 2.0

    hvp = _fd_hvp(f, [0.5, -1.0, 0.3], [1.0, 0.0, 0.0], eps=1e-4)
    assert abs(hvp[0] - 2.0) < 1e-3
    assert abs(hvp[1]) < 1e-3
    assert abs(hvp[2]) < 1e-3


def test_fd_hvp_scales_linearly_with_v():
    """fd_hvp(f, x, α·v) == α·fd_hvp(f, x, v)."""

    def f(x):
        return sum(xi**2 for xi in x) / 2.0

    x = [0.5, -0.3, 0.7, 0.1]
    v = [1.0, -1.0, 0.5, 2.0]
    alpha = 3.0
    av = [alpha * vi for vi in v]
    hvp_v = _fd_hvp(f, x, v, eps=1e-4)
    hvp_av = _fd_hvp(f, x, av, eps=1e-4)
    for h1, h2 in zip(hvp_v, hvp_av):
        assert abs(h2 - alpha * h1) < 1e-3, f"Linearity broken: {h2} != {alpha}*{h1}"


def test_fd_hvp_zero_v_returns_zeros():
    """Zero v → zero HVP (early-exit guard)."""

    def f(x):
        return sum(xi**2 for xi in x)

    assert _fd_hvp(f, [1.0, 2.0], [0.0, 0.0]) == [0.0, 0.0]


def test_softmax_sums_to_one():
    s = _softmax([1.0, 2.0, -0.5, 3.0])
    assert abs(sum(s) - 1.0) < 1e-12
    assert all(si > 0 for si in s)


def test_kl_self_is_zero():
    p = [0.25, 0.25, 0.25, 0.25]
    assert _kl(p, p) < 1e-12


def test_kl_non_negative():
    assert _kl([0.1, 0.4, 0.5], [0.3, 0.3, 0.4]) >= 0.0


def test_norm_known_value():
    assert abs(_norm([3.0, 4.0]) - 5.0) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Trust region invariant
# ─────────────────────────────────────────────────────────────────────────────


def test_trust_region_single_step(ctrl, theta_start):
    """delta_norm <= trust_radius after a single step."""
    _, state = ctrl.step(theta_start)
    assert state.delta_norm <= ctrl.trust_radius + 1e-9


def test_trust_region_all_steps(theta_start):
    """delta_norm <= trust_radius for every step over a 20-step run."""
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="test")
    theta = theta_start[:]
    for _ in range(20):
        theta, state = ctrl.step(theta)
        assert state.delta_norm <= ctrl.trust_radius + 1e-9, (
            f"Trust region violated at step {state.step}: "
            f"delta_norm={state.delta_norm} > {ctrl.trust_radius}"
        )


def test_trust_region_large_lr():
    """Even with lr=100, trust region must hold."""
    cfg = {**BASE_CONFIG, "lr": 100.0, "mu_repair": 0.0}
    ctrl = GenericRepairController(REF, _task, _safe, cfg, domain="stress")
    _, state = ctrl.step([5.0, -3.0, 2.0, 1.0])
    assert state.delta_norm <= ctrl.trust_radius + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bounded [0, 1]
# ─────────────────────────────────────────────────────────────────────────────


def test_confidence_bounded_at_reference(ctrl):
    assert 0.0 <= ctrl._confidence(REF, kl=0.0) <= 1.0


def test_confidence_bounded_far_away(ctrl):
    far = [100.0, -100.0, 100.0, -100.0]
    kl = ctrl._kl_from_reference(far)
    assert 0.0 <= ctrl._confidence(far, kl) <= 1.0


def test_confidence_monotone_with_kl(ctrl):
    """Higher KL → lower confidence."""
    c0 = ctrl._confidence(REF, kl=0.0)
    c1 = ctrl._confidence(REF, kl=0.5)
    c2 = ctrl._confidence(REF, kl=2.0)
    assert c0 >= c1 >= c2


def test_confidence_bounded_all_steps(theta_start):
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="test")
    theta = theta_start[:]
    for _ in range(20):
        theta, state = ctrl.step(theta)
        assert 0.0 <= state.confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Finite outputs
# ─────────────────────────────────────────────────────────────────────────────


def test_finite_outputs_normal(ctrl, theta_start):
    theta_new, state = ctrl.step(theta_start)
    assert all(math.isfinite(t) for t in theta_new)
    assert math.isfinite(state.task_loss)
    assert math.isfinite(state.safety_loss)
    assert math.isfinite(state.kl_from_reference)


def test_finite_outputs_extreme_start():
    """Output is finite even from a numerically extreme starting point."""
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="extreme")
    theta_new, _ = ctrl.step([1e6, -1e6, 1e6, -1e6])
    assert all(math.isfinite(t) for t in theta_new)


# ─────────────────────────────────────────────────────────────────────────────
# Saddle-point sign
# ─────────────────────────────────────────────────────────────────────────────


def test_saddle_objective_sign(ctrl, theta_start):
    """saddle_objective == task_loss − λ·safety_loss (minus sign preserved)."""
    _, state = ctrl.step(theta_start)
    expected = round(state.task_loss - ctrl.lambda_safety * state.safety_loss, 6)
    assert abs(state.saddle_objective - expected) < 1e-9


def test_saddle_objective_direct():
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="sign_check")
    theta = [1.0, 0.0, 0.0, 0.0]
    expected = _task(theta) - ctrl.lambda_safety * _safe(theta)
    assert abs(ctrl._saddle_objective(theta) - expected) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# KL basin boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_kl_basin_at_reference(ctrl):
    assert ctrl._kl_from_reference(REF) < 1e-12
    assert ctrl._in_basin(REF)


def test_kl_basin_far_away(ctrl):
    far = [50.0, -50.0, 50.0, -50.0]
    assert ctrl._kl_from_reference(far) > ctrl.epsilon_basin
    assert not ctrl._in_basin(far)


def test_kl_basin_matches_state(ctrl, theta_start):
    """state.in_basin is consistent with state.kl_from_reference."""
    _, state = ctrl.step(theta_start)
    assert state.in_basin == (state.kl_from_reference < ctrl.epsilon_basin)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence on quadratic
# ─────────────────────────────────────────────────────────────────────────────


def test_convergence_reduces_distance():
    """Over 30 steps on a quadratic, distance to reference decreases."""
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="convergence")
    theta = [1.0, -0.5, 0.8, -1.2]
    dist_start = _norm([t - r for t, r in zip(theta, REF)])
    for _ in range(30):
        theta, _ = ctrl.step(theta)
    dist_end = _norm([t - r for t, r in zip(theta, REF)])
    assert dist_end < dist_start, f"Distance did not decrease: {dist_start:.4f} → {dist_end:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# RepairState keys
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_STATE_KEYS = {
    "step",
    "theta",
    "task_loss",
    "safety_loss",
    "saddle_objective",
    "delta_norm",
    "trust_radius",
    "kl_from_reference",
    "in_basin",
    "repair_energy",
    "cumulative_repair",
    "kappa_eff_value",
    "trend",
    "phase",
    "confidence",
    "constraint_violations",
    "ISS_proof_pending",
}


def test_repair_state_keys(ctrl, theta_start):
    _, state = ctrl.step(theta_start)
    d = state.as_dict()
    for key in REQUIRED_STATE_KEYS:
        assert key in d, f"Missing key in RepairState: {key}"


def test_iss_proof_pending_always_true(ctrl, theta_start):
    """ISS_proof_pending is always True — open problem, never falsified."""
    _, state = ctrl.step(theta_start)
    assert state.ISS_proof_pending is True


def test_phase_is_valid_string(ctrl, theta_start):
    theta = theta_start[:]
    for _ in range(10):
        theta, state = ctrl.step(theta)
        assert state.phase in ("stable", "threshold", "critical")


# ─────────────────────────────────────────────────────────────────────────────
# Constraint violations
# ─────────────────────────────────────────────────────────────────────────────


def test_constraint_violations_recorded():
    def always_fail(x):
        return [("always_fails", False, "this always fails")]

    ctrl = GenericRepairController(
        REF, _task, _safe, BASE_CONFIG, constraint_fn=always_fail, domain="violations"
    )
    _, state = ctrl.step([1.0, 0.0, 0.0, 0.0])
    assert "always_fails" in state.constraint_violations


def test_constraint_no_violations_when_satisfied():
    def always_pass(x):
        return [("always_passes", True, "ok")]

    ctrl = GenericRepairController(
        REF, _task, _safe, BASE_CONFIG, constraint_fn=always_pass, domain="no_violations"
    )
    _, state = ctrl.step([0.1, 0.0, 0.0, 0.0])
    assert state.constraint_violations == []


# ─────────────────────────────────────────────────────────────────────────────
# Claim table structure
# ─────────────────────────────────────────────────────────────────────────────


def test_claim_table_structure(ctrl, theta_start, tmp_path):
    ctrl.step(theta_start)
    table = ctrl.to_claim_table(source_id="test", path=str(tmp_path / "claims.json"))
    assert len(table["claims"]) == 6
    for key in ("source_id", "domain", "total_claims", "summary"):
        assert key in table
    for claim in table["claims"]:
        for key in ("claim_id", "claim", "status"):
            assert key in claim


def test_claim_table_written_to_file(ctrl, theta_start, tmp_path):
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
    ctrl.run(theta_start, n_steps=5, verbose=False)
    s = ctrl.summary()
    for key in (
        "domain",
        "total_steps",
        "final_phase",
        "final_kl",
        "in_basin_final",
        "final_confidence",
        "cumulative_repair",
        "peak_kappa_eff",
        "violations_observed",
        "ISS_proof_pending",
        "calibration_note",
    ):
        assert key in s, f"Missing key in summary: {key}"
    assert s["ISS_proof_pending"] is True


def test_summary_empty_before_run():
    ctrl = GenericRepairController(REF, _task, _safe, BASE_CONFIG, domain="empty")
    assert ctrl.summary() == {}


# ─────────────────────────────────────────────────────────────────────────────
# Configurable confidence_dist_scale
# ─────────────────────────────────────────────────────────────────────────────


def test_confidence_dist_scale_configurable():
    """Higher confidence_dist_scale → lower confidence at distance."""
    theta_far = [2.0, 0.0, 0.0, 0.0]

    ctrl_low = GenericRepairController(
        REF, _task, _safe, {**BASE_CONFIG, "confidence_dist_scale": 0.01}, domain="low"
    )
    ctrl_high = GenericRepairController(
        REF, _task, _safe, {**BASE_CONFIG, "confidence_dist_scale": 10.0}, domain="high"
    )

    kl = ctrl_low._kl_from_reference(theta_far)
    c_low = ctrl_low._confidence(theta_far, kl)
    c_high = ctrl_high._confidence(theta_far, kl)
    assert c_low > c_high, f"Expected low={c_low:.4f} > high={c_high:.4f}"
