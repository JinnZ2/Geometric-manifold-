"""
Mathematical invariant tests for the basin repair framework.

These tests verify correctness properties that MUST hold,
not just "doesn't crash" but "the math is right."

Each test name documents the invariant it checks.
"""

import math

import torch
import torch.linalg as LA

from manifolds.data_manifold import DataManifold
from manifolds.parameter_manifold import ParameterManifold
from manifolds.policy_manifold import PolicyManifold
from repair.geometric_confidence import GeometricConfidence
from simulation.environment import Environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(drift=0.3):
    return Environment({"drift_strength": drift, "seed": 42})


def _make_param_layer(env, **overrides):
    defaults = {
        "trust_radius": 0.05,
        "asymmetry_lambda": 10.0,
        "curvature_weight": 2.0,
        "lr": 0.01,
    }
    defaults.update(overrides)
    return ParameterManifold(env.theta_ref, defaults)


# ---------------------------------------------------------------------------
# Parameter Manifold invariants
# ---------------------------------------------------------------------------


def test_repair_step_produces_finite_output():
    """Repair step must produce finite, non-NaN parameters and metrics."""
    env = _make_env(drift=0.5)
    pm = _make_param_layer(env)
    fn = env.get_model_fn()

    theta_new, metrics = pm.repair_step(
        env.theta_drifted,
        fn,
        env.safety_inputs,
        env.task_inputs,
        env.task_labels,
    )

    assert torch.isfinite(theta_new).all(), "Repair produced non-finite parameters"
    for key in ("task_loss", "safety_loss", "curvature", "confidence", "dist_to_ref"):
        assert key in metrics, f"Missing metric: {key}"
        assert math.isfinite(metrics[key]), f"Non-finite metric {key}={metrics[key]}"


def test_trust_region_respected():
    """The parameter update delta must never exceed trust_radius."""
    env = _make_env(drift=0.8)
    trust_radius = 0.03
    pm = _make_param_layer(env, trust_radius=trust_radius, lr=0.1)
    fn = env.get_model_fn()

    theta_new, _ = pm.repair_step(
        env.theta_drifted,
        fn,
        env.safety_inputs,
        env.task_inputs,
        env.task_labels,
    )
    delta_norm = LA.norm(theta_new - env.theta_drifted).item()

    assert delta_norm <= trust_radius + 1e-6, (
        f"Delta norm {delta_norm:.6f} exceeds trust radius {trust_radius}"
    )


def test_confidence_bounded_zero_one():
    """Parameter confidence must always be in [0, 1]."""
    for drift in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0]:
        env = _make_env(drift=drift)
        pm = _make_param_layer(env)
        fn = env.get_model_fn()
        _, metrics = pm.repair_step(
            env.theta_drifted,
            fn,
            env.safety_inputs,
            env.task_inputs,
            env.task_labels,
        )
        conf = metrics["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0,1] at drift={drift}"


def test_confidence_decreases_with_drift():
    """Higher drift (farther from ref) should yield lower confidence."""
    confs = []
    for drift in [0.1, 0.3, 0.5, 0.8]:
        env = _make_env(drift=drift)
        pm = _make_param_layer(env)
        fn = env.get_model_fn()
        _, metrics = pm.repair_step(
            env.theta_drifted,
            fn,
            env.safety_inputs,
            env.task_inputs,
            env.task_labels,
        )
        confs.append(metrics["confidence"])

    # Overall trend must hold: low drift -> higher confidence than high drift
    assert confs[0] > confs[-1], f"Confidence should decrease with drift: {confs}"


def test_zero_drift_gives_high_confidence():
    """When theta == theta_ref (no drift), confidence should be near 1."""
    env = _make_env(drift=0.0)
    pm = _make_param_layer(env)
    fn = env.get_model_fn()
    _, metrics = pm.repair_step(
        env.theta_ref.clone(),
        fn,
        env.safety_inputs,
        env.task_inputs,
        env.task_labels,
    )
    assert metrics["confidence"] > 0.8, (
        f"Zero-drift confidence should be high, got {metrics['confidence']:.4f}"
    )


def test_curvature_proxy_nonnegative():
    """Variance-based curvature proxy must always be >= 0."""
    env = _make_env()
    pm = _make_param_layer(env)
    fn = env.get_model_fn()
    _, metrics = pm.repair_step(
        env.theta_drifted,
        fn,
        env.safety_inputs,
        env.task_inputs,
        env.task_labels,
    )
    assert metrics["curvature"] >= 0.0, (
        f"Curvature proxy must be non-negative, got {metrics['curvature']}"
    )


def test_asymmetric_loss_penalizes_safety_more():
    """With lambda_asym=10, weighted safety contribution should exceed task loss."""
    env = _make_env(drift=0.5)
    pm = _make_param_layer(env, asymmetry_lambda=10.0)
    fn = env.get_model_fn()
    _, metrics = pm.repair_step(
        env.theta_drifted,
        fn,
        env.safety_inputs,
        env.task_inputs,
        env.task_labels,
    )
    weighted_safety = 10.0 * metrics["safety_loss"] * (1.0 + 2.0 * metrics["curvature"])
    assert weighted_safety > metrics["task_loss"], (
        f"Weighted safety {weighted_safety:.4f} should exceed task loss "
        f"{metrics['task_loss']:.4f} with lambda=10"
    )


# ---------------------------------------------------------------------------
# Policy Manifold invariants
# ---------------------------------------------------------------------------


def test_js_divergence_zero_for_identical():
    """JS(P, P) = 0, so confidence should be 1.0 for identical distributions."""
    pm = PolicyManifold({"confidence_threshold": 0.4})
    probs = torch.softmax(torch.randn(16, 32), dim=-1)
    conf = pm.trajectory_confidence(probs, probs)
    assert conf > 0.99, f"Identical distributions should give conf ~1.0, got {conf}"


def test_js_divergence_symmetric():
    """JS divergence is symmetric: JS(P,Q) == JS(Q,P)."""
    pm = PolicyManifold({"confidence_threshold": 0.4})
    p = torch.softmax(torch.randn(8, 16), dim=-1)
    q = torch.softmax(torch.randn(8, 16), dim=-1)
    conf_pq = pm.trajectory_confidence(p, q)
    conf_qp = pm.trajectory_confidence(q, p)
    assert abs(conf_pq - conf_qp) < 0.01, (
        f"JS divergence should be symmetric: {conf_pq:.4f} vs {conf_qp:.4f}"
    )


def test_reanchor_is_linear_blend():
    """Reanchor should be a convex combination: (1-s)*P + s*Q."""
    s = 0.3
    pm = PolicyManifold({"reanchor_strength": s})
    p = torch.softmax(torch.randn(8, 16), dim=-1)
    q = torch.softmax(torch.randn(8, 16), dim=-1)
    blended = pm.reanchor(p, q)
    expected = (1 - s) * p + s * q
    assert torch.allclose(blended, expected, atol=1e-6), (
        "Reanchor should produce exact convex combination"
    )


def test_reanchor_strength_zero_returns_original():
    """With reanchor_strength=0, output should equal input."""
    pm = PolicyManifold({"reanchor_strength": 0.0})
    p = torch.softmax(torch.randn(8, 16), dim=-1)
    q = torch.softmax(torch.randn(8, 16), dim=-1)
    blended = pm.reanchor(p, q)
    assert torch.allclose(blended, p, atol=1e-6)


def test_reanchor_strength_one_returns_reference():
    """With reanchor_strength=1, output should equal reference."""
    pm = PolicyManifold({"reanchor_strength": 1.0})
    p = torch.softmax(torch.randn(8, 16), dim=-1)
    q = torch.softmax(torch.randn(8, 16), dim=-1)
    blended = pm.reanchor(p, q)
    assert torch.allclose(blended, q, atol=1e-6)


# ---------------------------------------------------------------------------
# Data Manifold invariants
# ---------------------------------------------------------------------------


def test_minority_samples_never_fully_dropped():
    """Minority samples (label=1) should always get weight > 0."""
    dm = DataManifold(
        {
            "confidence_threshold_majority": 0.7,
            "confidence_threshold_minority": 0.3,
            "alpha": 0.5,
            "beta": 1.0,
            "beta_prime": 0.1,
        }
    )
    env = _make_env()
    weights = dm.asymmetric_weights(env.features, env.labels)
    minority_mask = env.labels == 1
    assert (weights[minority_mask] > 0).all(), "Minority samples should never be dropped entirely"


def test_majority_low_confidence_dropped():
    """Low-confidence majority samples should be dropped (weight=0)."""
    dm = DataManifold(
        {
            "confidence_threshold_majority": 0.7,
            "confidence_threshold_minority": 0.3,
            "alpha": 0.5,
            "beta": 1.0,
        }
    )
    env = _make_env()
    weights = dm.asymmetric_weights(env.features, env.labels)
    majority_mask = env.labels == 0
    assert (weights[majority_mask] == 0).any(), (
        "Some low-confidence majority samples should be dropped"
    )


def test_rectify_filters_zero_weight_samples():
    """Rectify output should only contain samples with weight > 0."""
    dm = DataManifold(
        {
            "confidence_threshold_majority": 0.7,
            "confidence_threshold_minority": 0.3,
            "alpha": 0.5,
            "beta": 1.0,
        }
    )
    env = _make_env()
    feat_clean, lbl_clean, wts = dm.rectify(env.features, env.labels)
    assert len(feat_clean) <= len(env.features)
    assert (wts > 0).all(), "Rectified output should have no zero-weight samples"


def test_rectify_preserves_minority_ratio():
    """After cleaning, minority fraction should increase (since majority is pruned harder)."""
    dm = DataManifold(
        {
            "confidence_threshold_majority": 0.7,
            "confidence_threshold_minority": 0.3,
            "alpha": 0.5,
            "beta": 1.0,
            "beta_prime": 0.1,
        }
    )
    env = _make_env()
    minority_frac_before = (env.labels == 1).float().mean().item()
    _, lbl_clean, _ = dm.rectify(env.features, env.labels)
    minority_frac_after = (lbl_clean == 1).float().mean().item()
    assert minority_frac_after >= minority_frac_before, (
        f"Minority fraction should increase after asymmetric cleaning: "
        f"{minority_frac_before:.3f} -> {minority_frac_after:.3f}"
    )


# ---------------------------------------------------------------------------
# Confidence aggregation invariants
# ---------------------------------------------------------------------------


def test_combined_confidence_bounded():
    """Combined confidence must stay in [0, 1] for any valid inputs."""
    gc = GeometricConfidence()
    for _ in range(100):
        d, p, pi = torch.rand(3).tolist()
        c = gc.combined(d, p, pi)
        assert 0.0 <= c <= 1.0, f"Combined confidence {c} out of bounds"


def test_combined_weights_sum_to_one():
    """Default weights (0.2, 0.5, 0.3) produce correct extremes."""
    gc = GeometricConfidence()
    assert abs(gc.combined(1.0, 1.0, 1.0) - 1.0) < 1e-6
    assert abs(gc.combined(0.0, 0.0, 0.0)) < 1e-6


def test_parameter_confidence_dominates():
    """Parameter manifold (50% weight) should outweigh data (20%) and policy (30%)."""
    gc = GeometricConfidence()
    c_param = gc.combined(0.0, 1.0, 0.0)
    c_data = gc.combined(1.0, 0.0, 0.0)
    c_policy = gc.combined(0.0, 0.0, 1.0)

    assert c_param > c_data, "Parameter should outweigh data"
    assert c_param > c_policy, "Parameter should outweigh policy"


# ---------------------------------------------------------------------------
# Multi-step repair convergence
# ---------------------------------------------------------------------------


def test_repeated_repair_stays_bounded():
    """Multiple repair steps must keep parameters bounded (no divergence)."""
    env = _make_env(drift=0.5)
    pm = _make_param_layer(env)
    fn = env.get_model_fn()

    theta = env.theta_drifted.clone()
    dist_initial = LA.norm(theta - env.theta_ref).item()

    for _ in range(10):
        theta, metrics = pm.repair_step(
            theta,
            fn,
            env.safety_inputs,
            env.task_inputs,
            env.task_labels,
        )
        assert torch.isfinite(theta).all(), "Parameters diverged to inf/nan"

    dist_final = LA.norm(theta - env.theta_ref).item()
    # Trust region bounds cumulative drift: at most 10 * trust_radius from start
    max_drift = dist_initial + 10 * 0.05
    assert dist_final < max_drift, (
        f"Distance grew beyond trust region bound: {dist_final:.4f} > {max_drift:.4f}"
    )


def test_asymmetry_lambda_changes_trajectory():
    """Different asymmetry_lambda values should produce different repair trajectories."""
    env = _make_env(drift=0.5)
    fn = env.get_model_fn()

    thetas = {}
    for lam in [1.0, 10.0]:
        pm = _make_param_layer(env, asymmetry_lambda=lam)
        theta = env.theta_drifted.clone()
        for _ in range(5):
            theta, _ = pm.repair_step(
                theta,
                fn,
                env.safety_inputs,
                env.task_inputs,
                env.task_labels,
            )
        thetas[lam] = theta

    # Different lambda values must produce different endpoints
    diff = LA.norm(thetas[1.0] - thetas[10.0]).item()
    assert diff > 1e-4, (
        f"Different asymmetry_lambda should produce different trajectories, but diff={diff:.6f}"
    )
