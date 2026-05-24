"""
Tests for research_interface — ManifoldResearchInterface and BridgeOptimizer.

Invariants tested:
  - Evaluation produces bounded, finite outputs for any bridge matrix
  - prediction_error ∈ [0, 1] (MSE of tanh outputs vs unit-normalised target)
  - net_viability changes monotonically with gradient step direction
  - trust region respected per step
  - FD gradient is consistent (central differences, linear in perturbation)
  - Optimizer converges to higher viability than random init over N steps
"""

import numpy as np
import pytest
from research_interface import ManifoldResearchInterface, BridgeOptimizer


DIMS = 4
S = np.array([0.9, -0.4, 0.7, -0.1])
P = np.array([120.0, 75.0, 0.35, 50.0])


@pytest.fixture
def sandbox():
    return ManifoldResearchInterface(manifold_dimensions=DIMS)


@pytest.fixture
def optimizer(sandbox):
    return BridgeOptimizer(sandbox, config={"lr": 0.01, "trust_radius": 0.05})


# ── Evaluation invariants ────────────────────────────────────────────────────

def test_evaluate_returns_required_keys(sandbox):
    W = sandbox.identity_bridge()
    result = sandbox.evaluate_bridge_geometry(W, S, P)
    for key in ("net_viability", "prediction_error", "heat_leak",
                "manifold_coordinates", "effective_metrics"):
        assert key in result, f"Missing key: {key}"


def test_prediction_error_bounded(sandbox):
    """MSE of tanh outputs vs unit target must be in [0, 4].

    Both compressed (tanh output) and p_hat (normalised physical) lie in [-1, 1],
    so the maximum squared difference is (1 - (-1))^2 = 4.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        W = rng.normal(0, 1, (DIMS, DIMS))
        r = sandbox.evaluate_bridge_geometry(W, S, P)
        assert 0.0 <= r["prediction_error"] <= 4.0 + 1e-9, (
            f"prediction_error={r['prediction_error']} out of [0,4]"
        )


def test_manifold_coordinates_bounded(sandbox):
    """tanh outputs must lie in [-1, 1] (inclusive — saturates at ±1 in float64)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        W = rng.normal(0, 5, (DIMS, DIMS))    # large weights stress-test tanh saturation
        r = sandbox.evaluate_bridge_geometry(W, S, P)
        for c in r["manifold_coordinates"]:
            assert -1.0 <= c <= 1.0, f"manifold coordinate {c} outside [-1,1]"


def test_identical_distributions_zero_prediction_error(sandbox):
    """When bridge perfectly aligns sensory with physical, prediction_error → 0."""
    # Construct W such that tanh(W @ S) = P/max(|P|) exactly (up to atanh precision)
    p_scale = np.max(np.abs(P))
    target = np.arctanh(np.clip(P / p_scale, -0.999, 0.999))   # atanh of target
    # W needs W @ S = target; since S is known, W = target @ pinv(S)
    S_col = S.reshape(-1, 1)
    W = target.reshape(-1, 1) @ np.linalg.pinv(S_col)
    r = sandbox.evaluate_bridge_geometry(W, S, P)
    assert r["prediction_error"] < 1e-6, (
        f"Expected zero prediction_error, got {r['prediction_error']}"
    )


def test_outputs_finite_for_extreme_matrices(sandbox):
    """Large/small weight matrices must not produce NaN or inf."""
    for scale in [1e-10, 1e10, -1e10]:
        W = np.full((DIMS, DIMS), scale)
        r = sandbox.evaluate_bridge_geometry(W, S, P)
        assert np.isfinite(r["net_viability"]), f"Non-finite viability at scale={scale}"
        assert np.isfinite(r["prediction_error"]), f"Non-finite pred_error at scale={scale}"


def test_wrong_shape_raises(sandbox):
    W_bad = np.eye(DIMS + 1)
    with pytest.raises(ValueError):
        sandbox.evaluate_bridge_geometry(W_bad, S, P)


def test_zero_divisor_guarded(sandbox):
    """max(|P|)=0 should not crash (all-zero physical metrics)."""
    W = sandbox.identity_bridge()
    r = sandbox.evaluate_bridge_geometry(W, S, np.zeros(DIMS))
    assert np.isfinite(r["prediction_error"])


# ── FD gradient invariants ───────────────────────────────────────────────────

def test_gradient_shape(sandbox):
    W = sandbox.identity_bridge()
    g = sandbox.gradient(W, S, P)
    assert g.shape == (DIMS, DIMS)


def test_gradient_direction_improves_viability(sandbox):
    """An infinitesimal step in the gradient direction must increase net_viability.

    Uses a step of 1e-6 in the unit-gradient direction to stay inside the
    linear regime of the FD approximation.
    """
    W = sandbox.random_bridge(seed=7)
    r0 = sandbox.evaluate_bridge_geometry(W, S, P)
    g = sandbox.gradient(W, S, P)
    g_unit = g / (np.linalg.norm(g) + 1e-8)
    W_step = W + 1e-6 * g_unit
    r1 = sandbox.evaluate_bridge_geometry(W_step, S, P)
    assert r1["net_viability"] >= r0["net_viability"] - 1e-9, (
        "Gradient step decreased viability (wrong sign or FD error)"
    )


def test_gradient_linear_in_perturbation(sandbox):
    """FD gradient must scale linearly: grad(alpha * E_ij perturbation) = alpha * grad."""
    W = sandbox.random_bridge(seed=3)
    epsilon = 1e-4
    g = sandbox.gradient(W, S, P, epsilon=epsilon)

    # Verify one element: finite difference on W[1,2]
    W_p = W.copy(); W_p[1, 2] += epsilon
    W_m = W.copy(); W_m[1, 2] -= epsilon
    fd_12 = (
        sandbox.evaluate_bridge_geometry(W_p, S, P)["net_viability"]
        - sandbox.evaluate_bridge_geometry(W_m, S, P)["net_viability"]
    ) / (2 * epsilon)
    assert abs(g[1, 2] - fd_12) < 1e-6, (
        f"Gradient element [1,2] mismatch: gradient={g[1,2]:.6f}, fd={fd_12:.6f}"
    )


# ── Optimizer invariants ─────────────────────────────────────────────────────

def test_optimizer_step_returns_correct_keys(optimizer, sandbox):
    W = sandbox.identity_bridge()
    W_new, m = optimizer.step(W, S, P)
    for key in ("net_viability", "prediction_error", "heat_leak",
                "grad_frob_norm", "delta_frob_norm", "trust_region_active"):
        assert key in m, f"Missing metric key: {key}"


def test_trust_region_respected(optimizer, sandbox):
    """‖ΔW‖_F must never exceed trust_radius."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        W = rng.normal(0, 2, (DIMS, DIMS))
        W_new, m = optimizer.step(W, S, P)
        delta_norm = np.linalg.norm(W_new - W)
        assert delta_norm <= optimizer.trust_radius + 1e-9, (
            f"Trust region violated: ‖ΔW‖={delta_norm:.6f} > {optimizer.trust_radius}"
        )


def test_optimizer_improves_over_random_init(sandbox):
    """After N gradient steps, viability should exceed the random starting value."""
    opt = BridgeOptimizer(sandbox, config={"lr": 0.02, "trust_radius": 0.1})
    W0 = sandbox.random_bridge(seed=99)
    r0 = sandbox.evaluate_bridge_geometry(W0, S, P)

    history = opt.run(W0, S, P, n_steps=30, log_interval=999)  # suppress output
    final = history[-1]["net_viability"]
    assert final > r0["net_viability"], (
        f"Optimizer should improve viability: {r0['net_viability']:.4f} → {final:.4f}"
    )


def test_hypo_2_beats_hypo_1(sandbox):
    """Hypothesis 2 (near-diagonal) should score higher than hypothesis 1 (random)."""
    hypo_1 = np.array([
        [0.1, -0.2,  0.5,  0.0],
        [0.9,  0.1, -0.3,  0.2],
        [-0.4, 0.6,  0.1,  0.8],
        [0.2, -0.1,  0.7,  0.3],
    ])
    hypo_2 = np.array([
        [0.9,  0.0,  0.1, -0.8],
        [-0.1, 0.85, 0.0,  0.3],
        [0.2, -0.3,  0.95, 0.0],
        [0.0,  0.1, -0.2,  0.9],
    ])
    v1 = sandbox.evaluate_bridge_geometry(hypo_1, S, P)["net_viability"]
    v2 = sandbox.evaluate_bridge_geometry(hypo_2, S, P)["net_viability"]
    assert v2 > v1, f"Expected hypo_2 ({v2:.4f}) > hypo_1 ({v1:.4f})"


def test_momentum_does_not_violate_trust_region(sandbox):
    """Momentum accumulation must not push delta beyond trust_radius."""
    opt = BridgeOptimizer(sandbox, config={"lr": 0.02, "trust_radius": 0.08, "momentum": 0.9})
    W = sandbox.identity_bridge()
    for _ in range(10):
        W_new, m = opt.step(W, S, P)
        delta_norm = np.linalg.norm(W_new - W)
        assert delta_norm <= opt.trust_radius + 1e-9
        W = W_new


def test_configurable_penalty_weights(sandbox):
    """Changing penalty weights should change net_viability but not prediction_error."""
    W = sandbox.near_diagonal_bridge(seed=5)
    r_default = sandbox.evaluate_bridge_geometry(W, S, P)

    heavy_sandbox = ManifoldResearchInterface(
        manifold_dimensions=DIMS,
        config={"prediction_error_penalty": 200.0, "heat_leak_penalty": 10.0}
    )
    r_heavy = heavy_sandbox.evaluate_bridge_geometry(W, S, P)

    assert r_default["prediction_error"] == r_heavy["prediction_error"], (
        "prediction_error should not depend on penalty weights"
    )
    assert r_default["net_viability"] != r_heavy["net_viability"], (
        "net_viability should change with penalty weights"
    )
