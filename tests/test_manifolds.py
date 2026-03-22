"""Smoke tests for manifold layers."""

import torch

from manifolds.data_manifold import DataManifold
from manifolds.parameter_manifold import ParameterManifold
from manifolds.policy_manifold import PolicyManifold
from simulation.environment import Environment


def _make_env():
    return Environment({'drift_strength': 0.3, 'seed': 42})


def test_data_manifold_rectify():
    dm = DataManifold({
        'confidence_threshold_majority': 0.7,
        'confidence_threshold_minority': 0.3,
        'alpha': 0.5,
        'beta': 1.0,
    })
    env = _make_env()
    feat_clean, lbl_clean, weights = dm.rectify(env.features, env.labels)

    assert len(feat_clean) <= len(env.features)
    assert len(feat_clean) == len(lbl_clean) == len(weights)
    assert (weights > 0).all()


def test_parameter_manifold_repair_step():
    env = _make_env()
    pm = ParameterManifold(env.theta_ref, {
        'trust_radius': 0.05,
        'asymmetry_lambda': 10.0,
        'curvature_weight': 2.0,
        'lr': 0.01,
    })
    fn = env.get_model_fn()
    theta_new, metrics = pm.repair_step(
        env.theta_drifted, fn,
        env.safety_inputs, env.task_inputs, env.task_labels,
    )

    assert theta_new.shape == env.theta_drifted.shape
    assert 'task_loss' in metrics
    assert 'safety_loss' in metrics
    assert 'confidence' in metrics
    assert 'dist_to_ref' in metrics
    assert isinstance(metrics['confidence'], float)


def test_policy_manifold_confidence():
    pm = PolicyManifold({'confidence_threshold': 0.4, 'reanchor_strength': 0.1})
    probs = torch.softmax(torch.randn(8, 16), dim=-1)
    ref_probs = torch.softmax(torch.randn(8, 16), dim=-1)

    conf = pm.trajectory_confidence(probs, ref_probs)
    assert 0.0 <= conf <= 1.0


def test_policy_manifold_identical_gives_high_confidence():
    pm = PolicyManifold({'confidence_threshold': 0.4})
    probs = torch.softmax(torch.randn(8, 16), dim=-1)
    conf = pm.trajectory_confidence(probs, probs)
    assert conf > 0.9


def test_policy_manifold_reanchor():
    pm = PolicyManifold({'reanchor_strength': 0.5})
    probs = torch.softmax(torch.randn(8, 16), dim=-1)
    ref_probs = torch.softmax(torch.randn(8, 16), dim=-1)
    blended = pm.reanchor(probs, ref_probs)
    expected = 0.5 * probs + 0.5 * ref_probs
    assert torch.allclose(blended, expected)
