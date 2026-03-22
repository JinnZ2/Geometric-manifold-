"""Smoke tests for the simulation environment."""

import torch

from simulation.environment import Environment, ToyLLM, model_fn


def test_toy_llm_forward():
    model = ToyLLM(input_dim=32, hidden_dim=64, output_dim=16)
    x = torch.randn(8, 32)
    out = model(x)
    assert out.shape == (8, 16)


def test_model_fn_matches_module():
    model = ToyLLM(input_dim=32, hidden_dim=64, output_dim=16)
    x = torch.randn(4, 32)
    module_out = model(x)

    params = [p.data.flatten() for p in model.parameters()]
    theta = torch.cat(params)
    fn_out = model_fn(x, theta, input_dim=32, hidden_dim=64, output_dim=16)

    assert torch.allclose(module_out, fn_out, atol=1e-5)


def test_environment_init():
    config = {'drift_strength': 0.3, 'seed': 42}
    env = Environment(config)

    assert env.theta_ref.shape == env.theta_drifted.shape
    assert env.safety_inputs.shape == (32, 32)
    assert env.task_inputs.shape == (32, 32)
    assert env.task_labels.shape == (32,)
    assert env.features.shape == (200, 32)
    assert env.labels.shape == (200,)


def test_environment_get_model_fn():
    config = {'drift_strength': 0.1, 'seed': 0}
    env = Environment(config)
    fn = env.get_model_fn()
    out = fn(env.task_inputs, env.theta_ref)
    assert out.shape == (32, 16)


def test_drift_changes_params():
    config = {'drift_strength': 0.5, 'seed': 42}
    env = Environment(config)
    assert not torch.allclose(env.theta_ref, env.theta_drifted)
