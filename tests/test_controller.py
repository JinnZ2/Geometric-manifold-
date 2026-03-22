"""Integration smoke test: run the full pipeline for a few steps."""

from simulation.controller import Controller
from simulation.environment import Environment


def test_controller_runs(tmp_path):
    config = {
        'simulation': {'drift_strength': 0.3, 'steps': 5, 'seed': 42},
        'manifolds': {
            'data': {
                'enabled': True,
                'confidence_threshold_majority': 0.7,
                'confidence_threshold_minority': 0.3,
                'alpha': 0.5,
                'beta': 1.0,
            },
            'parameter': {
                'enabled': True,
                'trust_radius': 0.05,
                'asymmetry_lambda': 10.0,
                'curvature_weight': 2.0,
                'lr': 0.01,
            },
            'policy': {
                'enabled': True,
                'confidence_threshold': 0.4,
                'reanchor_strength': 0.1,
            },
        },
        'monitoring': {
            'log_interval': 1,
            'output_dir': str(tmp_path),
        },
    }
    env = Environment(config['simulation'])
    ctrl = Controller(env, config)
    monitor = ctrl.run()

    summary = monitor.summary()
    assert summary['total_steps'] == 5
    assert (tmp_path / 'metrics.csv').exists()


def test_controller_layers_disabled(tmp_path):
    config = {
        'simulation': {'drift_strength': 0.1, 'steps': 3, 'seed': 0},
        'manifolds': {
            'data': {'enabled': False},
            'parameter': {'enabled': False},
            'policy': {'enabled': False},
        },
        'monitoring': {
            'log_interval': 1,
            'output_dir': str(tmp_path),
        },
    }
    env = Environment(config['simulation'])
    ctrl = Controller(env, config)
    monitor = ctrl.run()
    assert monitor.summary()['total_steps'] == 3
