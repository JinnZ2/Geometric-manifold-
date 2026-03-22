# Basin Repair Framework

A geometric manifold approach to maintaining neural network parameter safety. The framework treats model safety as a geometric problem — using three interconnected manifold layers to detect when a model drifts from a safe reference state and apply repair mechanisms.

## Core Idea

Neural network parameters live in a high-dimensional space. Safe model configurations occupy "basins" in this landscape. When fine-tuning or adversarial drift pushes parameters out of a safe basin, this framework detects the drift and pulls parameters back using curvature-aware, asymmetric repair.

The system operates across three geometric layers:

1. **Data Manifold** — GMR-style feature space cleaning with asymmetric majority/minority handling
2. **Parameter Manifold** — Curvature-aware basin repair in weight space with trust region constraints
3. **Policy Manifold** — Trajectory-level alignment using JS divergence monitoring

## Quick Start

```bash
pip install -r requirements.txt

# Run the default simulation (drift=0.3, 100 steps)
python main.py --config configs/default.yaml

# Run with adversarial settings (drift=0.8, 200 steps)
python main.py --config configs/adversarial.yaml

# Energy sweep experiment
python mode_flag.py --mode energy_sweep

# Run tests
python -m pytest tests/
```

## Project Structure

```
simulation/          Core simulation (ToyLLM environment, repair controller)
manifolds/           Three-layer geometric framework (data, parameter, policy)
repair/              Confidence aggregation and monitoring utilities
addon_thermodynamic_control/   Fisher metric energy accounting and stability analysis
configs/             YAML configuration files
experiments/         Experiment scripts (ablations, cost analysis, landscape studies)
tests/               Smoke tests for all modules
docs/                Theoretical notes
```

## Configuration

All hyperparameters are controlled via YAML configs in `configs/`. Key knobs:

| Parameter | Location | Effect |
|---|---|---|
| `drift_strength` | `simulation` | How far the model drifts from safe reference |
| `trust_radius` | `manifolds.parameter` | Max step size for parameter repair |
| `asymmetry_lambda` | `manifolds.parameter` | Safety vs task penalty ratio |
| `confidence_threshold` | `manifolds.policy` | When to trigger policy re-anchoring |

## License

CC0 1.0 Universal (public domain)
