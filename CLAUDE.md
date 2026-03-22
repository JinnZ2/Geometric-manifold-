# CLAUDE.md

## Project Overview

**Basin Repair Framework** — a geometric manifold approach to maintaining neural network parameter safety. Treats model safety as a geometric problem: safe model configurations occupy "basins" in parameter space, and the framework detects/repairs drift using three interconnected manifold layers.

Research-stage Python project (~2,100 lines). No production deployment.

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run simulation
python main.py --config configs/default.yaml

# Run tests (15 smoke tests, ~2s)
python -m pytest tests/ -v

# Lint
ruff check .

# Format check
ruff format --check .
```

## Repository Structure

```
main.py                          # Primary entry point (--config flag)
mode_flag.py                     # Mode selector (--mode simulate|energy_sweep)
pyproject.toml                   # Ruff + pytest config
requirements.txt                 # Python deps: torch, numpy, scipy, pyyaml, matplotlib, pandas, tqdm

simulation/
  environment.py                 # ToyLLM (2-layer MLP, ~8k params), synthetic data generation
  controller.py                  # Orchestrates all 3 manifold layers per step

manifolds/
  data_manifold.py               # Layer 1: GMR-style k-NN feature space cleaning
  parameter_manifold.py          # Layer 2: Curvature-aware basin repair in weight space
  policy_manifold.py             # Layer 3: JS divergence trajectory alignment

repair/
  geometric_confidence.py        # Unified confidence: 20% data + 50% param + 30% policy
  monitors.py                    # Per-step metric logging, CSV export, cost spike detection

addon_thermodynamic_control/
  energy.py                      # Fisher Information metric, thermodynamic energy accounting
  stability.py                   # Phase detection & stability analysis (552 lines, largest file)
  geometry_shaping.py            # Landscape shaping objectives
  addendum_formal_objectives.py  # 4-term Lagrangian formalization
  experiment_*.py                # Experiment scripts for each addon module

configs/
  default.yaml                   # Standard: drift=0.3, steps=100, seed=42
  adversarial.yaml               # Stress: drift=0.8, steps=200, tighter trust region

experiments/                     # Standalone experiment scripts
tests/                           # Pytest smoke tests for all modules
docs/theoretical_notes/          # Mathematical foundations
```

## Architecture

### Three-Layer Manifold Pipeline

The controller runs each step in sequence:

1. **Data Manifold** (`DataManifold.rectify()`) — runs once before the loop. Asymmetric k-NN cleaning: aggressively drops low-confidence majority samples, conservatively keeps minority samples.

2. **Parameter Manifold** (`ParameterManifold.repair_step()`) — runs per step. Computes `task_loss - λ * curvature_weighted_safety_loss`, takes gradient step within trust region. Returns `(new_theta, metrics_dict)`.

3. **Policy Manifold** (`PolicyManifold.trajectory_confidence()`) — runs per step. Measures JS divergence between current and reference action distributions. Triggers re-anchoring when confidence drops below threshold.

### Key Interfaces

Every manifold layer follows this pattern:
- Takes a config dict in `__init__`
- Has a primary method returning results + metrics
- Returns confidence as a float in [0, 1]

```python
# Parameter manifold example
theta_new, metrics = param_layer.repair_step(theta, model_fn, safety_inputs, task_inputs, task_labels)
# metrics = {'task_loss': ..., 'safety_loss': ..., 'curvature': ..., 'confidence': ..., 'dist_to_ref': ...}
```

The functional model interface is: `model_fn(inputs: Tensor, theta_flat: Tensor) -> Tensor`

### Confidence Aggregation

`GeometricConfidence.combined()` weights the three layers:
- Data: 20% weight
- Parameter: 50% weight (dominant signal)
- Policy: 30% weight

## Code Conventions

### Style & Formatting
- **Linter**: ruff (configured in `pyproject.toml`). Rules: E, F, W, I (pycodestyle, pyflakes, isort).
- **Line length**: 100 chars (soft, E501 ignored).
- **Import order**: stdlib, third-party, then first-party (`simulation`, `manifolds`, `repair`, `addon_thermodynamic_control`). Enforced by ruff isort.
- **Python version**: 3.11+

### Patterns to Follow
- **Configuration-driven**: All hyperparameters via YAML configs. Never hardcode values that should be tunable.
- **Metric dictionaries**: Methods return `{'metric_name': float_value}` for standardized logging.
- **Functional model interface**: Use `model_fn(inputs, theta)` — never pass nn.Module objects between layers.
- **Tensor-based**: PyTorch tensors throughout. Use `torch.no_grad()` for inference-only paths.
- **Docstrings**: Include on public methods. Explain the geometric/mathematical intuition, not just what the code does.

### Testing
- Tests live in `tests/` and use pytest.
- Run: `python -m pytest tests/ -v`
- Test files mirror source structure: `test_environment.py`, `test_manifolds.py`, `test_repair.py`, `test_controller.py`.
- Use `tmp_path` fixture for any file output (never write to real `results/` in tests).
- New code should have corresponding smoke tests.

## Key Design Decisions

- **Asymmetric penalties**: Safety violations penalized `λ` times more than task loss (`asymmetry_lambda`, default 10.0). This is intentional — safety > performance.
- **Trust regions**: Parameter updates capped at `trust_radius` (default 0.05) to prevent catastrophic jumps.
- **Curvature proxy**: Uses variance of softmax distribution as a cheap curvature estimate (not full Hessian).
- **Thermodynamic extensions** (`addon_thermodynamic_control/`): Adds Fisher metric energy accounting and phase transition detection. This is the most mathematically dense module.

## Common Tasks

### Adding a new manifold layer
1. Create class in `manifolds/` with `__init__(self, config: dict)` and a primary method returning metrics.
2. Wire it into `simulation/controller.py` in the step loop.
3. Add its confidence to `repair/geometric_confidence.py` (update weights tuple).
4. Add config section in both `configs/default.yaml` and `configs/adversarial.yaml`.
5. Add smoke tests in `tests/`.

### Tuning repair behavior
Edit YAML configs in `configs/`. The most impactful knobs:
- `drift_strength`: How far the model starts from safe reference.
- `trust_radius`: Max parameter step size (smaller = more conservative repair).
- `asymmetry_lambda`: Safety penalty multiplier (higher = stronger safety bias).
- `confidence_threshold`: Policy re-anchoring trigger point.

### Adding an experiment
Create a script in `experiments/` that:
1. Imports `Environment` from `simulation.environment`
2. Imports manifold layers or `Controller` as needed
3. Runs the pipeline with specific configs
4. Saves results to `results/` subdirectory

### Modifying the ToyLLM
The model is defined in `simulation/environment.py`. Both `ToyLLM` (nn.Module) and `model_fn` (functional) must stay in sync — the functional version is used throughout the framework. If you change the architecture, update both and verify with `test_model_fn_matches_module`.

## Gotchas

- `model_fn` uses a flattened parameter vector (`theta_flat`), not named parameters. The split indices are computed from layer dimensions.
- `DataManifold.rectify()` runs once (pre-loop), while the other two layers run every step.
- The monitor's `detect_cost_spike()` needs `2 * window` steps of history before it activates.
- `addon_thermodynamic_control/stability.py` is 552 lines — the largest and most complex file. Read the companion `README_stability.md` before modifying.
- Results are written to `results/` which is not gitignored. Don't commit generated CSV files.

## License

CC0 1.0 Universal (public domain).
