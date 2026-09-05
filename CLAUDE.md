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

# Run tests (183 tests: smoke + invariants + interface/engine suites, ~4s)
python -m pytest tests/ -v

# Run the hypothesis engine offline (stdlib only, no network)
python scripts/hypothesis_engine.py --config configs/topics.json --dry-run

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
  topics.json                    # Hypothesis-engine search topics

scripts/
  hypothesis_engine.py           # Autonomous literature pipeline (stdlib only, no torch)
  sample_findings.json           # Synthetic fixture data for --dry-run
  fieldlink_export.py            # Fieldlink export helper

.github/workflows/
  hypothesis-engine.yml          # Scheduled engine run; commits digest, opens issue

experiments/                     # Standalone experiment scripts
tests/                           # Pytest smoke tests for all modules
docs/theoretical_notes/          # Mathematical foundations
docs/hypothesis_engine.md        # Hypothesis-engine design doc
docs/research/                   # Literature notes + forward plans (see its README for provenance)
falsifier-survey/                # Delivered Run 2 falsifier survey, this repo's share; filed, instructions pending
```

### Hypothesis engine (`scripts/hypothesis_engine.py`)

A deliberately isolated subsystem: stdlib-only, deterministic, no LLM and no repo
imports. It queries free scholarly APIs, stakes each finding as a falsifiable claim,
tests claims by cross-source corroboration, escape-hatches repeat failures into an
unknown journal, and consolidates survivors into `hypotheses/*.md`.

It does **not** write to `manifolds/`, `simulation/`, or `repair/` — output is
literature grounding for humans to act on, never an automatic path into the safety
machinery. Keep it dependency-free so it runs on a bare GitHub runner; if you need
torch there, it belongs in a different script.

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

## Mathematical Invariants

These properties are tested in `tests/test_invariants.py` (21 tests). They encode the mathematical contracts each layer must satisfy. **Do not break these.**

### Parameter Manifold
- **Trust region**: `||delta|| <= trust_radius` after every step. No exceptions.
- **Confidence in [0, 1]**: `geometric_confidence()` returns a float in `[0, 1]` for any input.
- **Confidence monotonic with distance**: Higher drift from reference produces lower confidence.
- **Curvature non-negative**: The variance-based curvature proxy is always `>= 0`.
- **Finite outputs**: Every repair step must produce finite (non-NaN, non-inf) parameters and metrics.
- **Bounded drift**: Over N steps, distance from reference grows by at most `N * trust_radius`.

### Policy Manifold
- **JS(P, P) = 0**: Identical distributions must give confidence = 1.0.
- **JS symmetry**: `JS(P, Q) == JS(Q, P)` (within numerical tolerance).
- **Reanchor is convex combination**: `reanchor(P, Q) = (1-s)*P + s*Q` exactly.
- **Boundary behavior**: `reanchor_strength=0` returns P, `reanchor_strength=1` returns Q.

### Data Manifold
- **Minority never fully dropped**: Every minority sample (label=1) gets weight > 0 (either `beta` or `beta_prime`).
- **Majority pruning**: Some low-confidence majority samples are dropped (weight=0).
- **Asymmetric effect**: After rectification, minority class fraction increases or stays the same.

### Confidence Aggregation
- **Bounded**: `combined()` output is always in `[0, 1]`.
- **Weights sum to 1**: Default `(0.2, 0.5, 0.3)` must sum to 1.0.
- **Parameter dominates**: Parameter confidence (50%) outweighs data (20%) and policy (30%) individually.

## Do Not

These constraints exist for mathematical or safety reasons. Violating them silently breaks the framework.

- **Do not remove the trust region clamp** in `parameter_manifold.py`. It is the only hard guarantee against catastrophic parameter jumps. Without it, a single adversarial gradient can move theta arbitrarily far.
- **Do not change the confidence weight ordering** (param > policy > data) without updating both `geometric_confidence.py` and all downstream thresholds. The controller's abort logic depends on parameter confidence being dominant.
- **Do not make `beta_prime = 0`** in the data manifold config. This drops low-confidence minority samples entirely, violating the asymmetric cleaning contract (minority samples are never fully discarded).
- **Do not swap KL divergence for Euclidean distance** in `BasinDivergenceMonitor` or `parameter_manifold.py`. KL gives distributional basin boundaries; Euclidean is arbitrary in parameter space and doesn't correspond to behavioral difference.
- **Do not remove `torch.no_grad()`** from inference-only code paths (reference model evaluations, basin checks). Tracking gradients through the reference would corrupt the repair direction.
- **Do not compute full Hessians**. The framework deliberately uses cheap curvature proxies (softmax variance, diagonal Fisher). Full Hessians are O(n^2) in parameter count and will OOM on anything larger than ToyLLM.
- **Do not change the sign in `task_loss - λ * weighted_safety`** in `parameter_manifold.py`. This is an adversarial/saddle-point formulation, not a typo. The minus sign creates tension between task performance and safety alignment; the trust region resolves it.

## Config Rationale

Why these specific default values exist.

### `default.yaml`
| Parameter | Value | Why |
|-----------|-------|-----|
| `drift_strength` | 0.3 | Mild drift — enough to test repair without overwhelming the trust region in one step. |
| `trust_radius` | 0.05 | Bounds single-step movement to ~0.1% of typical `theta_ref` norm (~57). Conservative. |
| `asymmetry_lambda` | 10.0 | Safety loss dominates by 10x. Empirically, below 5x the repair trajectory wanders. |
| `curvature_weight` | 2.0 | Amplifies safety penalty in curved regions. At 1.0, curvature barely registers. |
| `lr` | 0.01 | Combined with trust_radius=0.05, most steps are trust-region-limited, not lr-limited. |
| `confidence_threshold` | 0.4 | Policy re-anchoring fires when JS divergence exceeds 0.6. Below 0.3: too aggressive. |
| `confidence_threshold_majority` | 0.7 | Drops ~30% of majority samples. At 0.5: too permissive (noise survives). |
| `confidence_threshold_minority` | 0.3 | Very low bar — keeps almost all minority samples even in noisy regions. |
| `reanchor_strength` | 0.1 | Gentle blend toward reference. At 0.5+: policy snaps back too hard, erasing adaptation. |

### `adversarial.yaml` differences
| Parameter | Default | Adversarial | Why |
|-----------|---------|-------------|-----|
| `drift_strength` | 0.3 | 0.8 | Stress test: model starts far from basin. |
| `trust_radius` | 0.05 | 0.03 | Tighter constraint under adversarial pressure. |
| `asymmetry_lambda` | 10.0 | 20.0 | Double safety bias needed when drift is 2.5x larger. |
| `curvature_weight` | 2.0 | 4.0 | Curved regions are more dangerous with large drift. |
| `lr` | 0.01 | 0.005 | Slower steps for stability under stress. |
| `confidence_threshold` (policy) | 0.4 | 0.5 | Earlier re-anchoring trigger. |
| `confidence_threshold_majority` | 0.7 | 0.8 | More aggressive majority pruning. |
| `reanchor_strength` | 0.1 | 0.2 | Stronger pull toward reference under adversarial drift. |

## Verification Workflow

Before merging any change, run this sequence:

```bash
# 1. Lint and format (must pass clean)
ruff check .
ruff format --check .

# 2. All tests including invariants (~4s)
python -m pytest tests/ -v

# 3. Smoke run with default config (should complete without error)
python main.py --config configs/default.yaml
```

If you changed any manifold layer, also run:
```bash
# 4. Invariant tests specifically (catches broken math contracts)
python -m pytest tests/test_invariants.py -v

# 5. Adversarial config (catches stability regressions)
python main.py --config configs/adversarial.yaml
```

If you changed `simulation/environment.py`:
```bash
# 6. Verify functional/module model parity
python -m pytest tests/test_environment.py::test_model_fn_matches_module -v
```

## Key Design Decisions

- **Saddle-point objective**: The parameter manifold loss is `task_loss - λ * safety_loss`, not `task_loss + λ * safety_loss`. The minus sign is intentional — it creates adversarial tension between task and safety. The trust region prevents runaway; the saddle-point structure ensures the repair explores the loss landscape rather than collapsing to a local minimum.
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
