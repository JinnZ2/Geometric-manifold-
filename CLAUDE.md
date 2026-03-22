# CLAUDE.md

## Project Overview

**Basin Repair Framework** — a geometric manifold approach to maintaining neural network parameter safety. The system treats model safety as a geometric problem, using three interconnected manifold layers (data, parameter, policy) to detect when a model drifts from a safe reference state and applies repair mechanisms.

Total codebase: ~2,040 lines of Python + documentation. Research-stage project, not production-hardened.

## Repository Structure

```
├── main.py                          # Primary entry point
├── Mode-flag.py                     # Experimental mode selector
├── requirements.txt                 # Python dependencies (torch, numpy, scipy, etc.)
├── configs/
│   ├── default.yaml                 # Standard params (drift=0.3, steps=100)
│   └── adversarial.yaml             # Stress test (drift=0.8, steps=200)
├── simulation/
│   ├── environment.py               # ToyLLM (2-layer MLP), synthetic data generation
│   └── controller.py                # Main repair orchestration loop
├── manifolds/
│   ├── data_manifold.py             # GMR-style feature space cleaning
│   ├── parameter_manifold.py        # Curvature-aware basin repair in weight space
│   └── policy_manifold.py           # Trajectory-level alignment via JS divergence
├── repair/
│   ├── geometric_confidence.py      # Unified confidence (20% data, 50% param, 30% policy)
│   └── monitors.py                  # Metrics tracking & spike detection
├── addon_thermodynamic_control/     # Advanced energy/stability extensions
│   ├── energy.py                    # Fisher Information metric, thermodynamic accounting
│   ├── stability.py                 # Phase detection & stability analysis
│   ├── geometry_shaping.py          # Landscape shaping objectives
│   ├── addendum_formal_objectives.py # 4-term Lagrangian formalization
│   ├── experiment_energy.py         # Energy sweep experiments
│   ├── experiment_formal.py         # Formal objective experiments
│   └── experiment_stability.py      # Stability experiments
├── experiments/
│   ├── full_pipeline.py             # Runs all three manifold layers
│   ├── toy_landscape.py             # Single landscape experiments
│   ├── toy_landscape_v2.py          # Improved landscape version
│   ├── toy_landscape_v3.py          # Latest landscape version
│   ├── ablations.py                 # Ablation studies
│   ├── cost_analysis.py             # Repair cost analysis
│   └── cost_analysis2.py            # Advanced cost analysis
└── docs/theoretical_notes/
    └── saddle_dynamics_and_repair_cost.md  # Theoretical analysis
```

## Architecture: Three-Layer Geometric Framework

1. **Data Manifold** (`manifolds/data_manifold.py`): Asymmetric k-NN cleaning — aggressively removes noisy majority samples, preserves rare minority samples.
2. **Parameter Manifold** (`manifolds/parameter_manifold.py`): Curvature-aware basin repair with asymmetric loss (safety > task performance), trust region constraints.
3. **Policy Manifold** (`manifolds/policy_manifold.py`): JS divergence monitoring with soft re-anchoring when policy diverges from reference.

The controller (`simulation/controller.py`) orchestrates all three layers sequentially each step.

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Main simulation
python main.py --config configs/default.yaml

# Mode-based execution
python Mode-flag.py --mode simulate
python Mode-flag.py --mode energy_sweep

# Individual experiments
python experiments/full_pipeline.py
python experiments/ablations.py
```

## Dependencies

- Python 3.x
- torch >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pyyaml >= 6.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- tqdm >= 4.65.0

## Code Conventions

- **Configuration-driven**: All hyperparameters via YAML configs, no hardcoding.
- **Manifold interface pattern**: Each layer class exposes `.rectify()`, `.repair_step()`, or `.confidence()` methods.
- **Functional model interface**: Models called as `model_fn(inputs, theta)` for functional evaluation.
- **Metric dictionaries**: Methods return `{'metric_name': value}` for standardized logging.
- **Tensor-based**: PyTorch tensors throughout for GPU compatibility.
- **Docstrings**: Present on key methods, explaining geometric intuition.
- **No formal linting/formatting**: No flake8, black, or pylint configured.
- **No test suite**: Research project without pytest/unittest infrastructure.
- **No CI/CD**: No automated pipelines.

## Key Design Decisions

- **Asymmetric penalties**: Safety violations are penalized more heavily than task performance loss (controlled by `asymmetry_lambda`).
- **Trust regions**: Parameter updates are constrained to prevent catastrophic jumps (`trust_radius` in config).
- **Confidence weighting**: Unified confidence = 20% data + 50% parameter + 30% policy (see `repair/geometric_confidence.py`).
- **Thermodynamic extensions**: The `addon_thermodynamic_control/` module adds Fisher metric-based energy accounting and phase transition detection.

## Common Modification Patterns

- **Adding a new manifold layer**: Create a class in `manifolds/`, implement a confidence/repair interface, wire it into `simulation/controller.py` and `repair/geometric_confidence.py`.
- **Tuning repair behavior**: Edit YAML configs in `configs/`. Key knobs: `drift_magnitude`, `trust_radius`, `asymmetry_lambda`, confidence thresholds.
- **Adding experiments**: Create a script in `experiments/` that imports from `simulation/` and `manifolds/`.
- **Results output**: Monitor writes CSV to `results/` directory (configured in YAML).

## License

CC0 1.0 Universal (public domain).
