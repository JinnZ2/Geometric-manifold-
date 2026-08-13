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

## Relationship to prior work

This repository studies runtime assurance for neural networks under distribution
shift. A reference model defines a KL trust region in behavior space; a controller
applies curvature-regularized, trust-region-bounded updates that pull drifting
parameters back toward the reference, with asymmetric penalties preserving
minority-class behavior. A spectral statistic of the safety Hessian is evaluated as a
leading indicator of instability. Stability guarantees (ISS, Lyapunov) are stated as
open problems, not results.

The framework's internal vocabulary maps onto standard research terminology:

| This repo | Standard term | Anchor literature |
|---|---|---|
| Basin repair framework | Trust-region-constrained fine-tuning with a reference anchor | Schulman et al., TRPO (2015); KL-regularized RLHF (Ouyang et al. 2022) |
| Safe basin `B_θ = {θ : KL(f_θ‖f_θ₀) < ε}` | KL-ball constraint / trust region in distribution space | TRPO (2015); behavioral-cloning anchors in offline RL |
| Parameter manifold repair | Projected/regularized optimization in weight space | Nocedal & Wright, trust-region methods; elastic weight consolidation (Kirkpatrick 2017) |
| `kappa_eff = θ̇ᵀHθ̇ / θ̇ᵀθ̇` | Rayleigh quotient of the Hessian; spectral instability indicator | Ghorbani et al. 2019; Cohen et al., "edge of stability" (2021) |
| Curvature-weighted safety loss | Second-order / curvature-aware regularization | Optimal Brain Damage/Surgeon (LeCun 1990; Hassibi 1992); K-FAC (Martens & Grosse 2015) |
| Asymmetric repair penalty (`asymmetry_lambda`) | Cost-sensitive loss; constrained optimization via Lagrangian | Altman 1999; Achiam et al., CPO (2017) |
| Saddle-form total loss (`task − λ·safety`) | Adversarial/Lagrangian saddle-point objective | Standard in safe RL and GAN literature |
| Fisher energy accounting `C = δᵀGδ` | Natural-gradient metric; Mahalanobis step cost | Amari, natural gradient (1998); K-FAC |
| Energy trend spike detection | Change-point / anomaly detection on loss curves | CUSUM; early-warning-signal literature (Scheffer 2009) |
| GMR data cleaning (minority never dropped) | Neighborhood-based label-noise filtering with class-conditional weighting | Wilson & Martinez editing; class-imbalance literature |
| Policy manifold JS re-anchoring | Distributional alignment via JS divergence; EMA target | JS-GAN lineage; target networks in RL |
| Phase classifier (stable/threshold/critical) | Regime classification on stability indicators | Scheffer et al. 2009, 2012 |
| `ISS_proof_pending` | Input-to-state stability analysis (open problem) | Sontag, "Input to State Stability" (1989+); Jiang & Wang 2001 |

Full cross-repo mapping and citation posture: [`docs/research/TERMINOLOGY_MAP.md`](docs/research/TERMINOLOGY_MAP.md).

## License

CC0 1.0 Universal (public domain)
