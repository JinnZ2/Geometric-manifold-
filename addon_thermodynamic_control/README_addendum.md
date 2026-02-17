# Addendum: Formal Objectives

## What This Adds

The base framework uses geometric confidence and curvature proxies as
heuristics. This addendum formalizes them as proper functionals with
physical meaning.

## The Four-Term Lagrangian

  L = L_task + lambda_s*L_safety + lambda_p*J_proactive + mu*C_repair

| Term | Physical meaning | Implemented in |
|------|-----------------|----------------|
| L_task | Performance pressure | `task_loss()` |
| L_safety | Basin potential: KL from ground state | `safety_basin_potential()` |
| J_proactive | Curvature flattening: smooth the basin wall | `proactive_objective()` |
| C_repair | Thermodynamic expenditure: kinetic energy cost | `repair_energy_step()` |

## Key Prediction

  kappa_eff = lambda_max(Hessian of L_safety)

Spikes **before** behavioral failure (basin_kl exceeds epsilon).

This is a leading indicator. Verify with:
  python addon_thermodynamic_control/experiment_formal.py

Expected output: kappa_eff crosses 90th percentile N steps before
basin_kl crosses epsilon_s. N > 0 validates the hypothesis.

## Riemannian Gradient Flow

  theta_dot = -G^{-1} grad_theta L

Not Euclidean gradient descent. Movement in parameter space is weighted
by the Fisher metric. High-curvature directions are expensive to traverse.

## Stability Condition

Safe basin is asymptotically stable iff:
  1. Hessian of L_safety is positive definite near basin floor theta_ref
  2. mu penalizes oscillatory boundary hopping

## Open Problem: Lyapunov Certificate

Candidate:
  V(theta) = L_safety(theta) + mu * C_repair_cumulative

If dV/dt <= 0 along trajectories then basin is provably stable.
Not yet proven. `UnifiedLagrangian.lyapunov_candidate()` tracks V empirically.

That's the next step: from engineering heuristic to stability theorem.
