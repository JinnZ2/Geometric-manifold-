# Stability Module

## What This Implements

A coupled nonlinear dynamical system with:
- Riemannian gradient flow in parameter space
- Induced policy evolution
- Data manifold state tracking
- Lyapunov function evaluation at each step
- Spectral stability certificate

## The Lyapunov Candidate

  V(theta, pi) = L_safety(theta)
               + lambda_pi * R_risk(pi)
               + alpha * KL(pi_theta || pi_theta0)
               + mu * C_repair_cumulative

dV/dt is estimated each step. Stability requires dV/dt <= 0.

## Spectral Stability Condition

  sup_theta lambda_max(Hessian L_safety) < C_bound

If this fails: repair cost diverges. Thermodynamic phase transition.
Tracked via power iteration on Hessian-vector products.

## Early Warning Scalar

  kappa_eff(t) = theta_dot^T H_safety theta_dot / theta_dot^T theta_dot

Rayleigh quotient of safety Hessian along current flow direction.
Spikes before behavioral collapse. Run experiment_stability.py to verify.

## Run

  python addon_thermodynamic_control/experiment_stability.py

Key output: results/stability/certificates.csv
Column 'empirical_stability' = True means both Lyapunov condition
and spectral bound held over the full trajectory.

## Open Problem

Prove input-to-state stability (ISS) under adversarial perturbations
given bounded curvature and bounded repair energy:

  ||delta(t)|| <= gamma(||disturbance||) + beta(||theta(0) - theta_ref||, t)

That converts empirical stability into a robustness theorem.
ISS_proof_pending = True in all certificate summaries until resolved.
