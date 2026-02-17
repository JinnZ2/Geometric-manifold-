# Saddle Dynamics and Repair Cost: Theoretical Notes

*Synthesized from collaborative analysis across multiple models and one truck driver.*

---

## The Core Question

Does repair energy scale superlinearly with instability magnitude,
or does it scale smoothly with gradient size?

These are different claims. The experiment must distinguish them.

---

## Why the Current Toy Landscape Is a Quartic Potential

The double-well potential along the axial direction is:

  V(s) = (s² - r²)²  =  s⁴ - 2r²s² + r⁴

Its gradient and curvature are analytically coupled:

  dV/ds       = 4s³ - 4r²s       (gradient)
  d²V/ds²     = 12s² - 4r²       (curvature = lambda_axial)

Near the saddle (s ≈ 0):

  dV/ds  ≈  -4r²s                 (linear in s)
  d²V/ds²≈  -4r²  + 12s²         (constant + quadratic in s)

This coupling has a direct consequence for the experiment:

  repair delta   = -gamma * grad
  repair energy  = ||delta||²  ∝  ||grad||²  ∝  s²   (near saddle)

  lambda_axial   ≈  -4r²  +  12s²

So repair energy scales as s², while lambda_axial shifts as s²
from a constant baseline of -4r².

**The separation only becomes measurable when |s| grows large enough
that the 12s² term meaningfully shifts lambda_axial magnitude
away from the -4r² baseline.**

This is not a flaw. It is physics.
The quartic potential forces a specific coupling between
gradient and curvature that constrains what is observable.

---

## What repair_start_s Actually Controls

`repair_start_s` is not just a trigger threshold.
It is the primary knob for exposing curvature-dominated scaling.

Small repair_start_s → repairs fire near saddle → s small →
  gradient and curvature both near baseline → energy scaling looks linear

Large repair_start_s → repairs fire deep in unstable region → s large →
  12s² term dominates → lambda_axial grows → curvature-energy separation visible

**Run the experiment across a range of repair_start_s values.
If quartile ratio increases with repair_start_s: geometry is biting.
If ratio stays flat: repair cost is gradient-dominated throughout.**

---

## Instability Is Not Just Negative Curvature

Negative lambda_axial means the saddle is unstable along the axis.
But instability requires two conditions:

  1. lambda_axial < 0          (negative curvature: necessary)
  2. theta_dot aligned with u  (velocity in unstable direction: necessary)

If theta_dot has low projection onto the axial unit vector u,
the system won't diverge even with lambda_axial < 0.
The unstable manifold exists but the trajectory isn't probing it.

**The strongest geometric test is not repair energy.
It is growth rate of small axial perturbations after saddle crossing.**

If perturbations along u grow exponentially when lambda_axial < 0:
  instability is real and the manifold is dynamically active.

Repair energy scaling is a secondary, controller-mediated effect.
Confirm the primary effect first.

---

## What the Divergence Ratio Actually Tests

Post-crossing speed increase confirms the unstable manifold
is active in the dynamics, not just in the analytic expression.

If divergence ratio stays near 1.0 after saddle crossing:

Three possible explanations (in order of likelihood):

  1. Integrator damping: discrete gradient descent with small lr
     damps exponential growth. Instability exists but is masked
     by the integrator, not absent.

  2. Drift dominance: drift force overwhelms curvature structure.
     The trajectory is being pushed, not growing.

  3. Task term stabilization: L_task is pulling toward safe well
     strongly enough to suppress axial divergence.

To distinguish: reduce lr, reduce lambda_s, reduce drift_strength
one at a time and recheck divergence ratio.

---

## Two Distinct Outcomes and What Each Means

### Outcome A: Instability exists, repair cost superlinear

  - No-repair: s crosses saddle, speed increases, perturbations grow
  - Repair quartile ratio > 2-3x as repair_start_s increases
  - Repair cost scales with lambda_axial magnitude

**Interpretation:**
The controller is paying to counteract instability, not just distance.
Thermodynamic framing has dynamical content.
Phase-transition-like cost structure is real.

### Outcome B: Instability exists, repair cost linear

  - No-repair: s crosses saddle, speed increases (instability confirmed)
  - Repair quartile ratio stays 1.0-1.5x regardless of repair_start_s
  - Repair cost tracks gradient magnitude, not curvature

**Interpretation:**
The thermodynamic metaphor is overstated for this potential.
Safety maintenance is a smooth control-energy tradeoff.
No phase transition. No spike.
The hypothesis weakens but does not die:
  it may require a sharper potential (steeper ridge, narrower barrier).

Both outcomes are scientifically interesting.
Outcome B is more honest and more common in first experiments.

---

## What Would Strengthen the Hypothesis

If Outcome B: the quartic potential's analytical coupling
between gradient and curvature prevents clean separation.

Options to sharpen the test:

  1. Use a potential with independent gradient and curvature control:
       V(s) = -A * exp(-s²/2σ²)   (Gaussian barrier)
     Gradient and curvature are now decoupled by σ.
     Narrow σ = sharp ridge = large curvature, controlled gradient magnitude.

  2. Measure repair cost per unit gradient magnitude:
       normalized_E = repair_energy / ||grad||²
     If this is nonconstant across repair_start_s values:
     curvature is contributing beyond pure gradient scaling.

  3. Test the axial perturbation growth rate directly:
     Add small axial perturbation after saddle crossing.
     Measure e-folding time. Compare to 1/sqrt(|lambda_axial|).
     If they match: you have confirmed the instability time scale analytically.

---

## The Honest Summary

The universe does not reward pretty theories.
It rewards ones that survive contact with eigenvalues.

Current status of the hypothesis:

  - Basin geometry is well-defined: confirmed analytically
  - Saddle structure is real: lambda_axial sign flip is analytic fact
  - Unstable manifold existence: testable via divergence ratio
  - Repair cost nonlinearity: still unconfirmed, requires deep unstable excursion
  - Phase-transition-like spike: plausible but undemonstrated

Next clean experiment: sweep repair_start_s from 0.1 to 1.2 in steps of 0.1.
Plot quartile ratio vs repair_start_s.
If monotonically increasing: geometry is biting.
If flat: gradient dominates throughout and a sharper potential is needed.

---

## Open Questions

  1. Can we construct a potential where gradient magnitude is bounded
     but curvature diverges near the barrier?
     That would isolate curvature contribution cleanly.

  2. Does the result generalize from 1D axial dynamics to high-dimensional
     parameter space of a real neural network?
     The quartic toy is necessary but not sufficient.

  3. Is the Lyapunov candidate V = L_safety + mu*C_repair
     monotonically decreasing under the repair dynamics demonstrated here?
     Empirical check: track V in the repair_on_unstable_manifold run.
     If V ever increases: the candidate needs revision.

  4. Input-to-state stability under adversarial perturbations:
     ISS_proof_pending = True.
     This remains the open theoretical problem.
