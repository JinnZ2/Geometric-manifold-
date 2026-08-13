# REFUTE — can a dark constraint be detected, and when?

**Written before running.** Prompted by the observation that dark matter is a kind of
shadow shape: it is not a fault hiding inside an observable, it is an **unmodeled
component inferred from the residual between what the visible model predicts and what is
observed.** Weak lensing is exactly this — the shapes of background galaxies are distorted
by mass that is invisible in the same channel.

This is a different and better operationalization than the one in `sims/shape_shadow/`,
which asked whether a *known-space* fault could hide in a projection. Here the fault is
outside the model's space entirely.

## The setup

The octahedron has 12 edges. Its only non-edges are the 3 antipodal pairs, so those are the
only places a hidden constraint can live. The true system carries a brace on one antipodal
pair; the observer's model knows only the 12 visible edges and fits residuals to them.

## The counting argument, which makes this sharp

The octahedron is isostatic: 12 shape degrees of freedom, 12 visible edges. Under a
**single** load case the observer has 12 observations and 12 free parameters, so the
visible-only model can fit *any* deformation exactly — including one produced partly by a
constraint it does not model. The dark component is absorbed into biased visible residuals
and leaves no trace.

Detection therefore requires **more independent observations than parameters**: K load
cases give 12K observations against 12 unknowns, and at K >= 2 the system is overdetermined
so an unmodelable component must show up as irreducible residual.

This is the same structure as the astrophysical case. A single probe admits degenerate
explanations — modified dynamics versus unseen mass, the mass-sheet and disk-halo
degeneracies — and the case is closed only by combining independent probes: rotation
curves, lensing, the CMB, and the Bullet Cluster's separation of lensing mass from baryonic
mass.

## Pass condition

- **H1 (degeneracy at K=1):** unexplained residual <= 1e-6 at K=1 for every dark strength.
  The visible model absorbs the dark constraint completely and the observer sees nothing
  wrong.
- **H2 (detection at K>=2):** unexplained residual grows monotonically with dark strength.
- **H3 (separability):** at K>=2 and the largest dark strength, the residual exceeds the
  dark_strength = 0 null by at least 10x.

## Refutation condition

Any of: the dark constraint is *not* absorbed at K=1 (the counting argument is wrong and
one probe suffices); the residual does not grow with dark strength at K>=2 (the extra load
cases carry no information); or the residual is not separable from the null (detection is
not achievable at these magnitudes).

## What a SUPPORTED verdict would mean

That the framework has a measurable dark-component blind spot with a known cure. A single
observation of a shape cannot distinguish "my modelled equations are misbehaving" from
"there is a constraint I have not modelled" — the two are exactly degenerate — and the
visible residuals reported by drill-down would be **biased, not merely incomplete**. It
would also say the cure is cheap: perturb the system in more than one way and the hidden
component announces itself.
