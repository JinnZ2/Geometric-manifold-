# NULL — fractal basin boundary vs damping

**Written before running.** `HARNESS.md` § 2: "null_model: mandatory. If you can't name the
null, you don't have an experiment."

## The null model

`smooth_boundary_control_field` — a synthetic basin field on the identical grid, whose
label comes from a smooth analytic function (`label = 0 if x < x_mid else 1`) instead of
from integrating any dynamics. The identical α estimator, at the identical scales and
probe count, is then run on it.

## Why this null

The measured quantity here is a property of the *estimator applied to a label field*, not
a direct observation. The failure mode that matters is therefore an estimator that reports
α < 1 on fields that are not fractal at all — a fractal-looking number manufactured by
finite grid resolution, probe placement, or the log-log fit range.

A straight boundary has uncertainty exponent α = 1 exactly: the probability that a random
interval of width ε straddles a single smooth curve is proportional to ε. So the null has a
known analytic answer, and any material departure from α = 1 is measurement artifact with
nowhere else to hide.

This is a null in the strict sense demanded by `HARNESS.md`: it is not a weaker version of
the effect, it is the condition under which the effect is *absent by construction*.

## What the null must show

- **α_null ≥ 0.90.** Above this, the estimator is trustworthy at this grid resolution and
  a measured α of 0.69 or 0.39 means what it appears to mean.
- **Below 0.90 the entire run is void**, whatever the sweep shows. An estimator that
  fractalizes a straight line cannot be used to argue that a real boundary is fractal, and
  the γ-trend would then be a trend in the artifact rather than in the physics.

## What this null does *not* cover

It validates the estimator, not the dynamics. It cannot detect an integration that has not
converged, a grid too coarse to resolve the true boundary structure, or a `T` too short
for trajectories to commit to a well. Those would show up as α values that drift with `N`
or `T`, which this sim does not sweep — a stated limitation, not a silent one.
