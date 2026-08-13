# NULL — shape shadow

**Written before running.**

## The null model

`random_residual_ensemble` — residual vectors drawn isotropically on the unit sphere in
R^12 and rescaled to the same norm as the test residual, pushed through the identical
relaxation and the identical observation levels.

## Why this null

The claim is comparative: a blind direction is blind *relative to* what an ordinary fault
of the same size produces. Without the ensemble there is no scale to judge "small" against,
and any absolute threshold on shadow magnitude would be arbitrary — it would depend on the
stiffness constant, the residual magnitude, and the relaxation step count, none of which
carry physical meaning here.

The ensemble fixes all three by construction: same magnitude, same relaxation, same
observables. Only the *direction* in fault space differs. That isolates the one variable
the claim is about.

## What the null must show

- A well-defined typical response with a spread narrow enough for a 10x ratio to be
  meaningful. If the random ensemble's own shadow magnitudes vary by more than 10x among
  themselves, a blind direction 10x below typical is not distinguishable from an ordinary
  quiet direction, and H3 cannot be evaluated — that outcome is reported as INCONCLUSIVE
  rather than being read as support.
- Non-zero typical response at every observation level, confirming the instrument responds
  to faults at all before anything is concluded about what it misses.

## What it does not cover

The ensemble is isotropic in residual space, which treats all 12 edges as exchangeable.
Real fault distributions are not isotropic — some equations fail more often than others —
so "typical" here means typical under a uniform prior, not under an empirical fault
distribution. Whether the blind directions coincide with faults that actually occur is a
separate question this sim cannot answer.
