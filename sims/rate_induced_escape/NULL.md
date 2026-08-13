# NULL — rate-induced escape

**Written before running.**

## The null model

`zero_drift_no_forcing` — identical runs with sigma = 0. No perturbation is injected, so
the only thing moving theta is the repair loop itself.

## Why this null

The measured quantity is a **rate balance**: drift adds KL, repair removes it, and the
claim is about which wins. The null removes the forcing entirely and leaves the corrector
running, which isolates the corrector's own behaviour.

This matters because the whole argument rests on the repair having a fixed, small capacity.
If the unforced system showed KL *rising*, the repair loop would be actively harmful rather
than merely capped, the saturation framing would be the wrong description, and every rate
comparison downstream would be measuring the wrong thing.

## What the null must show

- **Net dKL/step <= 0 with no drift.** The repair either holds the basin or slowly improves
  it when nothing is pushing.
- It also fixes the intercept: the sigma^2 fit for drift-induced KL must pass through
  approximately the null's value at sigma = 0. A large positive intercept would indicate
  the model is drifting for reasons unrelated to injected sigma, and the whole crossing
  calculation would be measuring an artifact.

## What it does not cover

The null shares the initial condition (`drift_strength = 0.08`) with every other arm, so it
does not test whether the result depends on starting near that particular distance from the
reference. The saturation balance predicts sigma_crit depends on the local KL geometry,
which varies with position; a start much closer to or further from theta_ref would move it.
That is a stated limitation, not a controlled variable.
