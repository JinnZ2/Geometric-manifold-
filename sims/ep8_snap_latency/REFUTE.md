# REFUTE — E-P8 snap latency

**Written before running.** These thresholds are *transcribed verbatim* from
`docs/research/15_physical_shape_instrument.md` § E-P8, where they were pre-registered.
They were not chosen by the author of this sim, and they were not adjusted after seeing
any output. That provenance is the whole reason this file can be trusted.

## The claim under test

> Under a controlled quasi-static load ramp (constant dε/dt), the time-to-snap after ramp
> start encodes the strut's initial distance-from-threshold via `t_snap ∝ ε_0^(−1/2)`
> (fold law), such that snap latency is a decodable measurement of the initial condition —
> the snap is an ADC.

## Pre-committed pass condition (notes/15, verbatim)

> **Pass:** exponent in [−0.65, −0.35] (fold −1/2 with tolerance for rate effects) AND
> decoder RMSE ≤ 0.02 compression (≈ one E-P2 step).

## Pre-committed refutation condition (notes/15, verbatim)

> **Refuted if:** exponent outside band (wrong bifurcation or rate-dominated dynamics),
> OR decoder no better than the mean-ε_0 baseline.

Note the asymmetry, which is in the source and is preserved here: the pass condition is a
conjunction over *exponent AND RMSE*, while the refutation condition fires on *exponent
OR baseline*. A run can therefore fail to pass without meeting the literal refutation
condition — that region is graded INCONCLUSIVE, not quietly rounded to either verdict.

## Aggregation rule (this sim's only addition)

notes/15 states the criteria per-fit; it does not say how to aggregate across seeds and
ramp rates. `HARNESS.md` § 4 does, so its rule is applied:

- **SUPPORTED** — pass condition holds at ≥ 80% of (seed, ramp_rate) cells, at every
  ramp rate, and the decoder beats both the mean-ε_0 baseline and the shuffled null.
- **REFUTED** — the refutation condition holds at ≥ 80% of cells.
- **INCONCLUSIVE** — anything else. This is a first-class verdict, per HARNESS § 4.

## A known ambiguity in the source, resolved in advance

The claim is stated as `t_snap ∝ ε_0^(−1/2)` (a power of the *initial compression*), but
the protocol's step 3 says "Fit log t_snap vs log ε-distance" (a power of the *distance
to threshold*, `c_snap − ε_0`). These are different regressors and cannot both be the
exponent the band refers to.

Resolution, committed before running: **the primary fit follows the protocol's own
instruction** — log t_snap vs log(c_snap − ε_0). The alternative regressor (log ε_0) is
computed and reported alongside it as a secondary metric, so the grade does not depend on
which reading a later reader prefers. Both are in `metrics.json`.

## What would make this sim itself wrong

- If the fold normal form is not the right local model for the von Mises truss near
  snap-through. The whole ecosystem assumes it is (`k_eff ∝ √(1 − c/c_snap)`); this sim
  inherits that assumption rather than testing it.
- If the integration is not converged. A refinement check at `dt/4` is run every time and
  written to `metrics.json` under `numerics_check`; a max relative deviation above 1e-3
  there invalidates the run regardless of verdict.
