# REFUTE — fractal basin boundary vs damping

**Written before running the sweep.**

## Provenance of these thresholds — read this first

Unlike `sims/ep8_snap_latency/`, whose criteria were transcribed verbatim from a
pre-registration in notes/15, **the thresholds below are authored here**. notes/17 reports
α at a single damping and states no quantitative prediction for how α varies with γ, so
there was nothing to transcribe. They are pre-committed in the sense that they are written
before the sweep is run and are not adjusted afterwards — not in the stronger sense that
someone else committed to them first. That is a weaker warrant and is labelled as such.

**One value here is already known**, and pretending otherwise would be dishonest: the
γ=0.25 result (α_double = 0.688, α_triple = 0.392, Wada 8.0%) was reproduced in this repo
before this sim was written, from `experiments/fractal_basin_sim.py`. It is therefore a
**regression check, not a discovery**, and is graded separately from the sweep claim. The
other four damping values have not been run.

## Background

notes/17 §1 measured, at γ = 0.25 only:

> Bistable (the physical strut's potential): **α = 0.69** → boundary dimension 1.31. The
> boundary is *already fractal* at moderate damping (chaotic saddle on the barrier).

`HARNESS.md` retrofit queue item 4 names the deficiency: "alpha measured at single damping
— sweep gamma mandatory." `gamma` is a parameter of the original `basin_grid()` but was
only ever called at its 0.25 default.

## The claim under test

**C1 (regression).** At γ = 0.25 the measured α reproduces notes/17 within ±0.03.

**C2 (the sweep claim).** The uncertainty exponent α increases with damping γ. Mechanism:
damping suppresses the chaotic transient on the barrier, so trajectories commit to a well
sooner and the boundary becomes smoother; α → 1 is a smooth boundary, α → 0 a maximally
fractal one.

Direction is committed in advance: **α should rise with γ.**

## Pass condition

- C1: |α_measured − α_notes17| ≤ 0.03 for both the double and triple well at γ = 0.25.
- C2: Kendall τ of α vs γ is **positive** at ≥ 80% of seeds, **and** the span
  α(γ=1.00) − α(γ=0.05) ≥ 0.10.
- Null: the smooth-boundary control returns α ≥ 0.90.

## Refutation condition

- C2 refuted if τ ≤ 0 or the span < 0.10 at ≥ 80% of seeds — i.e. α is flat or falls with
  damping. Either outcome contradicts the stated mechanism.
- The whole run is void, regardless of verdict, if the null returns α < 0.90: that would
  mean the estimator reports fractal boundaries where none exist, and no measured α on a
  real system could then be trusted.

## What the seeds do here, precisely

The grid integration is deterministic — a given (γ, system) always produces the same basin
field. **Seeds vary only the random probe sampling of the α estimator.** They therefore
quantify estimator variance, not dynamical variance, and a tight spread across seeds says
the α estimate is stable, not that the phenomenon is robust. Robustness across dynamics is
what the γ sweep is for. Stating this distinction matters because reporting eight seeds
would otherwise overstate the independence of the evidence.
