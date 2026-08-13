# REFUTE — κ_eff as a leading indicator of basin breach (IP-18)

**Written before running the sweep.**

## The claim under test

`addon_thermodynamic_control/stability.py` line 32 states it directly:

> Spike in kappa_eff precedes behavioral collapse.

and `experiment_stability.py` lists it as check 3: "Does kappa_eff spike before basin_kl
exceeds epsilon? (leading indicator)". `INTEGRATION_POINTS.md` IP-18 asks for this to be
turned into a pre-registered kill test — Theory A (κ_eff leads the KL breach) against
Theory B (coincident/lagging, and no better than a trivial baseline).

κ_eff is the Rayleigh quotient of the safety Hessian along the flow direction. It costs a
Hessian-vector product every step. The question is not only whether it leads, but whether
it earns that cost against a baseline that is free.

## Why the existing check cannot answer this

`experiment_stability.py` computes the alarm as:

```python
kappa_90 = df['kappa_eff'].quantile(0.9)
kappa_spike_step = df[df['kappa_eff'] > kappa_90]['step'].min()
```

Three defects, all fatal to a leading-indicator claim:

1. **Look-ahead.** The threshold is the 90th percentile of the *whole run*, including
   steps after the breach. A real indicator cannot use future data.
2. **It cannot fail to fire.** By construction exactly 10% of steps exceed the run's own
   90th percentile, so an alarm always exists and the false-alarm rate is never measured.
3. **No baseline.** Nothing checks whether a free signal does as well.

Every criterion in this sim is therefore **causal** — thresholds are fixed from a warm-up
window and applied forward — and every one is run against a null arm and a trivial
baseline.

## Pre-committed criteria

`HARNESS.md` retrofit queue item 3 records that a previous version of this test had its
"verdict flipped on criterion choice" and "needs the full criterion-sweep recorded in
config." So the criterion is swept, not chosen: 7 criteria (ratio-to-warm-up-median at
1.5/2/3, warm-up z-score at 2/3, trailing-window Kendall τ at 0.3/0.5) × 3 signals, all
recorded in `config.json`, and the per-criterion verdict table is part of the output.

**Theory A (SUPPORTED)** requires, for at least one criterion whose null false-alarm rate
is ≤ 0.2, at ≥ 80% of (seed, σ) cells:
- κ_eff alarms **before** the breach (lead > 0), and
- κ_eff's lead **exceeds** the θ-distance baseline's lead.

**Theory B (REFUTED)** holds if, under *every* criterion with an acceptable false-alarm
rate, κ_eff fails to lead or fails to beat the free baseline at ≥ 80% of cells.

Anything else is INCONCLUSIVE — a first-class verdict here, and the one the sibling
ecosystem's κ_eff attempt already landed on once.

**The grading is deliberately generous to Theory A**: it needs only one viable criterion
out of seven to succeed, while refutation requires all seven to fail. A refutation under
that asymmetry is strong; a support under it is weak and would need the per-criterion
table read carefully before anyone leaned on it.

## Disclosure: what was already known before writing this

Honesty about priors, since these thresholds are authored here rather than transcribed:

- A single exploratory trajectory was run while scoping the apparatus. In it κ_eff stayed
  in roughly 0.005–0.018 with no visible spike as basin_kl rose through ε. That is one
  seed, unswept, ungraded — but it was seen, and it would be dishonest to present the
  criteria below as chosen in ignorance of it.
- The criteria were nonetheless fixed on principle — causal, spanning threshold and trend
  families, spanning generous to strict — rather than tuned toward any outcome, and the
  generous-to-Theory-A grading above is the check on that.

## A prerequisite finding that shapes the scenario

The apparatus **cannot produce a breach event on its own**. Measured across drift levels
at ε = 0.15: drift 0.05 → basin_kl ≈ 0.021 (never breaches); 0.1 → 0.125 (never breaches);
0.2 → 1.12; 0.4 → 8.0; 0.6 → 20.5 (all already breached at step 0, then flat or slowly
repairing). There is no crossing, because `sigma_drift` is only a look-ahead sampling
radius inside `_proactive` and nothing perturbs θ during a run.

So "κ_eff leads the breach" is untestable as the framework stands: with no crossing there
is nothing to lead. This sim therefore **injects per-step drift** to create the walk-out
scenario the claim presupposes. That is a change to the experimental scenario, not to the
framework, and it is the minimum needed to give the claim content. It is declared here
rather than buried in the code.

## Scenario calibration, disclosed

The first configuration put the breach at step 9–13 with a 10-step warm-up, leaving almost
no window in which an alarm *could* lead — the experiment would have been incapable of
detecting the effect regardless of whether it exists. σ and the run length were therefore
recalibrated against measured breach times (σ ≤ 0.008 never breaches within 60 steps;
σ = 0.012 breaches at step 34–46), and the sweep set to σ ∈ {0.012, 0.016, 0.020} over 60
steps.

This is apparatus calibration — choosing an observation window in which the event occurs —
in the same sense that E-P8 must pick a load range where the strut actually snaps. **It
touches the scenario only. The criteria, the null gate, the baseline, and `refute_if` were
fixed before any of it and are unchanged.** Calibration performed after seeing which
criterion won would be a different and illegitimate act; that did not happen, and the
per-criterion table in the output is there so the claim can be checked rather than trusted.

## What would make this sim itself wrong

- If injected isotropic Gaussian drift is not representative of the drift the claim is
  about. A targeted adversarial drift might excite curvature differently. Only the
  isotropic case is tested.
- If 40 steps is too short for a κ_eff rise to develop. The breach lands near step 10–12
  at these σ values, so a slow indicator would be truncated. Cells where the breach occurs
  at step 0 or never are marked unusable rather than scored.
