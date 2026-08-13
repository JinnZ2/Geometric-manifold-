# NULL — κ_eff leading-indicator test

**Written before running.** `HARNESS.md` § 2: "null_model: mandatory. If you can't name the
null, you don't have an experiment."

## The null model

`no_injected_drift` — identical runs with the per-step drift injection set to zero.
Everything else is held fixed: same initial drift, same seeds, same thermodynamic config,
same warm-up window, same criteria, same signals. The model sits near the reference and
`basin_kl` stays below `epsilon_basin` for the whole run, so **no breach occurs and every
alarm is by definition a false alarm.**

## Why this null

The failure mode that matters for a leading indicator is an alarm that fires on anything.
An indicator that spikes reliably before a breach is worthless if it also spikes reliably
when nothing is happening, and the existing check in `experiment_stability.py` cannot
detect that at all — its alarm is defined as a within-run quantile, so it fires in 100% of
runs whether or not the model is drifting.

This is the same failure the E-P2 arc in notes/15 caught the hard way: v1 was refuted
because the null arm (rigid strut, creep only) fired detection at 96%. A leading-indicator
claim is only as good as its false-alarm rate, and the null arm is the only thing that
measures it.

## What the null decides

Per criterion and per signal, the null arm gives a false-alarm rate — the fraction of
null runs in which that criterion fires at all.

- A criterion with **null FP > 0.2 is disqualified**, and its results on the drift arm are
  reported but excluded from grading. A criterion that alarms on a quiet system has not
  detected anything on a drifting one.
- This gate is applied *before* looking at lead times, so a criterion cannot buy its way
  in by producing impressive leads.

The 0.2 threshold is the one notes/12 S6 used for the same purpose ("false-alarm rate
< 0.2 on null"), which is where this sim's design comes from.

## The second control: a trivial baseline

The null answers "does it fire on nothing?". It does not answer "is it worth its cost?".
That is the `theta_dist` baseline — ‖θ − θ_ref‖, which is free to compute and needs no
Hessian-vector product — run through the identical criteria and scored the same way.

IP-18 names this explicitly ("Theory B: coincident/lagging; trivial-baseline comparison").
κ_eff must beat it, not merely beat chance. `repair_energy` is carried as a third signal
for context.
