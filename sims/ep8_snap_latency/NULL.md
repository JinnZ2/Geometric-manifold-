# NULL — E-P8 snap latency

**Written before running.** `HARNESS.md` § 2: "null_model: mandatory. If you can't name
the null, you don't have an experiment."

## The null model

`shuffle_eps0_tsnap_pairing` — within each (seed, ramp_rate) cell, the observed snap times
are randomly re-paired against the initial compressions, then the identical decoder is
fitted and scored on the identical held-out split.

Everything else is held fixed: same trials, same integrator, same noise draws, same
80/20 split, same OLS decoder. The *only* thing destroyed is the correspondence between
an initial condition and the latency that is supposed to encode it.

## Why this null and not another

The claim is that snap latency **carries information about the initial condition**. The
honest null is therefore not "a strut that never snaps" — that arm would differ in
dynamics, noise, and sample count all at once, and a difference could be attributed to any
of them. Shuffling the pairing leaves every marginal distribution intact (the same set of
ε_0 values, the same set of t_snap values) and removes only the mutual information. Any
decoding skill that survives the shuffle is an artifact of the fitting procedure, not
evidence for the claim.

This is also what makes the null diagnostic rather than decorative. The prior sim in this
lineage, `experiments/snap_information_sim.py`, reported a "1 bit of history" result that
a shuffle test would have destroyed instantly — because the two arms it compared were
byte-identical trajectories, and the reported difference came from measuring distance to
two different reference points rather than from any dynamics.

## What the null must show for the main result to mean anything

A meaningful positive result requires the shuffled decoder to be **no better than the
mean-ε_0 baseline** — i.e. `rmse_null ≈ rmse_baseline`, both materially worse than
`rmse_decoder`. If the shuffled decoder scores well, the split or the fit is leaking and
the run is void regardless of what the real arm reports.

## Second baseline (not a null, but reported with it)

`mean_eps0_baseline` — predict the training-set mean ε_0 for every held-out trial,
ignoring t_snap entirely. This is the "no better than baseline" quantity the notes/15
refutation condition names explicitly, so it is computed per cell and stored alongside
the null in `metrics.json`.
