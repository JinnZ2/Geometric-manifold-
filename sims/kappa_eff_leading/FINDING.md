# FINDING — IP-18: κ_eff is too noisy to be graded, and the criterion sweep shows why

Run 2026-08-13T2001Z · 5 seeds × 3 drift rates (15 drift runs, 12 usable) + 5 null runs ×
60 steps · 7 criteria × 3 signals · ~41 s · graded by `run.py` against `REFUTE.md`.

## Verdict: INCONCLUSIVE

`HARNESS.md` § 4 lists INCONCLUSIVE as a first-class verdict and cites the sibling
ecosystem's own κ_eff attempt as the example. This run lands in the same place, and the
reason is specific rather than vague.

Only **one of seven criteria survived the null arm**, and under it κ_eff led the breach and
beat the free baseline in 3 of 12 usable cells (25%) — above the 20% refutation gate,
below the 80% support gate. The verdict turns on three cells.

## The criterion sweep is the result

| criterion | null FP (κ_eff) | null FP (θ-dist) | viable | κ leads | supports A |
|---|---|---|---|---|---|
| ratio_1.5 | **0.80** | 0.00 | no | 33% | 33% |
| ratio_2.0 | **0.80** | 0.00 | no | 25% | 25% |
| ratio_3.0 | 0.20 | 0.00 | **yes** | 25% | 25% |
| z_2.0 | 0.60 | 0.80 | no | 17% | 0% |
| z_3.0 | 0.60 | 0.80 | no | 17% | 17% |
| tau_w8_0.3 | **1.00** | 1.00 | no | **75%** | 0% |
| tau_w8_0.5 | 0.40 | 1.00 | no | 50% | 0% |

Read the `tau_w8_0.3` row against the `ratio_3.0` row. The τ-trend criterion gives κ_eff
its **best** leading performance in the table — it alarms before the breach in 75% of cells
— and it also fires on **every single quiet run** (FP = 1.00). A study that reported lead
times without a null arm would have picked that row and called it strong evidence for
Theory A. With the null arm, it is the weakest row in the table.

That is precisely the failure `HARNESS.md` retrofit queue item 3 records for the previous
version of this test — "verdict flipped on criterion choice" — reproduced here in a single
table instead of across two write-ups.

## What this says about κ_eff

**κ_eff alarms on quiet systems.** Under the two mild ratio criteria it fires on 4 of 5
runs where nothing is drifting and no breach ever occurs. Only a 3× threshold over the
warm-up median holds the false-alarm rate at the 0.2 gate, and at that strictness it
catches the breach in advance only a quarter of the time.

**The free baseline is better behaved.** ‖θ − θ_ref‖ costs nothing — no Hessian-vector
product — and holds FP = 0.00 under all three ratio criteria, against κ_eff's 0.80/0.80/0.20.
κ_eff is paying a Hessian every step to be noisier than a norm.

**Nothing here refutes the underlying geometry.** The claim tested is operational: does
this scalar, under a causal threshold, fire before the basin breach more usefully than a
free alternative? At this model scale and drift regime, it does not clearly do so. Whether
κ_eff leads at larger scale, under adversarial rather than isotropic drift, or with
smoothing applied, is untested and remains open.

## Two defects in the existing apparatus, found on the way

**1. The framework cannot produce a breach event.** Measured at ε_basin = 0.15:

| drift_strength | basin_kl at step 0 | behaviour |
|---|---|---|
| 0.05 | 0.021 | below ε, stays below — never breaches |
| 0.10 | 0.125 | below ε, stays below — never breaches |
| 0.20 | 1.12 | 7× above ε at step 0 |
| 0.40 | 8.01 | already breached |
| 0.60 | 20.52 | 137× above ε at step 0 |

There is no crossing in either regime — a run starts inside and stays inside, or starts
outside and repairs slowly inward. `sigma_drift` in the config reads as though it were
meant to supply ongoing perturbation, but it is only a sampling radius inside `_proactive`
and never perturbs θ. So `experiment_stability.py`'s check 3 ("does kappa_eff spike before
basin_kl exceeds epsilon?") has been computing a lead against `basin_fail_step = 0` at
every drift level it runs — the breach is at step 0, so the lead can never be positive.
This sim injects per-step drift to create the crossing the claim presupposes.

**2. The existing alarm cannot fail.** `experiment_stability.py` sets its threshold to
`df['kappa_eff'].quantile(0.9)` — the 90th percentile of the completed run. That is
look-ahead (it uses post-breach data to set a pre-breach threshold) and it is
self-fulfilling (exactly 10% of steps exceed it by construction, so an alarm always
exists). It has no null arm, so its false-alarm rate has never been measured. Every
criterion here is causal by contrast: thresholds are fixed on a warm-up window and applied
forward.

**3. The κ_eff branch of the phase classifier is effectively dead.** Observed κ_eff spans
0.0002–0.052 across all 20 runs. `_phase()` at `stability.py:441` calls "critical" when
`kappa > C_bound`, with `C_bound = 20.0` — roughly 400× the largest value ever observed.
The κ term can never fire at this model scale, so the phase classifier is driven entirely
by `basin_kl` and `dV_dt`. Either `C_bound` needs scale-relative calibration or the phase
logic should stop advertising a κ threshold it never reaches.

## What would settle it

The 25% result sits between the gates, so more of the same would not resolve it — the
sensible next steps change the measurement rather than repeat it:

1. **Smooth κ_eff before thresholding.** Much of the false-alarm rate looks like
   step-to-step noise rather than trend. A rolling median over 3–5 steps, pre-committed,
   is the cheapest thing that could move κ_eff's FP down to the baseline's.
2. **Scale-relative `C_bound`.** Calibrate against observed κ_eff rather than a fixed 20.0.
3. **Adversarial drift.** Isotropic Gaussian drift is the easy case and may simply not
   excite safety curvature. Drift along the top Hessian eigenvector is the case κ_eff is
   built to catch, and is the fairest remaining test of the claim.
