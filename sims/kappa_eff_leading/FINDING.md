# FINDING — IP-18: κ_eff does not earn its cost, and the adversarial case is worse

Run 2026-08-13T2033Z · 5 seeds × 3 drift rates × **2 drift modes** (30 drift runs, 27
usable) + 5 null runs × 60 steps · 7 criteria × 3 signals · ~71 s · graded by `run.py`
against `REFUTE.md`. Null arm: 0/5 breached, so the null is valid.

## Verdict: REFUTED

An earlier run with the isotropic arm alone returned INCONCLUSIVE — 25% of cells, sitting
between the 20% and 80% gates. Adding the adversarial arm (drift along the top eigenvector
of the safety Hessian, the case κ_eff exists to catch) moved it to REFUTED: under the only
criterion that survived the null arm, κ_eff both led the breach and beat the free baseline
in **19% of cells**, at or below the 20% refutation gate.

**The adversarial case is the worse one for κ_eff, not the better one** — 13% of 15 cells
under adversarial drift against 25% of 12 under isotropic, on that same criterion.

## The criterion sweep is the result

| criterion | null FP (κ_eff) | null FP (θ-dist) | viable | supports A (all) | isotropic | adversarial |
|---|---|---|---|---|---|---|
| ratio_1.5 | **0.80** | 0.00 | no | 37% | 33% | 40% |
| ratio_2.0 | **0.80** | 0.00 | no | 33% | 25% | 40% |
| ratio_3.0 | 0.20 | 0.00 | **yes** | **19%** | 25% | 13% |
| z_2.0 | 0.60 | 0.80 | no | 0% | 0% | 0% |
| z_3.0 | 0.60 | 0.80 | no | 7% | 17% | 0% |
| tau_w8_0.3 | **1.00** | 1.00 | no | 0% | 0% | 0% |
| tau_w8_0.5 | 0.40 | 1.00 | no | 0% | 0% | 0% |

Read the `tau_w8_0.3` row against the `ratio_3.0` row. The τ-trend criterion gives κ_eff
its **best raw leading performance** in the table — it alarms before the breach in 63% of
cells — and it also fires on **every single quiet run** (FP = 1.00). A study that reported
lead times without a null arm would have picked that row and called it strong evidence for
Theory A. With the null arm, it is worthless.

That is precisely the failure `HARNESS.md` retrofit queue item 3 records for the previous
version of this test — "verdict flipped on criterion choice" — reproduced here in a single
table instead of across two write-ups.

## What this says about κ_eff

**κ_eff alarms on quiet systems.** Under the two mild ratio criteria it fires on 4 of 5
runs where nothing is drifting and no breach ever occurs. Only a 3× threshold over the
warm-up median holds the false-alarm rate at the gate, and at that strictness it catches
the breach in advance 19% of the time.

**The free baseline is better behaved.** ‖θ − θ_ref‖ costs nothing — no Hessian-vector
product — and holds FP = 0.00 under all three ratio criteria, against κ_eff's
0.80 / 0.80 / 0.20. κ_eff is paying a Hessian every step to be noisier than a norm.

**κ_eff does respond to adversarial drift — erratically.** Its peak reaches 0.37 and 0.27
on some seeds, against a ceiling near 0.05 under isotropic drift. But at the same σ and
mode, other seeds peak at 0.0105, 0.0176, 0.0199 — an order of magnitude apart under
identical conditions. That spread is the mechanism behind the refutation: the signal is
real and it is too seed-dependent for any fixed causal threshold to catch without also
firing on quiet runs. The generous criteria that do catch it (ratio_1.5/2.0, where
adversarial reaches 40%) are exactly the ones with an 0.80 false-alarm rate.

**What is not refuted.** The claim tested is operational: does this scalar, under a causal
threshold, fire before the basin breach more usefully than a free alternative? At toy
scale, under both isotropic and adversarial drift, it does not. The underlying geometry is
untouched by this.

## A sign convention that inverts the result

Worth recording because it nearly produced a false finding. Power iteration returns an
arbitrary sign, so an unoriented top eigenvector alternates direction between steps and
the drift random-walks instead of accumulating. With that bug the model never leaves the
basin at all (KL flat at ≈0.06 for 40 steps), which reads naturally as *"the repair loop
handles adversarial drift better than isotropic"* — a clean, plausible, entirely false
conclusion.

Orienting the direction to increase the safety KL (`sign⟨v, ∇KL⟩`) makes adversarial drift
breach at step 25 against isotropic's 34–46 at the same σ, which is the expected ordering.
`adversarial_direction()` carries the fix and the reason.

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
0.0001–0.37 across all 35 runs, and stays under 0.05 in every isotropic run.
`_phase()` at `stability.py:441` calls "critical" when `kappa > C_bound`, with
`C_bound = 20.0` — roughly 50× the largest value ever observed and 400× the isotropic
ceiling. The κ term can never fire at this model scale, so the phase classifier is driven
entirely by `basin_kl` and `dV_dt`. Either `C_bound` needs scale-relative calibration or
the phase logic should stop advertising a κ threshold it never reaches.

## What would settle it

The adversarial arm was the fairest remaining test of the drift direction, and it has now
been run. What is left is about the signal's noise:

1. **Smooth κ_eff before thresholding.** The order-of-magnitude seed-to-seed spread in
   peak κ_eff, at identical σ and mode, is the direct cause of the refutation. A rolling
   median over 3–5 steps, pre-committed, is the cheapest change that could bring κ_eff's
   false-alarm rate down toward the baseline's. If it does not, the signal is noise at
   this scale.
2. **Scale-relative `C_bound`**, calibrated against observed κ_eff rather than a fixed
   20.0 that nothing approaches.
3. **Larger models.** The Rayleigh quotient of an ~8k-parameter safety Hessian may be
   dominated by sampling noise. This result does not transfer upward on its own.
