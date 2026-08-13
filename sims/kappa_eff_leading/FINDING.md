# FINDING — IP-18: κ_eff's problem is not noise, and smoothing proves it

Run 2026-08-13T2041Z · 5 seeds × 3 drift rates × 2 drift modes (30 drift runs, 27 usable)
+ 5 null runs × 60 steps · 7 criteria × **3 smoothing windows** = 21 combinations × 3
signals · ~68 s · graded by `run.py` against `REFUTE.md`. Null arm: 0/5 breached, so the
null is valid.

## Verdict: REFUTED, across three successive attempts to rescue it

| run | arms | verdict |
|---|---|---|
| 1 | isotropic drift only | INCONCLUSIVE — 25% of cells, between the gates |
| 2 | + adversarial drift | **REFUTED** — 19%, at the gate; adversarial *worse* than isotropic |
| 3 | + causal median smoothing (w ∈ {1,3,5}) | **REFUTED** — no window rescues it |

Each round added the fairest remaining test rather than repeating the last one, and each
was pre-registered before it ran. The escape hatches are now spent.

**The adversarial case is the worse one for κ_eff, not the better one** — 13% of 15 cells
under adversarial drift against 25% of 12 under isotropic, on the only criterion that
survived the null arm at window 1.

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

## The smoothing arm: C1 passed, C2 failed, and that combination is the answer

The previous round's defence was that κ_eff's peak varied by an order of magnitude across
seeds at identical σ and mode, so the failure looked like noise rather than absence of
signal. Smoothing is the cheapest fix for a noise problem. Both conditions were
pre-committed in `REFUTE.md` before the run.

**C1 — does smoothing reduce false alarms? Yes.** Median null FP for κ_eff falls
**0.60 → 0.40 → 0.40** across windows 1, 3, 5. Smoothing does exactly what it is for.

**C2 — does that rescue Theory A? No.** Four (criterion, window) combinations became
viable, and none supported Theory A at anywhere near the 80% gate:

| combination | null FP (κ_eff) | supports A | isotropic | adversarial |
|---|---|---|---|---|
| ratio_3.0 @ w=1 | 0.20 | 19% | 25% | 13% |
| ratio_3.0 @ w=3 | 0.20 | **4%** | 8% | 0% |
| ratio_2.0 @ w=5 | 0.20 | 11% | 17% | 7% |
| ratio_3.0 @ w=5 | 0.20 | 7% | 17% | 0% |

**The mechanism is a one-for-one trade.** Follow `ratio_2.0` down the windows: FP falls
0.80 → 0.40 → 0.20 while supports_A falls 33% → 22% → 11%. Smoothing buys quiet by giving
up exactly as much lead. And `ratio_3.0` is worse than that — its FP was already at the
gate and never moved, while smoothing cut supports_A from 19% to 4%. There, smoothing was
pure loss.

The trend criteria show the same thing from the other side: under smoothing, `tau_w8_0.3`
detects *more* (63% → 81% → 78%) and still fires on **every** quiet run at every window
(FP = 1.00). Smoothing makes it fire more reliably on everything, including nothing.

**What this settles.** If κ_eff carried a real leading signal buried in noise, smoothing
would separate them — false alarms would fall faster than lead. They fall together.
At this model scale, under both drift modes, there is no window at which κ_eff is a
usable leading indicator, and "it's just noisy" is no longer an available explanation.

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

Both cheap rescues have now been tried and both failed on their own pre-registered terms:
adversarial drift (the direction κ_eff exists to catch) made it worse, and smoothing traded
lead for quiet one-for-one. What remains is not a tweak to the measurement.

1. **Larger models.** The Rayleigh quotient of an ~8k-parameter safety Hessian may simply
   be dominated by sampling noise at this size. This is now the *only* live explanation
   that preserves the claim, and it is a real one — but it is also the expensive one, and
   nothing in this result transfers upward on its own. Anyone relying on κ_eff at scale
   should treat that as an open question rather than an established property.
2. **Scale-relative `C_bound`**, calibrated against observed κ_eff rather than a fixed
   20.0 that nothing approaches. This is worth doing regardless of the leading-indicator
   question, because the phase classifier currently advertises a threshold it never
   reaches.
3. **Retire or re-scope the claim in the code.** `stability.py` line 32 asserts "Spike in
   kappa_eff precedes behavioral collapse" as documentation, and
   `experiment_stability.py` check 3 reports a lead that its own construction cannot make
   positive. Whatever happens at larger scale, those two statements are not supported by
   anything measured here and should say so.
