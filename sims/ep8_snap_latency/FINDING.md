# FINDING — E-P8 snap latency: REFUTED, and informatively so

Run 2026-08-13T1942Z · 8 seeds × 3 ramp rates × 8 compression levels × 5 repeats = 24
cells, 960 trials · verdict graded by `run.py` against the criteria pre-committed in
`REFUTE.md` · numerics converged (max relative deviation at dt/4: 1.9e-06).

## Verdict: REFUTED — but only one half of the claim failed

E-P8 makes two assertions at once, and the measurement separates them cleanly.

| Half of the claim | Criterion | Measured | Outcome |
|---|---|---|---|
| The snap is a decodable ADC | RMSE ≤ 0.02 compression, better than mean-ε₀ baseline | **0.0016–0.0114**, mean ≈ 0.005, against a baseline of ≈ 0.05 | **passes in 24/24 cells**, ~10× better than baseline |
| It works via the fold law `t_snap ∝ ε₀^(−1/2)` | exponent in [−0.65, −0.35] | **+0.38 to +0.59** | **fails in 24/24 cells — wrong sign** |

The refutation condition is an OR (`exponent outside band OR decoder no better than
baseline`), so the verdict is REFUTED. But the reason matters more than the label: the
snap **does** report its initial condition, accurately and repeatably. It just does not do
it through the fold exponent.

The verdict does not depend on the regressor ambiguity flagged in `REFUTE.md`. The
protocol's stated regressor (log ε-distance) gives +0.38 to +0.59; the claim sentence's
regressor (log ε₀) gives −1.34 to −2.07. Both are outside [−0.65, −0.35], so both readings
refute.

## Why: the ramp geometry swamps the fold

Under a constant-rate ramp from ε₀, the snap time decomposes as

    t_snap ≈ (c_snap − ε₀)/rate  +  delay(rate)

The first term is linear in the distance to threshold and carries no fold physics at all —
it is just how long the ramp takes to arrive. The second term is where the saddle-node
lives, and it depends on the *rate*, not on ε₀. Since the linear term dominates, `t_snap`
is essentially affine in ε₀ — which is exactly why the decoder works so well, and exactly
why the exponent comes out positive instead of −1/2.

The sweep makes this visible directly. The fitted exponent drifts monotonically with ramp
rate:

| ramp_rate | exponent vs ε-distance |
|---|---|
| 0.005 | +0.585 |
| 0.01 | +0.482 |
| 0.02 | +0.377 |

A genuine fold law would be rate-independent. notes/15 anticipated precisely this failure
mode in its Risks section — "ramp-rate dependence (run a second rate; the law should
collapse when time is rescaled by rate)" — and it does not collapse: rescaling time by the
rate leaves a residual because the delay term scales as `rate^(−1/3)`, not as `rate^(−1)`.
The out-of-band exponent is therefore the second of the two causes the refutation
condition names: **rate-dominated dynamics**, not a misidentified bifurcation.

*(The `rate^(−1/3)` scaling is standard dynamic-bifurcation delay for a linearly swept
saddle-node. It is stated here as interpretation, not as something this sim measured —
labelled EXPLORATORY per `HARNESS.md` § 4. Measuring it would need delay isolated from
ramp time, which is the redesign below.)*

## Consequence for the physical protocol

The same shape as the E-P2 arc in notes/15, where v1 and v2 were refuted and v3 —
a two-arm differential design with one pre-committed checkpoint — survived.

**E-P8 as written measures ramp geometry.** On the printed instrument it will produce a
beautiful decoder and a meaningless exponent, and the beautiful decoder will look like
confirmation. Two changes are needed before the physical run is worth doing:

1. **Measure delay past the static threshold, not total ramp time.** Determine `c_snap`
   independently (E-P6's hysteresis sweep already does this), then record `t_snap` from
   the moment the ramp crosses it. The fold exponent lives in that residual; the ramp
   time is a constant offset to be subtracted, not the signal.
2. **Sweep the ramp rate and require collapse.** A rate-independent exponent is the actual
   evidence for a fold law. One rate cannot distinguish the fold from the ramp, which is
   the trap this sim just walked into on purpose.

If only the ADC property is wanted — "can the strut report its own initial compression?" —
then E-P8 already succeeds and should be re-registered as a decoder claim with the fold
language dropped. That would be a fair claim with a passing result, instead of an
unfair one with a failing result attached to a working instrument.

## Provenance

The refuted predecessor, `experiments/snap_information_sim.py`, is kept in the repo with a
header explaining that it never ran a snap at all. This sim replaces it. The difference
that matters is not better initial conditions — it is that the load actually ramps, so a
snap occurs and there is something to measure.
