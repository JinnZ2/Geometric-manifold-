# EXP-SDM-03 results: snapshot-delta monitoring in floating point (run 2026-11-27)

- Design: [`experiments/safety-drift-alignment-repair-and-parameter-space-monitoring.md`](../../safety-drift-alignment-repair-and-parameter-space-monitoring.md#exp-sdm-03--snapshot-delta-monitoring-in-floating-point).
- Code: [`run.py`](run.py). Raw output: [`raw.json`](raw.json).
- Positive-control rerun output: [`test1_rerun_results.json`](test1_rerun_results.json). The committed `docs/research/residual-tower/snapshot_delta_test1_results.json` is untouched.

## Verdict

**Error growth and the Kahan effect: SUPPORTED. Decision flips: INCONCLUSIVE (no near-threshold steps to test on). The falsifier is not met.**

- **Positive control reproduces.** The rerun of `snapshot_delta_test1.py --run` gives `test1_pass = true` and the same deterministic sha256, `b88473c4…5769`, as the committed results file. The rerun used Python 3.14.8; the committed file used 3.13.18.
- **float32 incremental error grows ∝ √T.** Measured relative to its own t = 0 offset, the log–log slope of RMS error vs T is 0.495, 95% CI [0.43, 0.54], over 20 seeds. The predicted heuristic slope is 0.5.
  - The absolute error grows more slowly: slope 0.08 [0.03, 0.12]. Up to T ≈ 10³ it is dominated by the t = 0 offset that the incremental arm inherits from its float32 starting value.
  - It is not independent of T, since the CI excludes 0.
- **Kahan compensation reduces it.**
  - Kahan drift is flat: slope 0.03 [−0.05, 0.14].
  - RMS drift at T = 10⁴: 7.1×10⁻⁷ with Kahan, against 4.6×10⁻⁵ without, 64 times smaller.
  - Kahan absolute error at T = 10⁴ is 2.58×10⁻⁵, against 4.82×10⁻⁵ uncompensated. That is the float32 full-recompute level (2.33×10⁻⁵), not below it.
- **Flips: no flip at any non-zero margin.** In the synthetic arm (200,020 monitored steps) and the ToyLLM arm (10 trajectories), there were no flips in the incremental arms at any margin ≥ 10⁻⁸ relative. The float32 full recompute had one, at margin 10⁻⁶–10⁻⁵.
  - The other flips counted (about 1 per run, in every arm, float64 included) all fall at the one step where the reference equals the threshold exactly. They are an artefact of the threshold choice, explained below.
  - Only 14 of 200,020 synthetic steps had a margin within 10⁻⁵ relative, the scale at which float32 error could flip a decision. The prediction "flips occur only near the threshold, and Kahan removes them at T = 10⁴" therefore has almost nothing to act on.

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Python 3.14.8, numpy 2.5.5, torch 2.15.0+cpu. BLAS single-threaded (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`).
- CPU: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs.
- Runtime: positive control 10 s, synthetic arm 53 s on 3 worker processes, ToyLLM arm 1.5 s.
- Seeds:
  - Synthetic: `default_rng([20261127, seed])`, seeds 0–19.
  - ToyLLM: environment seeds 0–4. Monitor weights use `default_rng([20261127, 7, seed])`.

## Deviations (recorded in the `run.py` docstring before results)

1. **Not written to `explore_harness.py`.** The design says to record runs in the `explore_harness.py` format beside Test 1. That was not done, because the run task restricts changes to `experiments/results/`, and the harness has its own sealing and independence rules. Raw output is in `raw.json` instead.
2. **Synthetic deltas.**
   - Shape: N = 2×10⁵, θ₀ ~ N(0, 0.05²). Each step changes 2,000 random entries (1%) by N(0, 10⁻⁶). T = 10⁴, with errors read at t ∈ {10, 10², 10³, 10⁴} as prefixes of one run per seed.
   - The structure is modelled on `make_snapshots`, but the function itself was not reused, because it generates integer snapshots.
   - Deltas are applied sparsely, as a snapshot-delta monitor would apply them: the per-step dot product is over the 2,000 changed entries. A first launch, with dense per-step deltas and multithreaded BLAS, ran at about 60 ms per step on the loaded machine. It was stopped before it wrote any output.
3. **ToyLLM arm.** The `main.py` parameter-repair trajectory, built as in EXP-FIT-01: default config 100 steps, adversarial config 200 steps, 5 seeds each, N = 3,152. Its T is capped by the trajectory length; T = 10⁴ is not reachable.
4. **The "drift" measure.** Drift is the error minus the arm's own t = 0 error. It was added after a 1,000-step smoke test on seed 99 (not one of the 20 seeds) showed the t = 0 offset dominating. It is reported beside the pre-registered absolute error, not instead of it.
5. **Flip threshold.** The threshold τ is the median of the reference series of each run, chosen before running. With an odd number of steps, the median is one of the reference values, so exactly one step per run has margin 0. That step produces the tie flips. A threshold not equal to any reference value (for example the midpoint between the two central values) would avoid this. Dated design note (2026-11-27): the design left the threshold open, and should fix it that way.

## Results: synthetic arm (20 seeds, N = 2×10⁵)

RMS over seeds, with the log–log slope over T ∈ {10, 10², 10³, 10⁴} and a bootstrap 95% CI over seeds (2,000 resamples).

| arm | abs error T=10 | T=10² | T=10³ | T=10⁴ | slope [CI] | drift T=10 | drift T=10⁴ | drift slope [CI] |
|---|---|---|---|---|---|---|---|---|
| float32 incremental | 2.59e-5 | 2.45e-5 | 2.64e-5 | 4.82e-5 | 0.08 [0.03, 0.12] | 1.48e-6 | 4.56e-5 | **0.50 [0.43, 0.54]** |
| float32 Kahan | 2.60e-5 | 2.57e-5 | 2.59e-5 | 2.58e-5 | 0.00 [−0.00, 0.00] | 6.1e-7 | 7.1e-7 | 0.03 [−0.05, 0.14] |
| float32 full recompute | 2.48e-5 | 2.36e-5 | 2.47e-5 | 2.33e-5 | −0.01 [−0.06, 0.04] | 8.1e-6 | 3.3e-5 | 0.21 [0.11, 0.29] |
| float64 incremental | 3.7e-14 | 5.8e-14 | 8.2e-14 | 1.2e-13 | 0.17 [0.10, 0.25] | (same) | (same) | 0.17 [0.10, 0.25] |

The monitor value is of order 10: the median |τ| is 14.4. So the float32 errors are about 2–3×10⁻⁶ relative.

Flips, pooled over 200,020 steps, by relative margin |ref − τ| / |τ|:

| Relative margin | Steps | Flips: float32 incremental | float64 incremental | Kahan | float32 full |
|---|---|---|---|---|---|
| Exactly 0 (the tie step) | 20 | 10 | 10 | 11 | 11 |
| 10⁻⁶ to 10⁻⁵ | 14 | 0 | 0 | 0 | 1 |
| 10⁻⁵ to 10⁻⁴ | 177 | 0 | 0 | 0 | 0 |
| ≥ 10⁻⁴ | the rest | 0 | 0 | 0 | 0 |

## Results: ToyLLM trajectories (N = 3,152)

| config | T | max abs error, float32 incremental | max abs error, Kahan | max abs error, float32 full | max abs error, float64 incremental | flips outside the tie step |
|---|---|---|---|---|---|---|
| default (5 seeds) | 100 | 3.4e-7 – 9.1e-6 | 1.6e-7 – 7.7e-6 | 5.8e-6 – 9.7e-6 | ≤ 2.3e-14 | 0 in every arm |
| adversarial (5 seeds) | 200 | 1.9e-6 – 3.3e-5 | 1.4e-6 – 1.2e-5 | 1.5e-5 – 2.5e-5 | ≤ 8.5e-14 | 0 in every arm |

## What this establishes and what it does not

- It establishes:
  - The integer bit-identity result of Test 1 reproduces on Python 3.14.8.
  - For a float32 linear monitor with random-sign sparse deltas, the incremental update drifts from the float64 reference as about √T. At T = 10⁴ that is about 2× the error of a float32 full recompute.
  - Kahan-compensated accumulation removes the T-dependence.
  - float64 incremental error stays at about 10⁻¹³.
  - At these monitor scales, no decision flipped at any margin above about 10⁻⁶ relative.
- It does not establish:
  - A flip rate as a function of margin. Too few steps fall near the threshold; a design with thresholds placed at controlled margins is needed.
  - The behaviour of non-random-sign (biased) deltas, where rounding error can grow as T rather than √T. This is UNCHECKED.
  - Anything about whether the monitor measures safety (SDM-01).
