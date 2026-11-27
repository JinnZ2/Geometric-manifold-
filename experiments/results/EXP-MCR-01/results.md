# EXP-MCR-01 results: replace vs accumulate in recursive Gaussian and GMM refitting (run 2026-11-27; GMM accumulate arms and replace n = 1,000 are PILOT)

- Design: [`experiments/model-collapse-variance-suppression-and-recursive-training.md`](../../model-collapse-variance-suppression-and-recursive-training.md#exp-mcr-01--replace-vs-accumulate-in-recursive-gaussian-and-gmm-refitting-calibration).
- Code: [`run.py`](run.py) and [`mcr_gmm.py`](mcr_gmm.py). The 2-D GMM toy is vectorised over seeds; EXP-MCR-02 reuses it.
- The 1-D part calls `experiments/sketches/mcr_recursive_gaussian.py::simulate`, unchanged.
- Raw output: [`raw.json`](raw.json).

## Verdict

- **1-D code check: SUPPORTED.**
- **GMM part:**
  - **The falsifier, read literally: met.** This is a design scale error, not evidence for H0; see the design note.
  - **The prediction on its relative scale: SUPPORTED at n = 100.** It misses at accumulate n = 20, where the median is 66% of t = 0 against the predicted ≥ 80%.

Details:

- **1-D check.** At n = 20 and t ≤ 10, the mean variance is within 3.4% of ((n−1)/n)^(t+1) over 500 seeds, and within 0.9% over the supplementary 20,000 seeds. The prediction required within 5%.
  - The median decays faster than the mean, as H1 says: at n = 100, t = 200, the median variance is 0.016 and the mean 0.15.
  - Accumulate stays in a bounded band: mean variance 0.90, 0.97 and 1.00 at n = 20, 100 and 1,000, flat from t = 50 to t = 200.
- **GMM, replace.** At t = 200 the median tail mass is 0 for n = 20 and n = 100 (0% of t = 0; bootstrap CI [0, 0]). The prediction was < 50%.
- **GMM, accumulate.**
  - n = 100 (10 seeds): median tail mass at t = 200 is 120% of t = 0, CI [62%, 194%]. The prediction was ≥ 80%.
  - n = 20 (50 seeds): 66% [27%, 97%]. **Misses the ≥ 80% band.**
- **The falsifier.** It reads: "replace and accumulate tail mass differ by < 10 percentage points at t = 200 for n = 100". The absolute difference in median tail mass is 1.17 pp (1.165% vs 0%), so the falsifier is literally met. But the true tail mass is only 0.54%, so no outcome could differ by 10 absolute pp. On the scale the prediction itself uses (percent of t = 0), the regimes differ by about 120 pp.

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Python 3.14.8, numpy 2.5.5.
- CPU: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs (load average 20–30).
- Total runtime: 901 s on 4 worker processes. Per-arm times are in `raw.json`.
- Seeds:
  - The 1-D sketch uses seeds 0..499 (supplementary: 0..19,999).
  - GMM RNG streams are keyed on (n, temperature, mode, first seed, chunk size).
  - Tail-mass Monte Carlo uses its own stream, 20,000 samples per chain and checkpoint.
- True tail mass P(‖x‖ > 3σ₀) of the real distribution is 0.00539, from 2,000,000 samples. σ₀ = 1.414.

## Deviations (recorded in the docstrings of `run.py` and `mcr_gmm.py` before results)

1. **Real distribution.** Not specified by the design. Chosen before running:
   - K = 5 isotropic components with sd 1.
   - Means on a radius-1.5 circle.
   - Weights (0.40, 0.25, 0.15, 0.12, 0.08).
   - EM: max 50 iterations, warm start, and a component frozen at weight 0 if its responsibility falls below 1e-9.
2. **Scale (PILOT).**
   - Pre-registered: 500 seeds everywhere, T = 200.
   - Run:
     - GMM replace: 500 seeds at n = 20 and 100; 100 seeds at n = 1,000.
     - GMM accumulate: 50 seeds at n = 20; 10 at n = 100; 3 at n = 1,000, with T = 100.
     - 1-D: 500 seeds everywhere, as pre-registered.
   - A first launch at a larger scale (accumulate 200/50/5 seeds, replace 500 everywhere) was stopped before it wrote output, because its projected runtime on the shared machine was more than 4 hours.
3. **Supplementary 1-D check (not pre-registered).** 20,000 seeds at n = 20, t ≤ 10, run as a tighter code check.
4. **Bootstrap.** The design asks for per-generation 95% bootstrap bands. Raw per-seed values at the checkpoints t ∈ {0, 10, 25, 50, 100, 150, 200} are stored. The bootstrap CIs below are for the median ratio at the final generation (2,000 resamples, seed 0, computed from `raw.json`).

## Results (GMM)

| n | mode | seeds | T | median tail t=0 | median tail t=T | median ratio [95% CI] | seeds with tail_T < 0.5·tail_0 | mean components > 1/(10K) at T | median overall sd at T |
|---|---|---|---|---|---|---|---|---|---|
| 20 | replace | 500 | 200 | 0.0058 | 0 | 0.00 [0, 0] | 98% | 1.00 | 0.004 |
| 100 | replace | 500 | 200 | 0.0073 | 0 | 0.00 [0, 0] | 94% | 1.24 | 0.077 |
| 1,000 | replace | 100 | 200 | 0.0058 | 0.0010 | 0.17 [0.07, 0.86] | 55% | 3.67 | 1.255 |
| 20 | accumulate | 50 | 200 | 0.0051 | 0.0033 | 0.66 [0.27, 0.97] | 40% | 4.96 | 1.314 |
| 100 | accumulate | 10 | 200 | 0.0097 | 0.0117 | 1.20 [0.62, 1.94] | 0% | 4.90 | 1.428 |
| 1,000 | accumulate | 3 | 100 | 0.0060 | 0.0062 | 1.02 [0.70, 1.13] | 0% | 5.00 | 1.409 |

Under replace, the mean tail mass rises while the median falls to 0. At n = 100 the mean goes from 0.008 at t = 0 to 0.031 at t = 200. A minority of chains spread out while most collapse. Medians and means tell different stories here, and the prediction is about medians.

## Dated design note (2026-11-27, added here, not in the design)

The falsifier "differ by < 10 percentage points" is on an absolute scale that cannot be reached: the true tail mass is 0.54%, so even total tail loss under replace is a 0.5–1 pp difference. It should have been written on the same relative scale as the prediction, for example "replace median tail ratio and accumulate median tail ratio at t = 200 differ by < 10 pp of the t = 0 value". The run reports both readings. The literal one is met, and the relative one is far from met.

## What this establishes and what it does not

- It establishes:
  - The 1-D recursion matches its derived formula; the sketch has no bug at this check.
  - In this 2-D GMM toy, replace drives the median tail mass to 0 within 200 generations at n ≤ 100. Accumulate holds it near or above its t = 0 value at n = 100 and n = 1,000, though at pilot seed counts.
  - The toy is calibrated in the sense MCR-02 and MCR-03 need.
- It does not establish:
  - The accumulate n = 20 shortfall (66% against ≥ 80%) beyond pilot noise. The CI is wide, [27%, 97%].
  - Anything about language models.
- Full-scale accumulate runs (500 seeds, T = 200 at n = 1,000) are UNCHECKED. On an unloaded 4-core machine, they would need roughly 1–2 hours of compute, from the n·T² cost measured here.
