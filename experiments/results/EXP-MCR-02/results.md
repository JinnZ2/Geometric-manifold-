# EXP-MCR-02 results: do standard EWS lead a GMM collapse event, against a matched null? (run 2026-11-27, PILOT)

- Design: [`experiments/model-collapse-variance-suppression-and-recursive-training.md`](../../model-collapse-variance-suppression-and-recursive-training.md#exp-mcr-02--do-standard-ews-lead-a-collapse-event-against-a-matched-null).
- Code: [`run.py`](run.py). The GMM toy is shared with EXP-MCR-01 ([`../EXP-MCR-01/mcr_gmm.py`](../EXP-MCR-01/mcr_gmm.py)). The EWS functions are `experiments/sketches/mcr_recursive_gaussian.py::ews`, used unchanged.
- Raw output: [`raw.json`](raw.json).

**Verdict: INCONCLUSIVE (underpowered PILOT, and the pre-registered event is degenerate in this toy).**

- **Falsifier.** AUC ≥ 0.7 for both τ statistics at temperature 1.0 in ≥ 2 of 3 n values. It is not met: no temperature-1.0 cell has both AUCs ≥ 0.7.
- **Prediction "AUC in [0.4, 0.6] at temperature 1.0".** Most cells sit there, but bootstrap CIs are 0.3–0.6 wide. A few small-sample cells fall outside:
  - sd, w = 10, tail event, n = 100: τ_var AUC 0.84 [0.65, 0.98].
  - min-weight, w = 10, mode-loss event, n = 100: τ_var AUC 0.80 [0.63, 0.92].
  Neither is matched by τ_ac1. Both come from 11–12 event runs, and they are 2 of the 100 AUCs in the table, with no correction for multiplicity.
- **Prediction "at temperature 0.9 rolling-variance τ is negative and AC1 τ is not informative".** Median event-window τ_var of σ_t is negative in 4 of 5 temperature-0.9 cells with ≥ 5 event runs. The null windows' τ_var is negative too, so it is not specific. The τ_ac1 AUCs are 0.38–0.56, which is consistent with "not informative".

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Python 3.14.8, numpy 2.5.5, scipy 1.18.1.
- CPU: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs (load average 20–30).
- Simulation wall time: 546 s on 2 worker processes.
- Seeds: replace runs use seeds 0..S−1 in chunks of 25, accumulate runs seeds 0–2. The RNG streams are keyed on (20261127, n, temperature, mode, first seed, chunk size). The bootstrap RNG is seed 7.

## Deviations (recorded in the `run.py` docstring before running)

1. **PILOT scale.**
   - Pre-registered: 500 seeds.
   - Run: replace 100 seeds at n = 50 and 100, 50 seeds at n = 300; accumulate 3 seeds at n = 50 and 100, none at n = 300; T = 200.
   - A first launch at 200/200/100 replace and 10/10 accumulate seeds was stopped before it wrote output, because its projected runtime on the loaded machine was over 1.5 h.
2. **Real distribution.** Not specified by the design. It is fixed in `mcr_gmm.py`: K = 5, means on a radius-1.5 circle, sd 1, weights (0.40, 0.25, 0.15, 0.12, 0.08).
3. **EWS segment.** "Window: the w generations before E" is implemented as the standard EWS: rolling variance and AC1 with window w on the segment [0, E), with Kendall τ over the rolling values. A run enters a w-cell only if it has ≥ 10 rolling values (E ≥ w + 9).
4. **Null.** The design says: replace runs with no event within the horizon, plus accumulate runs, at matched generation index. Implemented as the same [0, E) segment of every accumulate run, and of every replace run whose own event (if any) comes after E. Without this extension the null pool would be almost empty, since 90–100% of replace runs have an event.
5. **Tail-mass event (E2).** Tail mass is estimated each generation with common random numbers: 2,000 fixed draws per component. The series is therefore a smooth function of the fitted parameters.
6. **AUC.** AUC is over matched (event, null) pairs, with ties counted as ½. The bootstrap resamples event runs and null runs, 1,000 times. CIs are reported only where there are ≥ 5 event runs.

## Results

All (n, temperature, event, statistic, w) cells with at least one usable event run are listed below. "runs used" is the number of event runs with E ≥ w + 9. Cells not shown had no usable event run.

| n | temp | event | replace runs | event frac | E p10/p50/p90 | stat | w | runs used | AUC var [CI] | AUC ac1 [CI] | median τ_var event / null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 1.0 | E1_minw | 100 | 1.00 | 1/4/13 | sd | 10 | 3 | 0.67 (<5 runs) | 0.42 (<5 runs) | -0.33 / -0.64 |
| 50 | 1.0 | E1_minw | 100 | 1.00 | 1/4/13 | sd | 20 | 1 | 0.33 (<5 runs) | 0.33 (<5 runs) | -0.89 / -0.80 |
| 50 | 1.0 | E1_minw | 100 | 1.00 | 1/4/13 | min_w | 10 | 3 | 0.83 (<5 runs) | 0.25 (<5 runs) | -0.09 / -0.41 |
| 50 | 1.0 | E1_minw | 100 | 1.00 | 1/4/13 | min_w | 20 | 1 | 1.00 (<5 runs) | 0.17 (<5 runs) | -0.54 / -0.98 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | sd | 10 | 9 | 0.63 [0.38, 0.91] | 0.52 [0.25, 0.80] | -0.34 / -0.58 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | sd | 20 | 6 | 0.64 [0.38, 1.00] | 0.48 [0.14, 0.78] | -0.43 / -0.69 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | sd | 40 | 2 | 0.86 (<5 runs) | 0.43 (<5 runs) | -0.40 / -0.70 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | min_w | 10 | 9 | 0.63 [0.42, 0.84] | 0.40 [0.17, 0.60] | -0.42 / -0.61 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | min_w | 20 | 6 | 0.83 [0.68, 1.00] | 0.55 [0.16, 0.88] | -0.39 / -0.75 |
| 50 | 1.0 | E2_tail | 100 | 1.00 | 1/3/13 | min_w | 40 | 2 | 0.93 (<5 runs) | 0.14 (<5 runs) | -0.45 / -0.81 |
| 100 | 1.0 | E1_minw | 100 | 1.00 | 2/6/19 | sd | 10 | 12 | 0.49 [0.28, 0.72] | 0.36 [0.16, 0.58] | -0.38 / -0.60 |
| 100 | 1.0 | E1_minw | 100 | 1.00 | 2/6/19 | sd | 20 | 3 | 0.58 (<5 runs) | 0.50 (<5 runs) | -0.82 / -0.85 |
| 100 | 1.0 | E1_minw | 100 | 1.00 | 2/6/19 | min_w | 10 | 12 | 0.80 [0.63, 0.92] | 0.62 [0.38, 0.80] | -0.13 / -0.67 |
| 100 | 1.0 | E1_minw | 100 | 1.00 | 2/6/19 | min_w | 20 | 3 | 0.75 (<5 runs) | 0.50 (<5 runs) | -0.67 / -0.88 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | sd | 10 | 11 | 0.84 [0.65, 0.98] | 0.63 [0.42, 0.81] | 0.43 / -0.18 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | sd | 20 | 7 | 0.66 [0.29, 1.00] | 0.51 [0.19, 0.85] | 0.20 / -0.02 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | sd | 40 | 4 | 0.50 (<5 runs) | 0.21 (<5 runs) | 0.08 / -0.55 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | min_w | 10 | 11 | 0.40 [0.17, 0.70] | 0.40 [0.20, 0.61] | -0.46 / -0.49 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | min_w | 20 | 7 | 0.46 [0.14, 0.80] | 0.43 [0.16, 0.72] | -0.62 / -0.54 |
| 100 | 1.0 | E2_tail | 100 | 0.99 | 1/4/19 | min_w | 40 | 4 | 0.18 (<5 runs) | 0.07 (<5 runs) | -0.90 / -0.70 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | sd | 10 | 20 | 0.49 [0.30, 0.68] | 0.45 [0.28, 0.64] | 0.10 / 0.12 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | sd | 20 | 15 | 0.61 [0.38, 0.82] | 0.57 [0.33, 0.79] | 0.05 / 0.21 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | sd | 40 | 6 | 0.50 [0.00, 1.00] | 0.43 [0.00, 1.00] | -0.68 / -0.87 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | min_w | 10 | 20 | 0.47 [0.28, 0.67] | 0.51 [0.31, 0.69] | -0.23 / -0.20 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | min_w | 20 | 15 | 0.57 [0.32, 0.80] | 0.47 [0.23, 0.72] | -0.31 / -0.33 |
| 300 | 1.0 | E1_minw | 50 | 1.00 | 3/14/49 | min_w | 40 | 6 | 0.40 [0.00, 1.00] | 0.53 [0.00, 1.00] | -0.54 / -0.15 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | sd | 10 | 21 | 0.54 [0.36, 0.71] | 0.56 [0.42, 0.70] | 0.11 / 0.15 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | sd | 20 | 14 | 0.46 [0.25, 0.63] | 0.53 [0.33, 0.69] | -0.13 / 0.09 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | sd | 40 | 10 | 0.40 [0.21, 0.63] | 0.47 [0.29, 0.66] | -0.18 / 0.22 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | min_w | 10 | 21 | 0.46 [0.30, 0.63] | 0.46 [0.31, 0.63] | -0.37 / -0.39 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | min_w | 20 | 14 | 0.61 [0.44, 0.76] | 0.55 [0.37, 0.75] | -0.45 / -0.49 |
| 300 | 1.0 | E2_tail | 50 | 0.90 | 2/11/108 | min_w | 40 | 10 | 0.58 [0.37, 0.79] | 0.55 [0.31, 0.79] | -0.46 / -0.59 |
| 50 | 0.9 | E1_minw | 100 | 1.00 | 1/4/14 | sd | 10 | 2 | 1.00 (<5 runs) | 0.50 (<5 runs) | 0.35 / -0.59 |
| 50 | 0.9 | E1_minw | 100 | 1.00 | 1/4/14 | min_w | 10 | 2 | 1.00 (<5 runs) | 0.67 (<5 runs) | -0.29 / -0.89 |
| 50 | 0.9 | E2_tail | 100 | 1.00 | 1/1/4 | sd | 10 | 2 | 0.00 (<5 runs) | 1.00 (<5 runs) | -0.13 / -0.09 |
| 50 | 0.9 | E2_tail | 100 | 1.00 | 1/1/4 | sd | 20 | 1 | n/a (<5 runs) | n/a (<5 runs) | 0.77 / n/a |
| 50 | 0.9 | E2_tail | 100 | 1.00 | 1/1/4 | min_w | 10 | 2 | 0.00 (<5 runs) | 0.50 (<5 runs) | -0.54 / -0.62 |
| 50 | 0.9 | E2_tail | 100 | 1.00 | 1/1/4 | min_w | 20 | 1 | n/a (<5 runs) | n/a (<5 runs) | -0.54 / n/a |
| 100 | 0.9 | E1_minw | 100 | 1.00 | 2/6/20 | sd | 10 | 14 | 0.57 [0.38, 0.81] | 0.44 [0.23, 0.65] | -0.34 / -0.52 |
| 100 | 0.9 | E1_minw | 100 | 1.00 | 2/6/20 | sd | 20 | 5 | 0.80 [0.58, 1.00] | 0.56 [0.17, 0.88] | 0.04 / -0.78 |
| 100 | 0.9 | E1_minw | 100 | 1.00 | 2/6/20 | min_w | 10 | 14 | 0.58 [0.38, 0.81] | 0.53 [0.34, 0.74] | 0.06 / -0.02 |
| 100 | 0.9 | E1_minw | 100 | 1.00 | 2/6/20 | min_w | 20 | 5 | 0.68 [0.45, 1.00] | 0.44 [0.14, 0.79] | -0.15 / -0.78 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | sd | 10 | 21 | 0.46 [0.25, 0.66] | 0.44 [0.25, 0.62] | -0.37 / -0.38 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | sd | 20 | 18 | 0.52 [0.29, 0.74] | 0.46 [0.26, 0.68] | -0.32 / -0.44 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | sd | 40 | 10 | 0.30 [0.04, 0.58] | 0.38 [0.10, 0.67] | -0.77 / -0.09 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | min_w | 10 | 21 | 0.39 [0.25, 0.55] | 0.43 [0.26, 0.62] | -0.03 / 0.04 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | min_w | 20 | 18 | 0.44 [0.26, 0.63] | 0.40 [0.22, 0.61] | 0.07 / 0.13 |
| 300 | 0.9 | E1_minw | 50 | 1.00 | 4/13/60 | min_w | 40 | 10 | 0.49 [0.17, 0.81] | 0.47 [0.14, 0.82] | 0.46 / 0.47 |

What the table shows:
- **Events come early.**
  - The mode-loss event (a component weight below 0.02) fires at a median of generation 4 (n = 50), 6–6.5 (n = 100) and 13–13.5 (n = 300), in 100% of replace runs at both temperatures.
  - The tail event at temperature 0.9 fires at a median of generation 1–2, because the sampler shrinks the tails in the first step.
  - As a result, only 0–21 runs per cell have enough pre-event history for an EWS window, and the w = 40 cells are mostly empty.
- **The null pool.** It is dominated by replace runs whose event comes later, plus 3 accumulate runs. None of the accumulate runs had a mode-loss event at temperature 1.0.

## Dated design note (2026-11-27, added here, not in the design)

The pre-registered event (any GMM weight < 1/(10K)) is close to degenerate for a K = 5 fit to strongly overlapping components at n ≤ 300. Weights of overlapping components are poorly identified, and EM moves weight between them within a few generations. So E mostly measures label and weight wandering, not tail collapse. Any of the following would give the EWS test room to work:
- well-separated components;
- a persistence requirement (weight below the floor for k consecutive generations);
- an event defined on the fitted density rather than the parameters.

The choice of real distribution was not part of the design, and that gap is what made the event early.

## What this establishes and what it does not

- It establishes that, in this toy at pilot scale, none of the four τ statistics shows the AUC ≥ 0.7 pattern H1 needs at temperature 1.0 in more than one n value. It also establishes that the pre-registered event definition leaves too little pre-event history for a fair test.
- It does not establish H0 (AUC ≤ 0.6). The CIs are too wide, and several cells have fewer than 5 runs.
- It does not establish anything about EWS in the repo's basin-repair dynamics, or in real recursive training.
- The full-scale run (500 seeds) with a less degenerate event definition is UNCHECKED.
