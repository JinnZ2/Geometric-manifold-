# RESULTS — parameter-drift spectral signature, Tier 1

Run `2026-09-05T0301Z` · 5 seeds × 2 conditions × 1200 steps · 36.3 s · config sha256 `dddef55826e4` · emitted by `report.py`.

## 0. Licence (selftest S4) — read this first

**S4 FAILS.** The reference channel (RankMe of hidden activations, the channel where Li et al. publish the shape) reproduces the THREE-phase shape on **0/5** skewed seeds. It reproduces the first TWO phases — collapse then expansion — on **5/5** skewed seeds against **5/5** control seeds (the skew does NOT discriminate). The third phase, compression, appears on no seed and appeared in no setting of the calibration sweep recorded in `config.json`.

Added after the data was seen, and labelled so: the detector's margin is relative to each series' own range, so a shallow dip passes it. What separates the conditions is the DEPTH of the first leg in RankMe units — skewed 2.49–3.23, control 0.73–1.08, disjoint. The detector is left as declared; the depth is a second readout beside it, not a repair to the first.

Per the order (§6): *if the synthetic setup cannot reproduce the published shape in the channel where it is published, the parameter-space channel means nothing.* **Section 4 (LOCKED / ANTI-PHASE / DECOUPLED) is therefore NOT RUN and no alignment verdict is emitted.** What follows under §2–§3 are instrument readings on the drift axes, reported as observations and licensed by nothing.

What this negative is and is not: the generator (a 16→6→24 tanh MLP on Zipf-skewed prototypes, SGD with L2) does not consolidate after its expansion at any weight decay, learning-rate schedule or probe distribution tried. That is a statement about this generator. It is not evidence about whether real networks compress, and it is not evidence about basin repair.

## 1. Reference channel (REP) per seed

RankMe of the centered covariance of hidden activations over a fixed probe set, uniform over classes, 3 per class. Smoothed extrema; legs C=collapse E=expansion K=compression (Li et al.'s names, licensed on this channel only); each must clear 10% of the smoothed range.

**skewed** (skew 1.5)

| seed | start | trough @step | peak @step | end | depth | legs | alpha-ReQ start→end (min, max) | final loss / probe acc |
|---|---|---|---|---|---|---|---|---|
| s0 | 4.98 | 2.49 @15 | 5.96 @1200 | 5.96 | 2.49 | CE· two, nonmonotone  2.01→0.35 (0.35, 5.59) | 0.16 / 0.81 |
| s1 | 5.03 | 1.80 @14 | 5.89 @1200 | 5.89 | 3.23 | CE· two, nonmonotone  1.50→0.58 (0.58, 5.67) | 0.16 / 0.64 |
| s2 | 4.74 | 2.09 @14 | 5.82 @1200 | 5.82 | 2.65 | CE· two, nonmonotone  2.06→0.77 (0.77, 5.14) | 0.18 / 0.69 |
| s3 | 5.27 | 2.74 @15 | 5.86 @730 | 5.86 | 2.53 | CE· two, nonmonotone  1.71→0.70 (0.69, 4.61) | 0.25 / 0.68 |
| s4 | 4.61 | 1.89 @13 | 5.90 @1190 | 5.90 | 2.72 | CE· two, nonmonotone  2.47→0.59 (0.59, 5.91) | 0.06 / 0.75 |

**control** (skew 0.0)

| seed | start | trough @step | peak @step | end | depth | legs | alpha-ReQ start→end (min, max) | final loss / probe acc |
|---|---|---|---|---|---|---|---|---|
| s0 | 5.07 | 4.34 @40 | 5.82 @1200 | 5.82 | 0.73 | CE· two, nonmonotone  2.01→0.76 (0.76, 2.56) | 0.18 / 1.00 |
| s1 | 5.44 | 4.36 @47 | 5.82 @1190 | 5.82 | 1.08 | CE· two, nonmonotone  1.50→0.76 (0.76, 3.10) | 0.20 / 1.00 |
| s2 | 4.99 | 4.04 @33 | 5.85 @1200 | 5.85 | 0.95 | CE· two, nonmonotone  2.06→0.70 (0.69, 3.04) | 0.22 / 1.00 |
| s3 | 5.27 | 4.33 @57 | 5.88 @1190 | 5.88 | 0.94 | CE· two, nonmonotone  1.71→0.64 (0.63, 2.91) | 0.22 / 1.00 |
| s4 | 4.84 | 3.93 @50 | 5.86 @1200 | 5.86 | 0.91 | CE· two, nonmonotone  2.47→0.65 (0.65, 3.37) | 0.21 / 1.00 |

## 2. Drift axes — per axis, per seed, not averaged

Legs on a drift axis are F=fall R=rise F=fall — the same detector as §1, with no phase name attached. Each axis is a different sample dimension manufactured for theta (see `drift.py` header). RankMe here counts independent DIRECTIONS in the samples that axis supplies; its ceiling is the sample count (A1: 10, A2-L1: 6, A2-L2: 7, A3: 5). Uncentered second moment for all drift axes ([CHOICE 1]).

### A1/TIME

**skewed**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 5.44 | 5.18 @17 | 9.37 @370 | 9.18 | 0.25 | ·R· none |
| s1 | 5.17 | 5.17 @10 | 9.30 @360 | 9.17 | 0.00 | ·R· none |
| s2 | 5.36 | 5.32 @12 | 9.38 @690 | 9.22 | 0.03 | ·R· none |
| s3 | 5.43 | 5.43 @10 | 9.47 @290 | 9.14 | 0.00 | ·R· none |
| s4 | 4.92 | 4.92 @10 | 9.38 @610 | 9.31 | 0.00 | ·R· none |

**control**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 9.46 | 8.49 @55 | 9.41 @1140 | 9.23 | 0.97 | FRF three, nonmonotone |
| s1 | 9.39 | 8.51 @57 | 9.52 @480 | 9.21 | 0.89 | FRF three, nonmonotone |
| s2 | 9.41 | 8.50 @56 | 9.55 @580 | 9.35 | 0.91 | FRF three, nonmonotone |
| s3 | 9.32 | 8.41 @57 | 9.51 @1140 | 9.40 | 0.92 | FRF three, nonmonotone |
| s4 | 9.39 | 8.60 @52 | 9.41 @290 | 9.39 | 0.78 | FR· two, nonmonotone |

### A2/UNIT-L1

**skewed**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 3.82 | 2.00 @13 | 4.61 @490 | 4.08 | 1.81 | FRF three, nonmonotone |
| s1 | 2.74 | 1.45 @10 | 4.53 @380 | 3.94 | 1.29 | FRF three, nonmonotone |
| s2 | 2.95 | 1.62 @8 | 4.75 @240 | 4.36 | 1.33 | FRF three, nonmonotone |
| s3 | 3.27 | 1.95 @10 | 4.60 @660 | 4.25 | 1.32 | FRF three, nonmonotone |
| s4 | 3.43 | 1.51 @11 | 4.78 @460 | 3.93 | 1.93 | FRF three, nonmonotone |

**control**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 5.19 | 4.08 @1150 | 4.54 @1190 | 4.51 | 1.11 | FR· two, nonmonotone |
| s1 | 5.42 | 3.60 @810 | 4.74 @980 | 4.09 | 1.82 | FRF three, nonmonotone |
| s2 | 5.19 | 3.87 @30 | 5.31 @130 | 3.92 | 1.32 | FRF three, nonmonotone |
| s3 | 5.02 | 4.16 @750 | 4.70 @1030 | 4.52 | 0.87 | FRF three, nonmonotone |
| s4 | 5.33 | 3.93 @44 | 5.22 @130 | 4.50 | 1.40 | FRF three, nonmonotone |

### A2/UNIT-L2

**skewed**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 1.39 | 1.39 @1 | 6.16 @1190 | 6.12 | 0.00 | ·R· none |
| s1 | 1.25 | 1.25 @1 | 5.56 @1110 | 5.38 | 0.00 | ·R· none |
| s2 | 1.43 | 1.43 @1 | 5.58 @930 | 5.05 | 0.00 | ·RF none, nonmonotone |
| s3 | 1.33 | 1.33 @2 | 5.73 @1190 | 5.65 | 0.00 | ·R· none |
| s4 | 1.22 | 1.22 @1 | 5.58 @1200 | 5.58 | 0.00 | ·R· none |

**control**

| seed | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| s0 | 1.58 | 1.58 @1 | 6.32 @940 | 6.06 | 0.00 | ·R· none |
| s1 | 1.59 | 1.59 @1 | 6.34 @350 | 5.98 | 0.00 | ·R· none, nonmonotone |
| s2 | 1.65 | 1.64 @2 | 6.36 @760 | 6.09 | 0.01 | ·R· none |
| s3 | 1.57 | 1.57 @1 | 6.33 @1140 | 6.17 | 0.00 | ·R· none |
| s4 | 1.55 | 1.51 @6 | 6.28 @300 | 5.85 | 0.04 | ·R· none, nonmonotone |

### A3/SEED-raw (one series per condition, across 5 seeds)

| condition | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| skewed | 2.52 | 2.41 @5 | 4.96 @1200 | 4.96 | 0.11 | ·R· none |
| control | 4.85 | 4.80 @7 | 4.99 @170 | 4.99 | 0.05 | FR· two, nonmonotone |

### A3/SEED-aligned (one series per condition, across 5 seeds)

| condition | start | trough @step | peak @step | end | depth | legs |
|---|---|---|---|---|---|---|
| skewed | 2.54 | 2.47 @5 | 4.92 @1180 | 4.92 | 0.07 | ·R· none |
| control | 4.85 | 4.80 @7 | 4.98 @100 | 4.96 | 0.05 | FRF three, nonmonotone |

### Do the axes agree?

Question put to each axis on the skewed condition: does the smoothed RankMe series have any interior extremum clearing the margin on both sides ("nonmonotone")? Majority over seeds where there are seeds.

| axis | skewed nonmonotone | skewed two-phase | control nonmonotone | control two-phase | vote |
|---|---|---|---|---|---|
| A1/TIME | 0/5 | 0/5 | 5/5 | 5/5 | none |
| A2/UNIT-L1 | 5/5 | 5/5 | 5/5 | 5/5 | structure |
| A2/UNIT-L2 | 1/5 | 0/5 | 2/5 | 0/5 | none |
| A3/SEED-raw | 0/1 | 0/1 | 1/1 | 1/1 | none |
| A3/SEED-aligned | 0/1 | 0/1 | 1/1 | 1/1 | none |

**The axes DISAGREE.** Per the order §1 that disagreement is the primary result on the drift side and per §8 it opens BRANCH ENTRY 02 against the rule "parameter drift has a spectrum" — see `BRANCH.md`. It is recorded as an observation: with S4 failed, nothing here is licensed against the reference channel.

## 3. Repair-flag overlay

The repo's two flag rules (`INVENTORY.md` §2, §6), reference := theta_0, thresholds 0.5 (hardcoded in the controller) and 0.4 (config). On a run from init both are monotone in distance and fire once; the overlay places that step against the reference channel's turning points.

| condition | seed | param flag | policy flag |
|---|---|---|---|
| skewed | s0 | step 6: before the REP trough at 15 | never fires |
| skewed | s1 | step 5: before the REP trough at 14 | never fires |
| skewed | s2 | step 6: before the REP trough at 14 | never fires |
| skewed | s3 | step 5: before the REP trough at 15 | never fires |
| skewed | s4 | step 5: before the REP trough at 13 | never fires |
| control | s0 | step 38: before the REP trough at 40 | step 580: between REP trough 40 and peak 1200 |
| control | s1 | step 44: before the REP trough at 47 | step 600: between REP trough 47 and peak 1190 |
| control | s2 | step 37: between REP trough 33 and peak 1200 | step 560: between REP trough 33 and peak 1200 |
| control | s3 | step 44: before the REP trough at 57 | step 640: between REP trough 57 and peak 1190 |
| control | s4 | step 45: before the REP trough at 50 | step 570: between REP trough 50 and peak 1200 |

Placement counts over 10 runs — parameter flag: before REP trough 9, between trough and peak 1, after peak 0, never 0. Policy flag: before trough 0, between 5, after peak 0, never 5.

Reading: at the parameter flag's first crossing `||theta - theta_0||` is 0.70–0.75 and the curvature term `lambda_curv * curv` is 0.00–0.00; the flag fires when the distance from init exceeds ln 2 ≈ 0.69, which is a property of the flag's scale against this model's step size, not of any spectral transition; the policy flag depends on how far the output distribution has moved from init. Each fires exactly once on a run from init, so "clustered at a transition" versus "uniform across phases" is not a distinction this overlay can draw. That is a limit of the flag, recorded in INVENTORY.md §2, not of the run.

## 4. Alignment — NOT RUN

S4 failed (§0). `align.align()` exists and is exercised by the selftest on constructed series; it was not called on this run's data and no LOCKED / ANTI-PHASE / DECOUPLED verdict is emitted.

## 5. Declared choices

- [CHOICE 1] drift axes use the uncentered second moment; REP is centered (`drift.py`).
- [CHOICE 2] probe set uniform over classes (`config.json`); the reference instrument samples its training distribution — tried in calibration, no change to the S4 outcome.
- [CHOICE 3] phase detector: smoothing half-width 3 samples, margin 0.1 of the smoothed range, three legs (`align.phases`).
- [CHOICE 4] alignment thresholds corr 0.5, lag tolerance 10% of steps — declared, unused on this run.
- A1 window 10: the only defined window in the repo: Monitor.detect_cost_spike / recent_trend, all 10 steps (INVENTORY.md §1).
- A3 alignment: one hidden-unit permutation per seed, fitted at the FINAL checkpoint against seed 0, applied at every step (`run_t1.a3_across_seeds`).
- Added after the data was seen: the `depth` column (start minus trough, series units). The detector itself was not changed.
- Generator calibration: `config.json` → `calibration_record`.

## 6. Not claimed

No phase names are attached to any drift axis. No statement about real networks. No statement about whether basin repair is doing the right thing. Tier 2 (Pythia checkpoints) is specified in the order and not attempted here.
