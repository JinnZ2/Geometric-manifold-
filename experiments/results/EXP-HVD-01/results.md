# EXP-HVD-01 results: power and false-alarm rate of the engine's hidden-variable trigger on planted data

Run 2026-11-27. Design: [`../../hidden-variable-detection-causal-discovery-from-residuals.md`](../../hidden-variable-detection-causal-discovery-from-residuals.md) (EXP-HVD-01, issue #50).
Code: [`run.py`](run.py), which uses `experiments/sketches/hvd_planted_trigger.py` unchanged and the engine's own `pearson_r`, imported read-only. Raw output: [`raw.json`](raw.json), with fire counts and Wilson 95% CIs for all 144 cells.

**Verdict: SUPPORTED (the pre-registered prediction).** The engine's folded residual `|beta − 0.5|` has lower power than a signed residual in every base-logit-0 cell that is not at the floor or ceiling. At β_z = 1 its power never reaches 0.8 anywhere in the grid; the maximum is 0.46. The rule's own H1, that it detects β_z = 1 with power ≥ 0.8 at ≥ 30 claims and a false-alarm rate ≤ 0.05, is not met.

## Pre-registered prediction (from the design, unchanged)

- The folded residual has lower power than the signed one whenever the base logit is ≈ 0.
- Power at β_z = 1 is < 0.8 for ≤ 30 claims.
- Falsified if folded and signed power are equal (±0.05) across all cells, or if power is ≥ 0.8 at 30 claims and β_z = 1.

## Deviations and implementation choices (fixed in `run.py` before running)

- **No deviation from the grid.** The full grid was run: β_z ∈ {0, 0.5, 1, 2} × tests per claim ∈ {3, 5, 10} × claims ∈ {10, 30, 100, 300} × base logit ∈ {−1, 0, 1}, with 500 replicates per cell and a z rate of 0.3.
- **Seeding.** Each cell uses its own `random.Random(cell_index)`.
- **Signed arm keeps the engine's gate.** The signed-residual arm keeps the engine's `mean|residual| ≥ 0.1` gate and its `|r| > 0.5` threshold. Only the fold is removed.

## Environment

Python 3.14.8, numpy 2.5.5, scipy 1.18.1, on an Intel Xeon @ 2.90 GHz (shared). Commit `e814887bed510ef0d4bd67c54820399f0ad088d6`. Runtime: 19 s.

## Results

### False-alarm rate (β_z = 0)

| claims per topic | folded, max over cells | signed, max over cells |
|---|---|---|
| 10 | 0.168 | 0.174 |
| ≥ 30 | 0.008 | 0.006 |

### Power at β_z = 1 (500 replicates; Wilson half-width ≤ 0.045)

| tests per claim | claims | base −1: folded / signed | base 0: folded / signed | base +1: folded / signed |
|---|---|---|---|---|
| 3 | 10 | 0.15 / 0.36 | 0.20 / 0.32 | 0.25 / 0.22 |
| 3 | 30 | 0.01 / 0.23 | 0.03 / 0.15 | 0.09 / 0.08 |
| 5 | 30 | 0.04 / 0.43 | 0.12 / 0.40 | 0.16 / 0.12 |
| 10 | 30 | 0.25 / 0.75 | 0.42 / 0.74 | 0.45 / 0.42 |
| 10 | 100 | 0.10 / 0.89 | 0.23 / 0.84 | 0.40 / 0.36 |
| 10 | 300 | 0.02 / 0.98 | 0.09 / 0.98 | 0.28 / 0.21 |

The other rows are in `raw.json`.

- **Folded power never reaches 0.8 at β_z = 1.** Its maximum is 0.46. Signed power reaches ≥ 0.8 only with 10 tests per claim and ≥ 100 claims, at base logit −1 or 0.
- **Base logit 0, β_z > 0.** In 30 of 36 cells signed power is strictly higher than folded power. The other 6 are ties at 0 or 1 (β_z = 0.5 at ≥ 100 claims; β_z = 2 with 10 tests at ≥ 100 claims).
- **The fold can cancel the effect.** At base logit −1 and β_z = 2, folded power falls to 0.00–0.01 at ≥ 30 claims while signed power is 1.0. The planted effect moves beta across 0.5, so `|beta − 0.5|` changes little.
- **More claims can lower folded power.** Folded power at β_z = 1 decreases as the number of claims grows in several rows (for example 10 tests, base −1: 0.25 → 0.10 → 0.02). The fold and the `|r| > 0.5` bar interact with the sample size; a larger sample does not rescue the rule.

## What this establishes, and what it does not

- **Establishes (synthetic data).** The fold in stage 6's residual can hide a planted exogenous driver, and even without the fold the |r| > 0.5 bar is strict for a binary candidate. Even with a valid candidate series, the rule as written would mostly miss a moderate effect.
- **Establishes (synthetic data).** At 10 claims per topic, the false-alarm rate is 0.07–0.17, above the design's 0.1 bound in 7 of 9 cells.
- **Does not establish** anything about real hidden variables in the engine's data. As the design noted, the synthetic passes are independent binomials, which real engine passes are not, so these figures bound the rule's power from above rather than estimating it.
