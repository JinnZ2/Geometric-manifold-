# EXP-HVD-04 results: Q7, does K ≥ 2 recover an unmodelled constraint against a non-degenerate null? (run 2026-11-27)

- Design: [`experiments/hidden-variable-detection-causal-discovery-from-residuals.md`](../../hidden-variable-detection-causal-discovery-from-residuals.md#exp-hvd-04--q7-does-k--2-recover-an-unmodelled-constraint-against-a-non-degenerate-null).
- Pre-registration addendum, written before running: [`prereg.md`](prereg.md).
- Config: [`config.json`](config.json).
- Code: [`run.py`](run.py). Raw output: [`raw.json`](raw.json).
- Post-hoc check, not pre-registered: [`posthoc.py`](posthoc.py), output [`posthoc.json`](posthoc.json).

## Verdict

**SUPPORTED on the pre-registered criteria for K ≥ 2 (the falsifier is not met), with a material qualification.** In the low-strength range, what K ≥ 2 detects is that the hidden spring is present, not how strong it is.

- **K ≥ 2.** Spearman ρ(strength, residual) is 0.59–0.85 in all 18 cells (2 kinds × 3 locations × K ∈ {2, 3, 4}). The prediction required ρ ≥ 0.5. The falsifier, ρ < 0.5 in the majority of cells, is met in 0 of 18.
  - The median residual at the largest strength is 77–113 times the non-degenerate floor. The prediction required ≥ 3 times.
- **K = 1.**
  - Tether: absorbed at every strength (ratio to floor 1.0).
  - Brace: absorbed up to s = 0.3 (ratio ≤ 2.0). Absorption breaks down at s = 1.0, at 3.2–4.9 times the floor. The grid was meant to bracket exactly this point. By the letter of `prereg.md` ("≤ 3× floor at every strength"), the K = 1 part therefore fails for all three brace locations.
- **Qualification.** At K ≥ 2 the residual jumps 13–82 times the floor as soon as the constraint is present (s = 0.01). It is then nearly flat up to s = 0.1: the median ratio of s = 0.1 to s = 0.01 is 0.92–1.35. It grows only at s = 0.3–1.0.
  - Recomputed over s > 0 only (post hoc), ρ is 0.51–0.75 for the brace but 0.29–0.53 for the tether. That is below 0.5 in 7 of 9 tether cells at K ≥ 2.
  - The pre-registered ρ pools s = 0 (absent) with s > 0 (present), so the step alone carries much of it.

## Environment and provenance

- Commit e814887bed510ef0d4bd67c54820399f0ad088d6.
- Python 3.14.8, numpy 2.5.5, scipy 1.18.1.
- CPU: Intel Xeon @ 2.90 GHz, 4 cores, shared with other jobs.
- Runtime: 148 s on 3 worker processes.
- Seeds:
  - 0–9. Load cases use `sims/dark_constraint/run.py::load_cases`, rng 1000 + seed.
  - Visible residuals use rng 5000 + seed.

## Deviations (recorded in `prereg.md` and `config.json` before running)

1. **Placement of the new config and addendum.** They live in this results directory, not in `sims/dark_constraint/`. `FINDING.md`, `REFUTE.md`, `NULL.md` and `config.json` there are untouched.
2. **`relax` re-implementation.** `relax` was re-implemented in vectorised form so it could add the tether kind. For the brace, it matches the repo's `relax` to 8.9e-16 (asserted before the run). `align`, `load_cases` and the linear fit are the repo's.
3. **"Two constraint kinds".** The design left this open. Fixed as:
   - **brace**: the repo's hidden antipodal spring, rest length ref·(1 + s).
   - **tether**: a hidden spring pulling vertex i toward V0[i] + s·u, with u tangent.
   - Both have unit stiffness. Strength 0 means the constraint is absent, as in the repo, where the brace is omitted when `dark == 0`.
4. **Strength grid.** {0, 0.01, 0.03, 0.1, 0.3, 1.0}. Seeds: 10.

## Results

Median unexplained residual over 10 seeds, as a ratio to the floor (the s = 0 median for the same kind, location and K). Floors:

| K | Floor | Note |
|---|---|---|
| 1 | 2.6×10⁻¹³ | Still structurally exact: 12 parameters for 12 shape DOF, so the fit absorbs anything. |
| 2 | 1.15×10⁻⁴ | Non-degenerate. |
| 3 | 1.53×10⁻⁴ | Non-degenerate. |
| 4 | 1.64×10⁻⁴ | Non-degenerate. |

| kind | loc | K | s=0.01 | 0.03 | 0.1 | 0.3 | 1.0 | ρ (pre-reg) | ρ, s > 0 (post hoc) |
|---|---|---|---|---|---|---|---|---|---|
| brace | (0,1) | 1 | 1.0 | 0.9 | 0.9 | 1.6 | 4.5 | 0.48 | 0.56 |
| brace | (0,1) | 2 | 22.3 | 22.0 | 22.6 | 35.2 | 97.2 | 0.81 | 0.68 |
| brace | (0,1) | 3 | 30.0 | 29.6 | 28.9 | 34.1 | 93.5 | 0.72 | 0.53 |
| brace | (0,1) | 4 | 36.7 | 36.2 | 35.4 | 40.0 | 105.2 | 0.72 | 0.51 |
| brace | (2,3) | 1 | 1.0 | 1.0 | 1.1 | 2.0 | 4.9 | 0.46 | 0.51 |
| brace | (2,3) | 2 | 12.6 | 13.1 | 17.0 | 31.9 | 101.4 | 0.85 | 0.75 |
| brace | (2,3) | 3 | 20.6 | 20.1 | 20.1 | 33.0 | 102.9 | 0.82 | 0.69 |
| brace | (2,3) | 4 | 28.2 | 27.6 | 26.6 | 35.1 | 112.8 | 0.79 | 0.64 |
| brace | (4,5) | 1 | 0.9 | 0.9 | 1.1 | 1.5 | 3.2 | 0.52 | 0.61 |
| brace | (4,5) | 2 | 24.5 | 24.1 | 24.8 | 33.0 | 99.4 | 0.83 | 0.71 |
| brace | (4,5) | 3 | 35.1 | 34.3 | 32.5 | 35.8 | 97.7 | 0.72 | 0.51 |
| brace | (4,5) | 4 | 35.7 | 34.9 | 32.9 | 39.1 | 107.3 | 0.73 | 0.54 |
| tether | v0, +y | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.01 | 0.01 |
| tether | v0, +y | 2 | 54.3 | 55.1 | 57.9 | 65.9 | 83.0 | 0.65 | 0.39 |
| tether | v0, +y | 3 | 64.5 | 64.8 | 66.1 | 66.4 | 85.7 | 0.62 | 0.34 |
| tether | v0, +y | 4 | 82.0 | 82.5 | 82.2 | 82.5 | 100.9 | 0.65 | 0.39 |
| tether | v2, +z | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | −0.05 | −0.04 |
| tether | v2, +z | 2 | 55.1 | 54.2 | 51.1 | 46.1 | 77.1 | 0.60 | 0.31 |
| tether | v2, +z | 3 | 54.3 | 54.2 | 53.9 | 56.4 | 83.9 | 0.59 | 0.29 |
| tether | v2, +z | 4 | 64.4 | 64.4 | 64.7 | 67.6 | 95.0 | 0.64 | 0.38 |
| tether | v4, +x | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | −0.01 | −0.00 |
| tether | v4, +x | 2 | 47.2 | 47.5 | 48.8 | 53.6 | 77.7 | 0.64 | 0.38 |
| tether | v4, +x | 3 | 56.7 | 57.1 | 58.5 | 58.6 | 83.8 | 0.72 | 0.52 |
| tether | v4, +x | 4 | 59.5 | 59.9 | 61.2 | 66.8 | 100.6 | 0.73 | 0.53 |

Reading the table: a hidden spring of unit stiffness changes how the framework responds to the load cases, whatever its rest-length offset. With two or more independent load cases, the visible-only model cannot reproduce that change in compliance. With one load case it can, because the fit has as many parameters as there are shape DOF. This explains the step. The rest-length strength adds a second, smaller effect, visible from s = 0.3. This is a structural reading of the toy that fits the numbers. It was not tested separately, for example by varying stiffness at a fixed offset.

## Dated design note (2026-11-27, added here, not in the design or in `sims/`)

"Dark strength" in `sims/dark_constraint/` conflates two things: whether the hidden spring is present (s = 0 removes it) and its rest-length offset. The repo's earlier observation, that "a residual appears at K ≥ 2 but does not scale with dark strength", is reproduced here with a non-degenerate null. It is explained by the presence step. A follow-up that separates presence (stiffness k at fixed offset) from offset (s at fixed k) would answer whether K ≥ 2 measures strength. That follow-up is UNCHECKED.

## What this establishes and what it does not

- It establishes the following for this toy (octahedron, two hidden-constraint kinds, 10 seeds):
  - With a non-degenerate null, K ≥ 2 load cases separate "hidden constraint present" from "absent" by 13–100 times the linearisation floor.
  - K = 1 absorbs the constraint exactly, except for the brace at s = 1.0.
- It does not establish that the residual measures constraint strength in the 0.01–0.1 range; there it is flat.
- It does not establish anything about astrophysical dark matter, the analogy that motivated Q7.
- It does not replace the repo's own FINDING, which is unchanged.
