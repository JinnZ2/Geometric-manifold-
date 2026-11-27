# EXP-HVD-04 pre-registration addendum (written 2026-11-27, before running)

This plays the role of the "REFUTE addendum" the design asks for. It is kept here, in the results directory, and not in `sims/dark_constraint/`: the run task does not edit repo artefacts outside `experiments/results/`. `FINDING.md`, `REFUTE.md`, `NULL.md` and `config.json` in `sims/dark_constraint/` are untouched.

## The change Q7 asks for

A null arm that carries non-zero visible residuals.

- Every observed truth, in both arms, is produced with visible edge residuals r_vis ~ N(0, 0.03²), drawn once per seed.
- The strength-0 arm is therefore no longer identically equal to the baseline. Its fit residual measures the linearisation and convergence floor.
- **Floor** = the median over seeds of the unexplained residual at strength 0, for the same (kind, location, K).

## Grid

- K ∈ {1, 2, 3, 4}.
- Strength ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0}. This is a log grid; 0.3 and 1.0 are meant to bracket where K = 1 absorption (exact in the linear regime) breaks down through nonlinearity.
- Two kinds: brace and tether (see `config.json`).
- Three locations per kind.
- 10 seeds.

## Measures

- Unexplained residual ‖A r̂ − y‖ of the visible-only linear fit.
- Its ratio to the floor.
- Spearman ρ between strength and residual over all (seed, strength) pairs, per (kind, location, K).

## H1 passes for a (kind, location) if

- K = 1: residual ≤ 3× floor at every strength.
- Every K ≥ 2: ρ ≥ 0.5, and the median residual at the largest strength ≥ 3× floor.

## Falsified if

For K ≥ 2, ρ < 0.5 (flat in strength) in the majority of (kind, location, K) cells.

## Implementation

- `relax` is re-implemented in vectorised form so that it can add the tether kind.
- The brace kind is checked against `sims/dark_constraint/run.py::relax` before any result: agreement to 1e-10 or better is asserted.
- `align`, `load_cases` and the fit are the repo's, imported read-only.
