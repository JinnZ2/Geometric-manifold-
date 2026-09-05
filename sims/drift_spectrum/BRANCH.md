# BRANCH RECORD — drift_spectrum

## BRANCH ENTRY 02 — opened by axis disagreement (order §8)

Run `2026-09-05T0301Z`. Licence status at opening: S4 FAILED (see RESULTS.md §0).

```
rule as stated    parameter drift has a spectrum
forcing case      the three candidate axes return different answers to "does the drift
                  spectrum have non-monotone structure over training" on the same runs
axis              A1/TIME=none, A2/UNIT-L1=structure, A2/UNIT-L2=none, A3/SEED-raw=none, A3/SEED-aligned=none
derivation        parameter space has no sample dimension; every spectrum is a spectrum of
                  the axis that supplied one. Axes that disagree are measuring different
                  things, so "the drift spectrum" does not denote one object.
frame note        the disagreement is a property of the substitution, not of the drift.
                  It stands whether or not S4 licenses a comparison to the reference channel,
                  because it is internal to the drift side.
```
