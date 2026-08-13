# REFUTE — does the shape cast a faithful shadow?

**Written before running.** Thresholds authored here and derived from Maxwell counting, not
transcribed from a prior pre-registration.

## Defining "shadow shape", since nothing else does

The term has **no prior definition anywhere in this repo or the notes** — `grep -ri shadow`
returns nothing. What follows is a proposed operational definition, offered so it can be
corrected rather than assumed.

A Rosetta shape (notes 14 §2) carries equations on edges. A residual vector r in R^12 (one
per edge of the octahedron) deforms the framework into a new configuration. What an
observer actually records is not that configuration but a **reduced statistic** of it. The
shadow is that projection:

| level | observable | dimension | used for |
|---|---|---|---|
| full | Procrustes-aligned displacement field | 18 (12 after rigid modes) | the object itself |
| magnitudes | per-vertex displacement magnitudes | 6 | **notes 14 drill-down** ("per-vertex displacement [0.1445, 0.0225, ...]") |
| scalar | Procrustes distance | 1 | the global balance gauge |

**A shadow fault is a residual pattern that deforms the shape but leaves the observed
statistic unchanged.** The question is whether the drill-down observables are injective on
fault space, or whether real imbalance can hide in the projection's kernel.

## Why Maxwell counting makes this sharp

The octahedron is **isostatic** in 3D: 3x6 - 6 = 12 degrees of freedom, and it has exactly
12 edges. Edge lengths therefore determine the shape exactly — the full-displacement map
should be full rank, and no fault is invisible at that level.

But per-vertex *magnitudes* discard direction: 6 numbers cannot resolve 12 dimensions. If
the rank drops as predicted, at least half of fault space is invisible to the observable
notes 14 actually uses, and "drill-down localizes the fault" holds only on the visible
half.

## Pass condition

- **H1:** rank of the full-displacement Jacobian = 12 (isostatic, as Maxwell predicts).
- **H2:** rank of the vertex-magnitude Jacobian <= 6, so >= 6 residual dimensions are blind
  at the drill-down observation level.
- **H3:** applying the least-visible direction at realistic magnitude produces a
  vertex-magnitude shadow **<= 1/10** of a typical random residual of the same norm —
  verified nonlinearly, not only from the linearization.

## Refutation condition

Any of: the full map is rank-deficient (the isostatic argument is wrong and the whole
framing needs rebuilding); the magnitude map has rank > 6 (magnitudes carry more than
counting allows, so the projection is not the bottleneck); or the blind direction is
visible at more than 1/10 of typical (a near-kernel exists in the linearization but does
not survive to finite deformation, making it a mathematical curiosity rather than a
measurement blind spot).

## What this does and does not challenge

It does not challenge notes 14's positive results. Vertex localization at 6x, face
aggregation at 2.8x and low-mode dominance at 85% were all measured on faults that *do*
cast shadows, and nothing here says those numbers are wrong.

It asks the complementary question, which notes 14 did not: **what does the same instrument
miss?** A drill-down that resolves the faults it can see, while a measurable fraction of
fault space produces no signal at all, is a different instrument from one that sees
everything — and the difference matters for any claim that the shape is a diagnostic rather
than an illustration.
