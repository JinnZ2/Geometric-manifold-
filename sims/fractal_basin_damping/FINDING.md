# FINDING — basin-boundary fractality is governed by damping

Run 2026-08-13T1956Z · 5 damping values × 2 systems × 8 probe seeds, N=200 grid, ~47 s ·
verdict graded by `run.py` against `REFUTE.md`.

## Verdict: SUPPORTED at 100% of cells, and the regression check passes

**C1 (regression against notes/17, γ=0.25).** Reproduced within the ±0.03 tolerance:

| quantity | notes/17 | measured here |
|---|---|---|
| α, double well | 0.688 | **0.690** |
| α, triple well | 0.392 | **0.388** |
| Wada fraction | 8.0% | **8.0%** |

**C2 (the sweep claim).** α rises monotonically with damping in both systems, at every seed:

| γ | α (double) | D_b | α (triple) | D_b | Wada |
|---|---|---|---|---|---|
| 0.05 | 0.307 | 1.693 | 0.202 | 1.798 | **44.0%** |
| 0.10 | 0.470 | 1.530 | 0.251 | 1.749 | 37.4% |
| 0.25 | 0.690 | 1.310 | 0.388 | 1.612 | 8.0% |
| 0.50 | 0.802 | 1.198 | 0.550 | 1.450 | 0.0% |
| 1.00 | 0.857 | 1.143 | 0.688 | 1.312 | 0.0% |

Null: the smooth-boundary control returns α = 1.100 against a true value of 1, so the
estimator is not manufacturing fractality — the run is valid.

## What the sweep adds that the single-damping measurement could not

**1. γ = 0.25 was a mild case.** notes/17 concluded the boundary is "*already* fractal at
moderate damping," which reads as an upper bound on the effect. It is closer to the
middle. At γ = 0.05 the double well reaches α = 0.307 — less than half the α reported —
and the triple well reaches 0.202.

**2. The Wada property is not a fixed feature, it is a damping regime.** notes/17 reports
"8% of boundary cells are Wada — partial Wada at γ=0.25." Across the sweep it runs from
**44% at γ = 0.05 to exactly 0% at γ ≥ 0.5**. Wada structure is not a property of the
triple-well potential; it is a property of the potential *at low damping*, and it
disappears entirely once damping is strong enough. A single-damping measurement cannot
distinguish "this system has partial Wada" from "this system is near the edge of its Wada
regime," and those support very different claims.

**3. The boundary-cell count falls with damping too** (31,544 → 5,568), which is the same
story from another angle: damping shrinks the region where the outcome is uncertain at
all, as well as simplifying its structure.

## Consequence for the physical instrument

notes/17 X1 proposes α as a new instrument channel and reads it as "the exchange rate
between measurement precision and outcome certainty": doubling instrument precision buys
2^α times the certainty. The sweep sharpens that into a design constraint that depends on
a property of the build.

At γ = 0.25, α = 0.69 → doubling precision buys 1.61× certainty. At γ = 0.05, α = 0.31 →
doubling precision buys **1.24×**. A lightly damped printed strut — which is what a stiff
PETG frame ringing in air is — sits at the harsher end. The precision floor X1 describes is
therefore *worse* than the single measurement implies, and how much worse is set by a
quantity the build controls. Damping is a design variable for the instrument, not just a
simulation parameter: adding damping buys back certainty per unit of sensor precision, at
the cost of the long chaotic transients that make the fractal channel informative in the
first place.

That trade is the actual content of X1, and it was invisible at one damping value.

## Caveats

- **The estimator carries roughly +10% bias.** The smooth-boundary null returns α = 1.100
  where theory says exactly 1.0. The bias is common to every point, so the monotone trend
  and the notes/17 reproduction are unaffected, but individual α values are likely biased
  slightly high — meaning true fractality may be marginally *stronger* than the table
  shows. Reducing this needs a finer grid or a fit restricted to scales well inside the
  grid, neither of which is swept here.
- **Seeds vary probe sampling only**, not dynamics; the grid integration is deterministic.
  Eight seeds measure estimator stability, not independent evidence for the phenomenon.
  The γ sweep is what varies the dynamics.
- **Grid resolution and integration time are not swept.** α values that drift with N or T
  would indicate unconverged boundaries; this run cannot rule that out. That is the natural
  next retrofit if these numbers are to carry weight beyond the trend.
