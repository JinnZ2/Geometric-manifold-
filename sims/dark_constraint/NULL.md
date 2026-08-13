# NULL — dark constraint

**Written before running.**

## The null model

`dark_strength_zero` — the identical pipeline with the hidden brace present in the code but
set to zero strength. Same load cases, same relaxation, same fit, same observables. The
only thing removed is the dark component itself.

## Why this null

The measured quantity is an *unexplained residual*, and residuals are never exactly zero:
finite relaxation, numerical Jacobians and linearization error all leave a floor. Without
the null there is no way to tell a dark component from that floor, and any absolute
threshold would be a statement about the integrator rather than about hidden constraints.

The null fixes the floor by construction. Whatever residual it produces is what the
pipeline generates when there is provably nothing to find.

## What the null must show

- **A residual floor well below the smallest dark signal** — the pass condition asks for a
  10x separation, so a null within 10x of the weakest dark case makes that case
  undetectable regardless of what the mechanism does.
- **No growth in K.** The floor should not systematically rise with more load cases; if it
  does, the apparent detection at K >= 2 would be an artifact of stacking rather than
  evidence of an unmodelled component.

## What it does not cover

The null shares the same brace *location* as the test arm. It therefore does not test
whether detectability depends on which antipodal pair carries the dark constraint, or on
whether the dark component is a brace at all rather than, say, a soft coupling or an
external field. Those are separate experiments; this one establishes only that a
constraint-shaped dark component is degenerate at K=1 and detectable at K>=2.
