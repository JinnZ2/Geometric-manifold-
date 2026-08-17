# NULL — objective sign

**Written before running.**

## The null model

`start_at_reference` — both arms initialised at **θ_ref exactly**, where `KL = 0`, the
safety gradient is at the potential minimum, and a repair operator has nothing to repair.

## Why this null

It is the one condition where the correct behaviour is unambiguous under either reading of
the objective. Whatever "repair" means, an operator started at the target with zero
divergence should not increase divergence. This separates *the operator's intrinsic
direction* from any question about how well it recovers from drift — no recovery is being
asked for.

It is also the null that the existing invariant suite cannot express. `tests/test_invariants.py`
checks `‖Δθ‖ ≤ trust_radius` and `dist ≤ N · trust_radius`; both are satisfied by an
operator that walks steadily away from the reference, because the bound is linear in N and
unbounded. A null that starts at the target and watches the sign of dKL/dt is the missing
check.

## What the null must show

- **The plus arm must not increase KL from zero.** If it does, the objective is not a
  potential well in the way the prior claims and nothing downstream is interpretable.
- **The minus arm's behaviour here is the measurement**, not a control. Whatever it does at
  the exact centre of the basin is the cleanest available statement of what the sign does.

## What it does not cover

Behaviour far from the reference, where the task-loss term dominates and the two arms may
converge in practice regardless of sign. That is what the drift sweep is for; the null
isolates direction, not magnitude.
