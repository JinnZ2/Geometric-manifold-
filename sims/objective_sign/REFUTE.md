# REFUTE — is the minus sign in `task_loss − λ·safety` doing what the docs claim?

**Written before running.** The repo's governing rule is the scientific method with physics
laws as an *a priori* base held **defeasibly** — a strong prior, not a certainty. So
`CLAUDE.md`'s "Do Not change the sign in `task_loss - λ * weighted_safety`" is treated here
as a **claim under test**, not an axiom, and the physics prior is likewise stated so it can
lose.

## The physics prior (strong, and falsifiable)

`safety_loss` is `KL(f_θ ‖ f_θ_ref)`: non-negative, zero exactly at the reference, with the
Fisher information as its local metric (`KL ≈ ½ Δθᵀ F Δθ`). That is a **potential well
centred on θ_ref**. A restoring force toward a basin is `−∇(potential)`, i.e. gradient
*descent* on KL. Ascending it makes the reference an unstable equilibrium — repulsive, not
attractive.

The saddle-point reading has a specific standard form, and it is the one
`TERMINOLOGY_MAP.md` §1 already cites (Altman 1999; Achiam et al., CPO 2017):

    min_θ max_{λ≥0}  [ task_loss + λ·(constraint_violation − budget) ]

**θ descends; λ ascends.** The safety term enters θ's objective with a *plus*; the
adversarial tension lives in the multiplier, not in the sign of θ's gradient. Notably
`addon_thermodynamic_control/stability.py` already implements exactly this — `+λ_s·l_safe`
with an adapting `μ` — while `manifolds/parameter_manifold.py` uses the minus and holds λ
fixed. The framework contains both conventions.

**Prior prediction:** the plus arm restores toward the basin; the minus arm diverges from
it, including from a start where there is nothing to repair.

## The documented claim (given its fair chance)

`CLAUDE.md` states the minus sign "creates tension between task performance and safety
alignment" and "ensures the repair explores the loss landscape rather than collapsing to a
local minimum." That is an **exploration** claim, and exploration claims are testable: a
method that explores better should *end up somewhere better*, not merely move more.

Operationalised, and deliberately generous to the documentation: the minus arm wins if it
reaches **lower final safety KL at matched-or-better task loss** than the plus arm. A
formulation that explores to a better basin must eventually show a better basin.

## Pass / refute

- **H1 (restorative operator).** From a start at θ_ref, where KL = 0 and there is nothing
  to repair, a repair operator must not walk away. Plus arm: final KL ≤ its own start.
  Minus arm: reported, not assumed.
- **H2 (the documented exploration claim).** The minus arm attains lower final safety KL
  than the plus arm at matched-or-better task loss, in ≥ 80% of (seed, drift, λ) cells.
- **H3 (the λ prediction).** If the minus sign is an escape force, doubling λ (default 10 →
  adversarial 20) increases divergence rather than decreasing it.

**The documentation is refuted if H2 fails**, i.e. the minus arm never reaches a better
basin — leaving "explores rather than collapsing" as motion without arrival.

**The physics prior is refuted if H2 holds**, or if the plus arm fails H1. Either would
mean a KL-to-reference term is not acting as the potential well the prior says it is, and
the prior would have to give way to the measurement.

## What this cannot settle

Whether the *intent* behind the minus sign was something neither reading captures. The sim
tests the two readings that are stated in the repo. If a third is meant, it needs writing
down before it can be tested.
