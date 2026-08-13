# FINDING — the documented rationale for the minus sign is refuted on both axes

Run 2026-08-13T2246Z · 8 seeds × 3 drift levels × 2 λ values × 2 signs, 60 steps · ~11 s ·
graded against `REFUTE.md`.

## Verdict: DOCUMENTED CLAIM REFUTED

`CLAUDE.md` states the minus sign in `task_loss − λ·weighted_safety` "creates tension
between task performance and safety alignment" and "ensures the repair explores the loss
landscape rather than collapsing to a local minimum." Operationalised generously — a
formulation that explores to a better basin must *reach* a better basin — the minus arm won
in **0% of 48 cells**.

## It is not a trade-off. It is worse on both axes.

| drift | λ | sign | KL first → final | task final |
|---|---|---|---|---|
| 0.1 | 10 | minus | 0.1319 → **8.4641** | 7.5727 |
| 0.1 | 10 | plus | 0.1319 → **0.0322** | **2.3691** |
| 0.3 | 10 | minus | 3.3616 → **20.4656** | 17.2230 |
| 0.3 | 10 | plus | 3.3616 → **0.4040** | **2.6645** |
| 0.8 | 10 | minus | 32.4031 → **62.8399** | 47.3343 |
| 0.8 | 10 | plus | 32.4031 → **16.6321** | **26.5861** |

At every drift level the plus arm **reduces** safety KL (4×, 8×, 2× respectively) while the
minus arm **increases** it (64×, 6×, 2×). And the plus arm simultaneously reaches **lower
task loss** — 2.37 vs 7.57, 2.66 vs 17.22, 26.59 vs 47.33.

That last column is what kills the "tension" framing. A genuine saddle-point trade would
buy task performance with safety, or the reverse. The minus arm buys nothing: it is worse
at the task *and* further from the basin, everywhere, at every λ.

## The null: started at θ_ref, where there is nothing to repair

| sign | final KL from zero | dist from ref |
|---|---|---|
| minus | **0.205685** | 1.0453 |
| plus | 0.002568 | 0.3685 |

Both move, because θ_ref minimises only the safety term and the composite objective also
carries task loss, whose minimum is elsewhere. But the minus arm leaves **80× faster in
KL**. Started at the exact centre of the basin, it departs.

Every invariant in `tests/test_invariants.py` passes throughout: `‖Δθ‖ ≤ trust_radius` holds
each step and `dist ≤ N·trust_radius` holds trivially, because that bound is linear in N and
unbounded. The suite cannot express "the operator should not be leaving."

## H3: doubling λ increases divergence

Median final KL for the minus arm: **8.46 at λ=10, 8.49 at λ=20** (drift 0.1); 20.47 → 20.54;
62.84 → 62.99. Small but consistently upward at every drift level. `adversarial.yaml` raises
λ to 20 precisely for stress conditions, which under this sign increases the escape force
rather than the restoring force.

## A gate I mis-specified, disclosed

The first run graded **PRIOR REFUTED** because H1 demanded the plus arm hold `KL ≤ 1e-6`
from a start at θ_ref, and it reached 2.57e-03.

That gate was wrong **on physics grounds, not results grounds**: θ_ref minimises the safety
term alone, while the objective under test also contains task loss, whose minimum lies
elsewhere. No composite objective can be stationary at θ_ref, so the absolute threshold was
unmeetable by construction and would have failed for *any* sign. It was replaced with the
comparative form the null existed for — does the plus arm stay markedly closer than the
minus arm — which it passes by 80×.

Per `HARNESS.md` §4 the correction was made after seeing output, so this run wears that
label. Two things limit the damage: the correction is justified by the objective's
structure rather than by which arm it favours, and it does not touch H2, which is the
hypothesis that actually decides the verdict and which the minus arm failed at 0% of cells
under both the original and corrected gates.

## What this does and does not establish

**Established:** the documented rationale is false as written. The minus sign does not
produce a better basin, does not trade task for safety, and moves away from the reference
from a start where nothing needs repairing.

**Not established:** what the sign was *for*. If a third reading exists that neither the
physics prior nor the exploration claim captures, it needs writing down before it can be
tested — this sim can only test the readings the repo states.

**Also relevant:** the framework already contains the other convention.
`addon_thermodynamic_control/stability.py` uses `l_task + λ_s·l_safe + λ_p·j_pro + μ·fish_reg`
with an adapting μ — which is the standard Lagrangian form `TERMINOLOGY_MAP.md` §1 cites
(Altman 1999; Achiam et al. 2017), where θ descends and the *multiplier* ascends. The plus
arm here is not a novel proposal; it is what half of this repo already does.

**Consequence for earlier work in this session:** `sims/rate_induced_escape/` measured
σ_crit ≈ 0.0008 and a repair capped at ~0.000025 KL per step. That was run on
`CoupledDynamicalSystem` — the **plus** path. Those numbers describe a restorative
controller and do not transfer to the `main.py` → `Controller` → `ParameterManifold`
pipeline, which is the minus path. I should have caught that the two disagreed before
generalising.
