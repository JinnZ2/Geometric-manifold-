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

---

## Addendum: is the minus sign "like imaginary numbers"?

A good question, and it names something real. Physics does use wrong-sign quantities as
essential machinery rather than errors:

- **An imaginary frequency is how you label an unstable mode.** `ω² < 0` means the mode
  grows instead of oscillating; below a spinodal the curvature is negative and `√(k/m)` is
  imaginary. Physicists write instability that way on purpose.
- **Escape rates are computed by leaving the real axis.** Kramers escape and false-vacuum
  decay go through the **bounce** — a solution existing only in imaginary time — and the
  decay rate is `Γ = −2 Im F`, literally the imaginary part of the free energy. `notes/17`
  §3 already cites Coleman thin-wall nucleation, so this connection is native to the
  ecosystem rather than imported.
- **Cardano's formula** needs complex intermediates to return three real roots.

So the intuition is not loose. It also supplies a sharp criterion, because every one of
those cases shares one property: **the excursion returns.** Analytic continuation comes
back to the real axis; the bounce starts and ends at the false vacuum; the cubic's complex
intermediates cancel. Machinery that leaves must come back, or it is not machinery.

**Measured, 1200 steps on the minus arm from a drifted start:**

| step | KL | dist | task |
|---|---|---|---|
| 0 | 3.79 | 17.08 | 7.56 |
| 200 | 104.53 | 21.19 | 79.61 |
| 500 | 489.85 | 32.42 | 299.84 |
| 1199 | **2571.02** | 64.32 | 1405.08 |

- **Minimum KL over the entire run occurs at step 0.** It is never closer to the basin than
  where it began.
- **Strictly monotone increasing at every one of 1200 steps.** No turning point exists.
- Both objectives diverge together — KL by 678×, task loss by 186×.

There is no bounce. The excursion does not return, at any timescale tested.

A second disanalogy is worth naming precisely: an instanton saddle has **exactly one**
negative mode — the single direction along which the barrier is crossed — and that lone
mode is what makes `Im F` non-zero. The code applies the minus sign to the *entire* safety
gradient, flipping every direction at once. That is not a saddle with one unstable
direction; it is an inverted potential.

## But the intuition recovers the intent, and names what is missing

`CLAUDE.md` says the sign "ensures the repair explores the loss landscape rather than
collapsing to a local minimum." That is a recognisable algorithm: **basin hopping** —
deliberately go uphill, then re-minimise, keep the result if it is better. Simulated
annealing has the same shape.

The minus sign implements the *first half* of basin hopping and omits the second. There is
no re-minimisation, no acceptance test, no return. Uphill moves are not the error; uphill
moves with no downhill phase and no accept/reject criterion are.

That makes the constructive version concrete, and testable in the same harness:

1. **Ascend, then descend.** Alternate a bounded uphill excursion with a re-minimisation,
   accepting only if final KL improves. This is the documented intent, implemented.
2. **Ascend along one mode only.** Flip the sign for the lowest-curvature direction and
   descend the remainder — an actual saddle rather than an inverted bowl.
3. **Use it as a rate, not a trajectory.** `Im F` is a number to read off, not a path to
   walk. An escape-rate estimator built from the unstable mode would be a diagnostic, and
   diagnostics do not have to be followed.

None of these is the current code, and any of them would be a fair test of what the sign
was reaching for.

---

## Addendum 2: is this being read in too few dimensions?

A fair challenge, and aimed at the method rather than the code: every verdict above was
read off `KL(t)`, one scalar summarising a 3152-dimensional trajectory. `sims/dark_constraint/`
established that a low-dimensional observable can absorb a cause entirely, so the same
worry applies here — a scalar that rises monotonically could hide a return happening in
some orthogonal subspace.

**Measured.** Displacement snapshots along the trajectory, Gram matrix, participation ratio
as effective rank. Three seeds per cell:

| arm | drift | effective rank | variance in top direction |
|---|---|---|---|
| minus | 0.1 | 1.01 | 0.9936 |
| minus | 0.3 | 1.01 | 0.9939 |
| minus | 0.8 | 1.00 | 0.9978 |
| plus | 0.1 | 1.04 | 0.9808 |
| plus | 0.3 | 1.06 | 0.9700 |
| plus | 0.8 | 1.08 | 0.9638 |

**Effective rank ≈ 1 out of 3152.** Over 99% of the trajectory's variance lies in a single
direction, for both signs at every drift level. There is no orthogonal subspace for a
return to hide in, because there is essentially no orthogonal motion. The scalar was a
faithful description, and the verdict stands.

But the check was worth running, because it changes two things.

**It makes one of the proposed fixes a near no-op.** Addendum 1 offered "ascend along one
mode only, descend the rest — an actual saddle rather than an inverted bowl." The dynamics
*already* select a single mode: repeated gradient steps align with the dominant
eigendirection, exactly as power iteration does. Restricting the ascent to one mode would
change little, because the realized path is already one-dimensional. **The intervention
that matters is the return/acceptance step, not the mode restriction.**

**It partially rehabilitates the imaginary-number reading.** Addendum 1 dismissed it partly
on the grounds that "an instanton saddle has exactly one negative mode, and the code flips
every direction — an inverted potential, not a saddle." That is true of the *objective* and
false of the *trajectory*: the realized escape has precisely the single-unstable-direction
character an instanton has. The analogy was closer than that refutation allowed. What it
lacks is only the return — which is the same conclusion, reached with one fewer bad reason.

**A by-product relevant to Q1.** `kappa_eff = θ̇ᵀHθ̇ / θ̇ᵀθ̇` is a Rayleigh quotient along the
flow direction, and the flow direction is rank-1 aligned. So κ_eff is not measuring general
curvature; it is approximating the top eigenvalue of the safety Hessian along whichever
mode the optimizer has locked onto. That is a sharper characterisation than
`DOMAIN_PHYSICS.md` §5 gave, and it holds for both signs, since both arms collapse to a
dominant direction. It belongs in the Q1 write-up when that experiment runs.
