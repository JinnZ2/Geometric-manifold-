# Open questions — to be settled by measurement, not by preference

Each entry is a design or interpretation question that surfaced from a measurement and
cannot be answered by opinion. Every one gets a named experiment. Status is the honest
current state, not an intention.

The rule: nothing here gets decided by asserting a preferred answer. If an entry stays
open, it stays open in the code too — the code should not quietly imply a resolution the
measurements have not reached.

---

## Q1 — Does κ_eff work near the critical rate? (the fair test it has not had)

**Status: UNTESTED.** `sims/kappa_eff_leading/` refuted κ_eff across three rounds, but
`sims/rate_induced_escape/` then measured σ_crit ≈ 0.0008–0.0012 and those rounds all ran
at σ = 0.012–0.020 — **10–24× above critical**, deep in the committed regime where net
ΔKL/step exceeds the repair ceiling by two orders of magnitude and *no* indicator can lead
because nothing is approaching anything.

**Experiment:** `sims/critical_regime_indicators/` (designed, not yet run). Run at
σ ∈ [0.0006, 0.0016] bracketing σ_crit. Candidate signals: κ_eff, λ_min of the safety
Hessian, the saturation margin ρ = drift ΔKL ÷ repair cap, and time-to-boundary
(ε − KL)/(dKL/dt). Same causal-criterion sweep, same null arm, same free θ-distance
baseline. This is the first regime in which a leading indicator could exist at all.

**Why it matters:** the current honest status of κ_eff is "refuted in the committed regime,
untested near σ_crit." Leaving it there would let a stronger claim stand than the evidence
supports, in either direction.

---

## Q2 — Is `trust_r` shrinking with μ an inverted sign?

**Status: PARTIALLY TESTED — arm added to `sims/rate_induced_escape/`.**

`trust_r = lr / (1 + μ·max(fisher))` caps the repair step, and μ grows when the repair
budget is exceeded or the Lyapunov derivative goes positive — i.e. *when the system is in
trouble*. So corrective authority contracts exactly when more of it is wanted, which lowers
σ_crit under stress. That may be a deliberate homeostat (spend less when the budget is
gone) or an inverted sign.

**Experiment:** μ-adaptation arm, `mu_mode ∈ {adaptive, frozen}`. Frozen μ holds `trust_r`
constant. If σ_crit is materially *higher* with μ frozen, adaptation is costing basin
stability; if it is unchanged, the concern is theoretical.

**What the measurement cannot settle:** whether trading basin stability for budget
conservation is the right call. That is a design intent question. The measurement can only
price the trade.

---

## Q3 — Is `C_bound = 20.0` calibrated to anything?

**Status: MEASURED, UNRESOLVED IN CODE.** Observed κ_eff spans 0.0001–0.37 across 35 runs
and stays under 0.05 in every isotropic run. `_phase()` at `stability.py:441` calls
"critical" when `kappa > C_bound = 20.0` — roughly 50× the largest value ever observed. The
κ branch of the phase classifier cannot fire; phase is decided entirely by `basin_kl` and
`dV_dt`.

**Experiment:** none needed to establish the fact — it is measured. What is untested is
whether a *scale-relative* bound (e.g. a multiple of the running κ_eff median) produces a
phase classifier that discriminates better than `basin_kl` alone. That is a small addition
to Q1's harness and should ride along with it.

**Why it is still open:** either calibrate the bound or remove the branch. Both are code
changes with consequences; the measurement says only that the current value is inert.

---

## Q4 — Does any of this survive at model scale?

**Status: UNTESTED, and the only live explanation preserving the κ_eff claim.** The
Rayleigh quotient of an ~8k-parameter safety Hessian may be dominated by sampling noise.
Every result here is at ToyLLM scale and none of it transfers upward on its own.

**Experiment:** the same indicator battery on a model 1–2 orders of magnitude larger, with
the Hessian-vector product cost measured rather than assumed. This is the expensive one and
should not be attempted until Q1 has run — if κ_eff fails *near* σ_crit at toy scale, scale
is a weaker hypothesis than it looks; if it succeeds, scale becomes the priority.

---

## Q5 — Do the shape assignments carry information, or only the shadow of it?

**Status: EXPERIMENT BUILT — `sims/shape_shadow/`.** See that sim's `REFUTE.md` for the
operational definition; "shadow shape" had no prior definition anywhere in this repo or the
notes, so one is proposed there and is open to correction.

The question in short: notes 14 claims drill-down localizes a fault from the deformed
shape. But the observables it uses are a *projection* of the full deformation — per-vertex
displacement magnitudes discard direction, and Procrustes distance discards everything but
scale. Maxwell counting says the octahedron is isostatic (3×6 − 6 = 12 = its edge count),
so edge residuals determine shape exactly — but a projection of that shape need not
determine the residuals. If the projection has a kernel, there are real faults that cast no
shadow.

---

## Q6 — Does the TETRA assignment mean three weights or four channels?

**Status: OPEN, AND CURRENTLY UNFALSIFIABLE AS WRITTEN.** notes 14 makes each shape
assignment a falsifiable claim — "if confidence aggregation ever carries ≠4 components, the
shape is wrong" — but justifies TETRA by four channels (data/param/policy/combined) while
`bridges/rosetta-bridge.json` justifies it by three weights ("the irreducible three-weight
combination"). Both land on TETRA; they disagree on what would refute it.

**Experiment:** none possible until the claim names one count. This is the one entry here
that is blocked on a definition rather than on a measurement, and it should be resolved by
whoever wrote the assignment, not by a sim picking whichever reading is convenient.
