# Where the physics leads: this framework's failure mode is saturation, not curvature

Written 2026-08-13, after `sims/kappa_eff_leading/` refuted κ_eff across three rounds and
`sims/rate_induced_escape/` found the mechanism. Measured, not asserted — every number
below has a ledger entry.

---

## 1. The transition class was misidentified

`addon_thermodynamic_control/stability.py` line 32 states the claim:

> Spike in kappa_eff precedes behavioral collapse.

That is a **bifurcation** framing: the landscape develops a feature, and a curvature
statistic sees it coming. It is the physics of a fold, and it is the physics the whole
ecosystem is built around — `k_eff ∝ √(1 − c/c_snap)`, recovery time diverging, the
spinodal at 2/√27.

The measured mechanism is a different class of transition entirely.

`CoupledDynamicalSystem` caps every repair step at `trust_r = lr / (1 + μ·max(fisher))`,
which is at most `lr = 0.01` and shrinks as μ adapts upward. Measured in the basin
coordinate, that cap makes the repair remove a **near-constant amount of KL per step** no
matter how hard the system is driven, while injected drift adds KL in proportion to **σ²**:

| σ | drift ΔKL/step | repair ΔKL/step | net |
|---|---|---|---|
| 0.0005 | +0.000012 | −0.000024 | **−0.000012** |
| 0.001 | +0.000031 | −0.000023 | +0.000008 |
| 0.002 | +0.000092 | −0.000023 | +0.000069 |
| 0.004 | +0.000308 | −0.000028 | +0.000280 |
| 0.012 | +0.002620 | −0.000033 | +0.002587 |

The repair column is flat across a 24× range of forcing. That is the signature of an
actuator at its limit, not of a landscape changing shape.

## 2. It makes a prediction with no free parameters, and the prediction holds

A capped corrector against quadratic forcing crosses over at

    σ_crit = √(repair_cap / k),    where  ΔKL_drift = k·σ²

Both constants come from a single calibration run at one σ, so the crossing point is
predicted rather than fitted. Measured against it:

- predicted **σ_crit = 0.00120**
- measured (zero crossing of net ΔKL/step) **σ_crit = 0.00084**
- ratio **0.70×**, inside the pre-registered 2× tolerance → SUPPORTED
- null arm (no forcing): net ΔKL/step = −0.000021, so the repair does hold an unforced basin

## 3. The same object appears in five domains, and none of them measures curvature

This is the cross-domain answer. The quantity that governs a saturation transition is a
**ratio of rates**, and the ecosystem's own notes already contain it four times over
without connecting it to κ_eff:

| Domain | Name | Statement |
|---|---|---|
| Cybernetics | **Ashby requisite variety** (notes 10 §1.4) | V(regulator) ≥ V(disturbance) − V(buffer). A hard lower bound on corrector capacity. |
| Collapse modelling | **Drag ratio L/A** (notes 07) | L/A > 1 ⇒ COMMITTED. Load exceeds adaptive capacity. |
| Dynamical systems | **R-tipping** (Ashwin 2012; notes 07 §3.1) | "Collapse can occur at M(S) > 0 if forcing outpaces A" — tipping without the control parameter reaching a bifurcation. |
| Control engineering | **Actuator saturation** | Once the actuator clips, the loop is open and the error integrates freely. |
| Chemical physics | **Kramers escape** | Escape when forcing exceeds the maximum restoring force, independent of well curvature. |

Every one compares a disturbance rate to a maximum correction rate. **None of them looks
at the curvature of the landscape**, because in a saturated regime the landscape's shape no
longer determines the outcome — the corrector's ceiling does.

The saturation margin ρ = (drift ΔKL/step) / (repair cap) is literally MCPM's L/A applied
to this framework. It is free to compute, and it predicted the critical rate to 0.70×.

## 4. Why the fold physics does not transfer, even though it is right elsewhere

Critical slowing down is real and the ecosystem measures it correctly elsewhere — notes 14
§8 recorded recovery time going from 70 steps to 600+ across 25% of the load range on the
bistable shape. But CSD is the signature of a **fold**: λ_min → 0, the restoring force
vanishes, recovery time diverges as 1/|λ|.

The manifold framework's basin is a **KL sublevel set**, `{θ : KL(f_θ‖f_ref) < ε}`. There
is no barrier, no second stable state, and no eigenvalue going to zero at the boundary —
KL simply grows as you move away. Crossing it is a level-set crossing under competing
rates, not a bifurcation.

**The framework imported the instrument's physics without the instrument's bistability.**
The octahedral strut in notes 15 has a genuine von Mises snap-through, so CSD applies there
and the E-P2 protocol is sound. A KL ball around a reference model has no such structure,
so no curvature statistic has privileged early-warning status over the basin coordinate
itself.

## 5. Two instabilities the framework conflates

κ_eff is a Rayleigh quotient along the flow direction, so it is bounded between λ_min and
λ_max and is cleanly neither:

- **Edge of stability** (Cohen et al. 2021, cited in `TERMINOLOGY_MAP.md` §1) — λ_max·lr > 2
  and the optimizer diverges. Here **large** curvature is the danger, and a spike is the
  right alarm.
- **Fold / basin escape** — λ_min → 0 and the restoring force vanishes. Here **small**
  curvature is the danger, and a spike is the wrong sign entirely.

The claim attached to κ_eff ("spike precedes collapse") is an edge-of-stability alarm
wired to a basin-escape claim. In this framework neither applies, because the operative
failure is saturation.

## 6. What this revises about the κ_eff refutation — including my own reading of it

The three rounds in `sims/kappa_eff_leading/` ran at σ ∈ {0.012, 0.016, 0.020}. Against a
critical rate of 0.0008–0.0012, that is **10–24× above the transition**. In that regime the
system is committed from step 0: net ΔKL/step is positive by two orders of magnitude and
the outcome carries no information beyond the rate inequality.

So those runs are a fair test of *"does κ_eff work in the committed regime"* — it does not,
and neither does anything else, because there is nothing to lead. They are **not** a test of
*"does κ_eff work near the critical point"*, which is the claim that matters and which
remains untested. This was written into `sims/rate_induced_escape/REFUTE.md` before the run,
precisely so that a SUPPORTED verdict would be allowed to undercut the earlier conclusion
rather than quietly coexisting with it.

The honest status of κ_eff is therefore weaker than the last write-up implied: **refuted in
the committed regime, untested near σ_crit.**

## 7. What to do next, in order

1. **Re-run the leading-indicator comparison at σ ≈ 0.0008–0.0012**, where an approach to
   the boundary actually exists. Candidates: κ_eff, λ_min of the safety Hessian, the
   saturation margin ρ, and time-to-boundary extrapolation (ε − KL)/(dKL/dt), which is
   MCPM's ttc. This is the fair test κ_eff has not yet had.
2. **Report the saturation margin regardless.** ρ > 1 is a free, derived, physically
   meaningful COMMITTED flag, and it currently has no counterpart in the framework.
3. **Fix the phase classifier's dead branch.** `_phase()` calls "critical" at
   `kappa > C_bound = 20.0`; observed κ_eff never exceeds 0.37. Either calibrate
   `C_bound` to observed scale or stop advertising a threshold nothing reaches.
4. **Reconsider `trust_r` shrinking with μ.** μ grows when the repair budget is exceeded or
   the Lyapunov derivative goes positive — i.e. when the system is in trouble — and that
   *shrinks* the maximum repair step, which lowers σ_crit exactly when more corrective
   authority is wanted. Whether that is the intended homeostat or an inverted sign is a
   design question the measurements cannot settle, but it should be a deliberate choice.

## 8. What is not claimed

None of this refutes the geometry of the framework, the trust region as a safety
guarantee, or curvature-based diagnostics in general. It says that *in this framework, at
this scale, under injected drift*, the operative failure mode is corrector saturation, and
that a rate ratio describes it while a curvature statistic does not. Whether a curvature
signal becomes informative near σ_crit, or at model scales where the Hessian estimate is
less noisy, is open and worth measuring.
