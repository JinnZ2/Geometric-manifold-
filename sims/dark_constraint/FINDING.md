# FINDING — the K=1 degeneracy is exact; my detection half is not established

Run 2026-08-13T2132Z · 4 load-case counts × 4 dark strengths × 8 seeds · ~5 min · graded
by `run.py` against `REFUTE.md`.

## Verdict: REFUTED as posed — but H1 is the real result and it is clean

| hypothesis | result |
|---|---|
| **H1** the dark constraint is absorbed without trace at K=1 | **holds, to machine precision** |
| **H2** unexplained residual grows with dark strength at K≥2 | **fails — it is flat** |
| **H3** residual separable from the null | **passes trivially, against a degenerate null — see below** |

## H1: a perfect fit to the wrong model

| dark strength | unexplained residual at K=1 | inferred visible ‖r̂‖ | true visible residuals |
|---|---|---|---|
| 0.00 | 0.00e+00 | 0.0000 | 0 |
| 0.02 | 3.22e-14 | 0.0140 | **0** |
| 0.05 | 7.62e-14 | 0.0343 | **0** |
| 0.10 | 1.45e-13 | 0.0674 | **0** |

This is the dark-matter degeneracy reproduced exactly. With one load case the observer's
visible-only model fits the deformation to **machine precision** — there is no residual, no
anomaly, nothing to notice. And the visible residuals it reports are **entirely
fictitious**: the true visible residuals are zero, yet the fit confidently attributes
0.014–0.067 of imbalance to edges that are perfectly healthy, scaling linearly with the
strength of the constraint it cannot see.

For notes 14's drill-down this is sharper than anything in `sims/shape_shadow/`. It is not
that a fault hides — it is that **the instrument reports a specific, confident, wrong
answer**, and its own goodness-of-fit gives no warning. A drill-down conclusion of the form
"edge e's equation is the imbalance source" is unfalsifiable from a single observation,
because an unmodelled constraint elsewhere produces exactly that signature.

The counting argument behind it is exact and needs no experiment to believe once stated:
the octahedron is isostatic, so 12 shape degrees of freedom meet 12 free parameters, and
any deformation whatsoever is representable. The experiment confirms the arithmetic is
not defeated by nonlinearity at these amplitudes.

## H2 and H3: my null was degenerate, so the detection half is unresolved

At K ≥ 2 a residual does appear, but it does **not** scale with dark strength:

| K | dark=0.0 | dark=0.02 | dark=0.05 | dark=0.10 |
|---|---|---|---|---|
| 2 | 0.00e+00 | 2.84e-03 | 2.81e-03 | 2.88e-03 |
| 3 | 0.00e+00 | 4.38e-03 | 4.25e-03 | 4.16e-03 |
| 4 | 0.00e+00 | 5.69e-03 | 5.57e-03 | 5.46e-03 |

Flat in dark strength, growing with K. That is not the signature of a dark component; it
is the signature of something that scales with the number of stacked load cases.

**The null explains why this cannot be resolved as designed.** With `dark = 0` and true
visible residuals also zero, the observed truth *is* the baseline — `y = truth − base = 0`
identically — so the fit is trivial and the residual is exactly `0.00e+00` by construction
rather than by measurement. `NULL.md` required "a residual floor well below the smallest
dark signal"; what it got was a structural zero that measures nothing. H3's separation
therefore passed against a null that could not have failed, and H2's flatness cannot be
attributed between a saturating dark signature and a fit error that a real null would have
exposed.

**The fix, for whoever runs this next:** the null must carry non-zero *visible* residuals,
so the fit is doing real work in both arms and the floor reflects linearization and
convergence error rather than an identity. Until then, the detection half of the claim —
that K ≥ 2 recovers what K = 1 cannot — is **untested**, not refuted.

## What stands

- **The degeneracy at K=1 is established**, and it is the half that matters for the
  drill-down claim: one observation cannot distinguish "these visible equations are
  misbehaving" from "there is a constraint I have not modelled," and the visible residuals
  reported are biased rather than merely incomplete.
- **The cure is indicated but not demonstrated.** More independent probes than parameters
  is the right structural answer — it is what closed the astrophysical case, where no
  single probe sufficed and rotation curves, lensing, the CMB and the Bullet Cluster
  together did. This run does not yet show it working here.

## Unplanned replication at 2x relaxation

A first run at `steps = 3000` was left running in the background while the sim was being
optimised (the Jacobians are independent of dark strength and were being recomputed for
each one). It finished after the graded run at `steps = 1500`, which makes it a free
convergence check on a different config hash. Both result directories are kept.

| quantity | 1500 steps | 3000 steps |
|---|---|---|
| K=1 absorption, dark=0.10 | 1.45e-13 | 1.21e-13 |
| K=1 fictitious ‖r̂‖ | 0.0140 / 0.0343 / 0.0674 | 0.0142 / 0.0345 / 0.0676 |
| K=2 unexplained | 2.84 / 2.81 / 2.88 e-3 | 2.85 / 2.78 / 2.81 e-3 |
| K=4 unexplained | 5.69 / 5.57 / 5.46 e-3 | 6.14 / 5.95 / 5.75 e-3 |

Same verdict, same structure. **H1 is robust** — absorption stays at machine precision and
the fictitious residuals agree to about 1%.

The K ≥ 2 residual also reproduces, agreeing to 2% at K=2 and 8% at K=4, and it is **flat
in dark strength at both step counts**. That matters for the diagnosis above: the flatness
is not an under-relaxation artifact, it is a real and reproducible feature. Note also that
the K=4 residual *rose* slightly with more relaxation rather than falling, which is the
wrong direction for simple under-convergence and is one more reason the degenerate null —
not the integrator — is the thing blocking H2 and H3.

## Limits

- One dark location (the antipodal pair 0–1), one dark *kind* (a brace). A soft coupling,
  a field, or a constraint on a different pair may behave differently.
- Linear fit against a nonlinear truth; at larger deformations the K=1 absorption should
  eventually degrade, and where it does is unmeasured.
- The dark strengths tested (0.02–0.10) all produce absorption at K=1; the amplitude at
  which the degeneracy breaks is not bracketed.
