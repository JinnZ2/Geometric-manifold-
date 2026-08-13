# REFUTE — is basin escape rate-induced rather than curvature-induced?

**Written before running.** Thresholds authored here; there is no prior pre-registration
to transcribe, and the prediction being tested was derived in this repo (see below).

## Where this came from

Three rounds of `sims/kappa_eff_leading/` refuted κ_eff as a leading indicator. Inspecting
*why* turned up something that reframes all three: `CoupledDynamicalSystem` hard-caps each
repair step at `trust_r = lr / (1 + mu * max(fisher)) <= lr = 0.01`, and measured in the
basin coordinate the repair removes a **constant ~0.00002 of KL per step** regardless of
how hard the system is being driven, while injected drift adds KL in proportion to
**sigma^2**:

| sigma | drift dKL/step | repair dKL/step | repair counters |
|---|---|---|---|
| 0.004 | +0.00011 | -0.00002 | 17% |
| 0.008 | +0.00055 | -0.00002 | 3% |
| 0.012 | +0.00136 | -0.00003 | 2% |

A capped corrector against a forcing that grows quadratically is not a curvature problem.
It is **control saturation**, and it makes a parameter-free prediction.

## The claim under test

**Basin escape here is rate-induced.** There is a critical drift rate

    sigma_crit = sqrt(repair_cap / k),      where  dKL_drift = k * sigma^2

below which net dKL/step is negative (the repair holds the basin) and above which it is
positive (escape, at a speed set by the excess). `k` and `repair_cap` are both measured
from a single calibration run at one sigma, so the predicted crossing point has **zero
free parameters** and can be wrong.

## Pass condition

- The measured critical rate — the zero crossing of net dKL/step against sigma — falls
  within a factor of 2 of the prediction.
- Net dKL/step is monotone increasing in sigma.
- The null arm (no drift) has net dKL/step <= 0.

## Refutation condition

Any of: the crossing is off by more than 2x; net dKL/step is not monotone in sigma; or the
null shows positive net dKL/step. The first would mean the saturation balance is not what
sets the transition. The second would mean the sigma^2 scaling is wrong. The third would
mean the repair loop cannot hold the basin even unforced, and nothing else here is
interpretable.

## What this would mean for kappa_eff, either way

If supported, the earlier refutation gains a mechanism and loses some of its force. The
three previous rounds all ran at sigma in {0.012, 0.016, 0.020} — which, if the prediction
holds near 0.002, is **6-10x above the critical rate**. In that regime the system is
committed from step 0 and there is no approach to a threshold for *any* indicator to
detect. That would make those runs a fair test of "does kappa_eff work in the committed
regime" (it does not) and **not** a test of "does kappa_eff work near the critical point",
which is the claim that actually matters and which has not yet been tested.

Stating that in advance: a SUPPORTED verdict here partially undercuts my own earlier
conclusion, and the correct response is to re-run the indicator comparison near sigma_crit
rather than to leave the stronger reading standing.
