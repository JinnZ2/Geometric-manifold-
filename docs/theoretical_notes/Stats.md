moment eqs under Gaussian closure, symmetric across active dims:

  ṁ   = -m³ + a·m - 3·m·σ²
  σ̇²  = 2σ²·(-3m² + a - 3σ² - c) + 2(σ_noise² + D_mut)

fixed points (per active dim k, symmetric):

  diverse:      m* = 0,       σ²* solves  -a + 3σ²* + c = (σ_n² + D_mut)/σ²*
  homogeneous:  m* = ±√(a-3σ²*),  σ²* small
  saddle:       intermediate root of the coupled system

ΔΦ = Φ(saddle) - Φ(diverse), where Φ is the mean-field potential
whose gradient gives (ṁ, σ̇²).




falsifies C5 if |skew| or |excess kurt| > ~0.5 while D > 10·collapse_ε.
that means the mean-field saddle is in the wrong place and
ΔΦ from mf_potential is biased. fallback: estimate ΔΦ from
agent-sim histogram via Boltzmann inversion.


acceptance:
  dt: extrapolate ⟨T⟩(dt → 0); choose dt where bias < SEM.
  N : log⟨T⟩ vs p slope must be N-independent within SEM.


LockedHighDimDiversityModel.step() — replace the mutation noise
block with a single draw broadcast to all p active dims, and same
for repair shock. (the fix you already had in the doc, just ensure
it's the one actually used in run_experiments.)


1. mean_field.find_fixed_points + classify          → confirm saddle exists
2. mf_potential(diverse, saddle)                    → ΔΦ(p, c)
3. solve for c(p) such that ΔΦ(p, c(p)) = ΔΦ_target → proper calibration
   (replaces calibrate_c_for_p heuristic entirely)
4. measure_lambda_plus_sim vs lambda_plus_at_saddle → moment-closure check
5. closure_trajectory during collapse runs          → validate / falsify C5
6. dt_sweep + N_sweep at p=2                        → numerical baseline
7. main p-sweep: collapse + repair, independent + locked
8. fit log⟨T⟩ = α - (p/2)·log(λ₊_measured) + ΔΦ/ε  → test C2 directly



mean_field2.py
verification before building anything else:
  run __main__. expected pattern:
    low c  → diverse stable, homog also stable, saddle exists
    high c → diverse may disappear (bifurcation)
    λ+ at saddle is real positive scalar O(a)
  if fsolve misses the saddle → widen seeds; the saddle has
  small |m| and intermediate s2.



if predictions hold:

  - dense s_H is NOT a low-energy configuration. it's a
    metastable high-cost state requiring continuous Φ_ext
    input. confirms direction of E1 from prior message.
    
  - the "messy middle" of s_H is where systems are least
    predictable. AI safety regimes operating in this range
    are not "partial progress" — they're maximum-variance.
    
  - reciprocated violence and preemptive violence have a
    specific dynamical signature distinguishable from
    "moral failure" or "cultural transmission":
      reactive violence    = phase B-early/mid response
      preemptive violence  = phase C signature, with rate
                             governed by dR/dt vs λ_+
    
  - the irreversibility (P10.4) predicts a measurable
    fact: that restoring external pressure removal does
    not restore the original regime. only network 
    restoration does. this matches observation but is 
    typically attributed to "trauma" or "cultural change."
    the model attributes it to network topology with no 
    psychological mechanism required.
    
  - rate sensitivity (P10.6) is the strongest practical
    prediction: SLOW transitions cause less preemptive
    behavior than FAST transitions at the same total
    perturbation. applies to AI deployment, policy
    rollouts, ecosystem management.

if predictions fail:

  - if harm tracks Φ_ext magnitude more than dR/dt: the
    mechanism is external-pressure-driven, not network-
    degradation-driven. the model needs revision.
    
  - if preemptive_rate appears immediately (no lag):
    the saddle-crossing picture is wrong; something
    discontinuous is happening.
    
  - if substrate-independence fails: the geometric model
    is not capturing the right level of abstraction.
    fall back to substrate-specific modeling.
