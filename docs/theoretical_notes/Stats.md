moment eqs under Gaussian closure, symmetric across active dims:

  ṁ   = -m³ + a·m - 3·m·σ²
  σ̇²  = 2σ²·(-3m² + a - 3σ² - c) + 2(σ_noise² + D_mut)

fixed points (per active dim k, symmetric):

  diverse:      m* = 0,       σ²* solves  -a + 3σ²* + c = (σ_n² + D_mut)/σ²*
  homogeneous:  m* = ±√(a-3σ²*),  σ²* small
  saddle:       intermediate root of the coupled system

ΔΦ = Φ(saddle) - Φ(diverse), where Φ is the mean-field potential
whose gradient gives (ṁ, σ̇²).
