# FINDING — basin escape here is rate-induced, and the critical rate is predictable

Run 2026-08-13T2053Z · 8 drift rates × 5 seeds + calibration + null arm, 25 steps each ·
~38 s · graded by `run.py` against `REFUTE.md`.

## Verdict: SUPPORTED

A parameter-free prediction was made and held:

- predicted **σ_crit = √(repair_cap / k) = 0.00120**
- measured **σ_crit = 0.00084** (zero crossing of net ΔKL/step)
- **ratio 0.70×**, inside the pre-committed 2× tolerance
- null arm net ΔKL/step = −0.000021 ≤ 0, so the repair does hold an unforced basin
- net ΔKL/step monotone increasing in σ across the full sweep

`k` and `repair_cap` were both fixed from a single calibration run at σ = 0.004, so the
crossing point was predicted before the sweep and could have been wrong by any amount.

## The measurement

| σ | drift ΔKL/step | repair ΔKL/step | net |
|---|---|---|---|
| 0.0005 | +0.000012 | −0.000024 | **−0.000012** |
| 0.001 | +0.000031 | −0.000023 | +0.000008 |
| 0.0015 | +0.000057 | −0.000023 | +0.000035 |
| 0.002 | +0.000092 | −0.000023 | +0.000069 |
| 0.003 | +0.000184 | −0.000025 | +0.000159 |
| 0.004 | +0.000308 | −0.000028 | +0.000280 |
| 0.006 | +0.000654 | −0.000028 | +0.000625 |
| 0.012 | +0.002620 | −0.000033 | +0.002587 |

**The repair column is flat across a 24× range of forcing.** That is the whole finding.
`CoupledDynamicalSystem` caps each repair step at `trust_r = lr/(1 + μ·max(fisher)) ≤ 0.01`,
so the corrector has a ceiling; the drift term grows as σ². A capped corrector against
quadratic forcing has a crossover, and that crossover is what decides escape — not any
property of the landscape's curvature.

## What it revises

The three rounds in `sims/kappa_eff_leading/` ran at σ ∈ {0.012, 0.016, 0.020} — **10–24×
above σ_crit**. There the system is committed from step 0, net ΔKL/step exceeds the repair
ceiling by two orders of magnitude, and the outcome carries no information beyond the rate
inequality. No indicator of any kind can lead in that regime, because nothing is
approaching anything.

So the earlier work fairly refutes *"κ_eff works in the committed regime"* and does **not**
test *"κ_eff works near the critical point."* `REFUTE.md` said so before this run precisely
so a SUPPORTED verdict would be allowed to weaken the earlier conclusion. The honest status
of κ_eff is now: **refuted in the committed regime, untested near σ_crit.**

## Cross-domain reading

The full synthesis is in `docs/research/DOMAIN_PHYSICS.md`. In short: this is Ashby's
requisite variety, MCPM's drag ratio L/A, R-tipping, actuator saturation, and Kramers
escape — five domains, one object, all of them comparing a disturbance rate to a maximum
correction rate, none of them measuring curvature. The saturation margin
ρ = drift ΔKL/step ÷ repair cap is literally L/A applied here, it is free to compute, and
it predicted the critical rate to 0.70×.

## Caveats

- **One initial condition.** Every arm starts at `drift_strength = 0.08`. σ_crit depends on
  local KL geometry through `k`, so starting nearer or further from θ_ref will move it. The
  *form* of the prediction should survive; the number is position-specific.
- **Short runs by design.** 25 steps measures a rate balance, not a breach. That is the
  point — escape is decided by the sign of net ΔΚL/step, and waiting for a crossing only
  measures how long the arithmetic takes — but it does mean slow transient effects near
  σ_crit are not captured.
- **Sub-quadratic at the low end.** The drift term is cleanly quadratic in the upper half
  of the sweep (0.006 → 0.012 gives exactly 4.0×) but runs sub-quadratic at the smallest
  σ, where the per-step signal approaches the measurement floor. That is the most likely
  source of the 0.70× discrepancy, and it is why the tolerance was set at 2× rather than
  something tighter.
