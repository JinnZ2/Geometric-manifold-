# EXP-LLG-03 results: coordinate dependence of κ_eff under an exact symmetry

Run 2026-11-27. Design: [`../../loss-landscape-geometry-and-basin-stability.md`](../../loss-landscape-geometry-and-basin-stability.md) (EXP-LLG-03, issue #56).
Code: [`run.py`](run.py). Raw output: [`raw.json`](raw.json). Toy scale: ToyLLM, 3,152 parameters.

**Verdict: SUPPORTED (H1).** κ_eff changes under a rescaling that leaves the function unchanged, on 20 of 20 seeds at every α ≠ 1. In arm 2 the pass/fail outcome of the repo's existing κ_eff leading-indicator test also changes with the coordinates.

## Pre-registered prediction (from the design, unchanged)

- Max logit difference < 1e−4 at every α.
- κ(α)/κ(1) outside [0.95, 1.05] for α ≠ 1 on ≥ 90% of seeds.
- Falsified if κ_eff is within 5% at all α on ≥ 90% of seeds.
- Arm 2: rerun the κ_eff-leading criterion sweep on rescaled runs (α = 0.5, 2) to see whether its verdict changes. No direction was predicted.

## Deviations and implementation choices (fixed in `run.py` before running)

1. **Arm 1 setup.** Arm 1 uses the sketch's `setup()`: random θ_ref = 0.1·N(0, I), a random drift direction and random inputs per seed. It does not use the `Environment` θ_ref. This matches the design's "20 seeds of (θ_ref, direction, inputs)".
2. **Arm 2 runs each trajectory once.** Each trajectory runs once, in the original coordinates, with the repo's `CoupledDynamicalSystem`. Along it, κ_eff and the θ-distance baseline are re-evaluated in α-coordinates: θ, θ_ref and θ̇ are all pushed forward by R_α.
   - Gradient descent is not invariant under reparameterisation. Running the dynamics in rescaled coordinates would give a different trajectory, and would answer a different question (what the optimiser does, rather than what the monitor reads).
   - With one trajectory, the function, `basin_kl` and the breach steps are identical across α. Only the coordinates differ.
3. **Grading re-implemented.** The arm 2 grading rule is a re-implementation of `sims/kappa_eff_leading/run.py`. It reuses that file's `alarm_step` and `smooth_causal_median` (imported read-only) and keeps its config unchanged: 5 seeds, σ ∈ {0.012, 0.016, 0.02}, 2 modes, 21 (criterion, window) pairs, and a null arm with no drift. The C1 "smoothing reduces false alarms" check was not re-implemented.
4. **α values are powers of two.** Multiplying by them is exact in floating point, so the logit difference is exactly 0 rather than merely < 1e−4.

## Environment

Python 3.14.8, numpy 2.5.5, scipy 1.18.1, torch 2.15.0+cpu, on an Intel Xeon @ 2.90 GHz (4 cores, shared with other jobs). Commit `e814887bed510ef0d4bd67c54820399f0ad088d6`. Runtime: arm 1 0.5 s, arm 2 38 s.

## Results

### Arm 1: κ(α)/κ(1) over 20 seeds

| α | min | median | max | seeds outside [0.95, 1.05] | max \|Δlogit\| |
|---|---|---|---|---|---|
| 0.25 | 0.179 | 0.191 | 0.207 | 20/20 | 0 |
| 0.5 | 0.643 | 0.679 | 0.728 | 20/20 | 0 |
| 2 | 0.349 | 0.361 | 0.372 | 20/20 | 0 |
| 4 | 0.090 | 0.093 | 0.096 | 20/20 | 0 |

- **Sign.** The signed Rayleigh quotient was positive in 100 of 100 (seed, α) rows, so the `abs()` in `effective_repair_curvature` changed nothing in this arm.
- **KL.** KL to the reference is identical across α (difference 0).

### Arm 2: the κ_eff-leading test in α-coordinates

The α = 1 positive control reproduces the repo's per-step κ_eff to within 9.8e−7 (maximum absolute difference). Its verdict and viable count also match the committed ledger entry `sims/kappa_eff_leading/results/2026-08-13T2041Z`: REFUTED, with 4 viable criteria.

| α | viable (criterion, window) pairs | verdict under the sim's own rule | max fraction of cells supporting Theory A, any pair |
|---|---|---|---|
| 0.5 | 5 of 21 | REFUTED | 0.56 |
| 1 (control) | 4 of 21 | REFUTED | 0.37 |
| 2 | 0 of 21 | INCONCLUSIVE (nothing left to grade) | 0.59 |

- **Tracked κ ratio.** Along the drift trajectories, κ(α)/κ(1) per step ranged from 0.25 to 4.0 (median 0.50 at α = 0.5 and 0.45 at α = 2).
- **What decides viability.** It is the null-arm false-alarm rate, which comes from 5 null runs, so it moves in steps of 0.2. At α = 2, κ_eff's false-alarm rate rose above 0.2 for every criterion; for example `ratio_3.0` at window 1 went from 0.2 at α = 1 to 0.6 at α = 2.

## What this establishes, and what it does not

- **Establishes (toy scale).** κ_eff as implemented is a coordinate quantity, not a function-space quantity. Rescalings that leave every logit, the KL and the basin unchanged move it by factors of 0.09 to 0.73 in arm 1.
- **Establishes (toy scale).** The outcome of the repo's existing κ_eff test (REFUTED / INCONCLUSIVE) depends on the parameterisation as well as on the dynamics. The REFUTED verdict at α = 1 does not carry over unchanged to equivalent coordinates.
- **Does not establish** that κ_eff is useless in a fixed parameterisation. The design itself noted that a fixed parameterisation in practice may make the dependence harmless; usefulness in fixed coordinates is the question for EXP-LLG-01.
- **Does not establish** anything about larger models, or about rescalings other than the ReLU one.
- **Arm 2 caveat.** The arm 2 viability flips rest on a 5-run null arm. They show that the verdict is fragile under reparameterisation, not that any particular α gives the right verdict.
