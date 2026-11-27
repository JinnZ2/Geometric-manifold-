# EXP-FIT-01 results: is repair energy a path property or a step-size property?

Run 2026-11-27. Design: [`../../fisher-information-and-thermodynamics-of-learning.md`](../../fisher-information-and-thermodynamics-of-learning.md) (EXP-FIT-01, issue #46).
Code: [`run.py`](run.py). Raw output: [`raw.json`](raw.json). Toy scale: softmax regression with 12 parameters (Part 1) and ToyLLM with 3,152 parameters (Parts 2–3).

**Verdict.** The two parts come out differently:

- **Parts 1–2: SUPPORTED on the falsification criterion (H0 rejected), but the pre-registered numerical bands were missed in three places, listed below.**
  - `C` is not a path invariant. Sub-sampling the same path by m multiplies `C` by roughly m, between 2.6× and 4.5× at m = 4, never inside the [0.9, 1.1] band that would have falsified H1.
  - The Fisher length `L` changes far less.
- **Part 3 (trend-detector null arm): FALSIFIED.** Multiplying the repair learning rate by √2 did not move `recent_trend()` above 2.0 in any seed (0/10 in both configs). The trust region caps the step size, so the lr change is mostly absorbed.

## Pre-registered prediction (from the design, unchanged)

- **Bands.** `C_m / C_1` should lie in [0.8m, 1.2m] for m ≤ 4 on all seeds, and `L_m / L_1` in [0.9, 1.0].
- **Part 3.** The trend detector should fire within 20 steps of the lr change in ≥ 80% of seeds, and in ≤ 20% of control seeds.
- **Falsified if** `C_m / C_1` stays within [0.9, 1.1], or the lr change does not move the trend ratio.

## Deviations and implementation choices (fixed in `run.py` before running)

1. **Trajectory reproduced in a loop, not through `main.py`.** Part 2 reproduces the `main.py` trajectory by calling `ParameterManifold.repair_step` in a loop, exactly as `simulation/controller.py` does. The data and policy layers in `Controller.run` do not modify θ, so the trajectory is the same. `main.py` itself does not use `RepairEnergyAccumulator`; the energy is recomputed on the saved path.
2. **Env seeds 0–4 (Part 2) and 0–9 (Part 3).** The design did not fix which seeds to use. The repo configs fix `seed: 42`, which is overridden here.
3. **`FisherMetricEstimator.diagonal` seeded per checkpoint.** It samples labels with `torch.multinomial`, so its sampler is seeded per checkpoint index, giving the same G(θ_k) for every m. The exact full Fisher is computed through Jacobian-vector products, which give δᵀFδ exactly without forming the 3,152 × 3,152 matrix.
4. **How Part 3 is scored.** The design said "within 20 steps of the change". This is scored as `recent_trend() > 2.0` at any step in [50, 70). The control arm is the same env seed without the lr change.
5. **Part 1 base resolution.** In Part 1, `C_m / C_1` is read as `C_{160/m} / C_{160}`, so the N = 160 cut is the base (m = 1).

## Environment

Python 3.14.8, numpy 2.5.5, scipy 1.18.1, torch 2.15.0+cpu, on an Intel Xeon @ 2.90 GHz (4 cores, shared with other jobs). Commit `e814887bed510ef0d4bd67c54820399f0ad088d6`. Runtime: Part 1 8 s, Part 2 15 s, Part 3 49 s.

## Results

### Part 1: exact Fisher, curved fixed-endpoint path, 10 seeds

| G | m | C_m / C_1 (range over seeds) | L_m / L_1 (range) |
|---|---|---|---|
| full | 2 | 1.996–2.020 | 0.999–1.004 |
| full | 4 | 3.977–4.120 | 0.997–1.011 |
| full | 8 | 7.883–8.559 | 0.993–1.025 |
| diag | 2 | 1.997–2.008 | 0.999–1.002 |
| diag | 4 | 3.979–4.048 | 0.998–1.005 |
| diag | 8 | 7.898–8.222 | 0.994–1.011 |

### Part 2: ToyLLM repair trajectories, 5 seeds per config

G = repo diagonal (16-sample) or exact full Fisher on the safety inputs.

| config | m | G | C_m / C_1 | L_m / L_1 | Σ2KL_m / Σ2KL_1 |
|---|---|---|---|---|---|
| default | 2 | diag | 1.963–2.073 | 0.993–1.014 | 1.977–1.981 |
| default | 2 | full | 1.940–1.953 | 0.988–0.990 | |
| default | 4 | diag | 3.810–4.512 | 0.981–1.048 | 3.855–3.878 |
| default | 4 | full | 3.635–3.712 | 0.963–0.970 | |
| default | 8 | diag | 7.318–9.466 | 0.951–1.054 | 7.246–7.352 |
| default | 8 | full | 6.242–6.613 | 0.902–0.919 | |
| adversarial | 2 | diag | 1.959–2.025 | 0.991–1.007 | 1.888–1.903 |
| adversarial | 2 | full | 1.729–1.776 | 0.940–0.950 | |
| adversarial | 4 | diag | 3.862–4.009 | 0.980–1.004 | 3.334–3.443 |
| adversarial | 4 | full | 2.589–2.802 | 0.830–0.857 | |
| adversarial | 8 | diag | 7.272–7.920 | 0.948–0.999 | 5.173–5.498 |
| adversarial | 8 | full | 3.096–3.683 | 0.659–0.710 | |

- **Path shape.** The median angle between successive steps was 3.9–5.2°, with a 90th percentile of 5.8–8.7°.
- **Trust-region clipping.** It bound on 47–52% of steps (default) and 100% of steps (adversarial).

### Part 3: lr × √2 at step 50, 10 seeds per arm

| config | fires, lr-change arm | fires, control arm | Fisher exact p (two-sided) | max trend in [50, 70): lr-change / control |
|---|---|---|---|---|
| default | 0/10 | 0/10 | 1.0 | 0.95–1.02 / 0.87–1.00 |
| adversarial | 0/10 | 0/10 | 1.0 | identical in both arms (1.04–1.51) |

Mechanism, read from the saved paths:

- **Default config.** Steps were at the trust radius (0.05) until about step 47 and had shrunk to about 0.047 by step 50. Scaling by √2 pushes them back to the clip, so per-step energy rises by about 1.4× rather than 2×, and only briefly. The trend ratio stays near 1.
- **Adversarial config.** Every step is clipped, so the lr change has no effect at all: both arms give the same series.

### Pre-registration checks, cell by cell

- **`C_m / C_1` in [0.8m, 1.2m] for m ≤ 4.** Met in Part 1 and in Part 2 with the repo's diagonal G. **Missed** with the full Fisher on the adversarial config at m = 4: 2.59–2.80, against a floor of 3.2.
- **`L_m / L_1` in [0.9, 1.0].** **Missed in two ways:**
  - The upper bound was exceeded in Part 1 (up to 1.011 at m = 4) and in Part 2 with the diagonal G (up to 1.048).
  - The lower bound was missed with the full Fisher on the adversarial config at m = 4 (0.83–0.86).
- **H1 falsification test** (`C_m / C_1` within [0.9, 1.1]). Not triggered anywhere; the smallest m = 2 ratio is 1.73.
- **Part 3.** The ≥ 80% firing prediction failed (0%). The falsification condition, that the lr change does not move the trend ratio, is met.

### Side observation (not a pre-registered test; it bears on EXP-FIT-02)

At m = 1, the repo's diagonal 16-sample `C` was 0.09–0.14 of Σ2KL (default) and 0.19–0.31 (adversarial). The exact full-Fisher `C` was 0.93–0.99 of Σ2KL.

## Design notes (2026-11-27, added beside the design, not overwriting it)

- **The L upper bound of 1.0 was wrong as stated.** "Chords are never longer than the path" holds for a fixed metric. Here G is evaluated at the left endpoint of each chord, and with the repo estimator G is also sampled, so `L_m` can exceed `L_1` by a few percent. This is a measurement-convention effect, not evidence against H1.
- **Part 3 did not anticipate trust-region clipping.** The design listed clipping as a threat to Parts 1–2 only. With the repo's trust radius, a learning-rate change is not a clean way to change step size. A follow-up should change the trust radius instead, or apply the √2 scaling to δ after clipping.

## What this establishes, and what it does not

- **Establishes (toy scale).** The repo's "repair energy" `C = Σ δᵀGδ` depends on how finely a fixed path is stepped. It scales roughly with the sub-sampling factor, so it is a step-size-dependent quantity, not a path invariant. The Fisher length changes by at most about 17% at m = 4 (full Fisher, adversarial), and usually by a few percent.
- **Establishes (toy scale).** Under the repo's own trust region, the "spike" detector `recent_trend() > 2.0` did not react to a √2 learning-rate change at step 50. The FIT-G2 worry, that a ×√2 step change would trip the detector, did not materialise here, because the trust region absorbs it.
- **Does not establish** that the trend detector is a valid early warning (EXP-FIT-03, not run). It also does not establish anything about physical energy, or about models other than ToyLLM.
