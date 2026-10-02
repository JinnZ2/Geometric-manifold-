# Step 1 result — reference + noise (Geometric-manifold- copy; identical in the four repos)

Pre-registration: `PREREGISTRATION.md`, commit fde6c0e (this repo).
Run: `python3 funnel_probe/funnel_noise.py` in a temp directory, 200 runs per
point, seeds as pre-registered. Raw output: `samples/funnel_noise_results.json`.

## What did not hold, first

    P3  FAILED  model (a) sigma_half predicted in [1e-9, 1e-7]; measured
                2.42e-7, 2.47e-7, 2.71e-7 — above the range, and ~5x ABOVE the
                deterministic funnel edge at the start (10^-7.30 = 5.0e-8).
    P4  FAILED  model (b) sigma_half predicted in [1e-2, 1]; P(e0) never
                halves inside the grid. At sigma = 1 (the top of the grid)
                P(e0) is still 0.52 / 0.46 / 0.53.
    P5  FAILED  the ratio sigma_half(b)/sigma_half(a) is unmeasured because
                (b) never halves; the one (b) start that crosses
                (x0 = 10^-8.5, 0.61) gives a ratio of 2.5e6 > 1e4.
    P6  FAILED  as a conjunction: holds for (a) (spread factor 1.12), not
                evaluable for (b) (two of three starts never halve).

What held: P1 (P(e0) = 1.000 at the smallest sigma, every start, both models
— the deterministic funnel reproduced under the declared exits) and P2
(P(e0) non-increasing in sigma at every start, both models).

## Outcome enum

    model (a) additive on x, reflecting:   FUNNEL_FOUND
              sigma_half = 2.42e-7 (x0=1e-9), 2.47e-7 (10^-8.5), 2.71e-7 (1e-8)
    model (b) additive on mu:              NOT_FOUND_IN_RANGE
              searched sigma in [1e-4, 1e0]; P(e0) at sigma = 1: 0.52, 0.46, 0.53
              (one start, 10^-8.5, interpolates to 0.61 by the halving rule;
              the other two never reach half)
    INCONCLUSIVE fraction: 0 at every grid point (no run hit T_max)

## P(e0), 200 runs per cell

    model (a)   log10 sigma  -11  -10.5  -10  -9.5   -9  -8.5   -8  -7.5    -7   -6.5    -6
      x0 = 1e-9                1     1    1    1     1    1     1    1   0.950  0.365  0.130
      x0 = 10^-8.5             1     1    1    1     1    1     1    1   0.925  0.385  0.155
      x0 = 1e-8                1     1    1    1     1    1     1    1   0.885  0.440  0.135

    model (b)   log10 sigma   -4  -3.5   -3  -2.5   -2  -1.5    -1   -0.5     0
      x0 = 1e-9                1     1    1    1     1    1   1.000  0.625  0.520
      x0 = 10^-8.5             1     1    1    1     1    1   0.920  0.560  0.455
      x0 = 1e-8                1     1    1    1     1    1   0.695  0.510  0.530

## Reading (interpretation, not a pre-registered result)

The funnel is more robust to noise than its width at the start suggests, in
both models. The edge x0*(mu) is 5e-8 at mu = 4 but grows to ~1e-5 by the
time mu has decayed to 3 and to ~1e-2 by mu = 2 (RUN 1 of the reference
script), so the window in which an x-kick of ~1e-7 is fatal lasts only a few
time units, and reflecting noise of 2.4e-7 per sqrt(time) has not yet
accumulated past the moving edge when it closes. Under mu-noise the margin is
~3.9 nats in ln x integrated over the dwell time, and it takes sigma of order
1 — the same size as mu's own fixed-point separation — to erase it. Whether
P(e0) halves for model (b) at sigma > 1 is not measured here; a wider grid
needs a new pre-registration and was not run.

Scope: noisy (Euler-Maruyama, dt = 0.01, T_max = 600), one parameter set
(a=3, b=2, eps=0.1, mu0=4), continuous phase space, the paper's own system.
Nothing here is a statement about this repository's engine.
