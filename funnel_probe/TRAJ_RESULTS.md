# funnel_probe — funnel width ALONG the trajectory (step 1 follow-on)

Pre-registration: `PREREGISTRATION_TRAJ.md` (committed alone, identical in the
four repos). Script `funnel_traj_width.py` (stdlib; imports the vendored
reference unchanged). Comparators: step 1's `samples/funnel_noise_results.json`,
not re-run. Output: `samples/funnel_traj_width.sample.txt`,
`samples/funnel_traj_width_results.json`. Identical text in the four repos,
since the reference, the starts and the step-1 comparators are byte-identical
in all four. 1703 s, one process.

## What did not hold, first

    T2  sigma_half_pred(a) within 1.5x of measured     FAILED on x0 = 1e-8   (ratios 0.841, 0.788, 0.616)
    T3  sigma_half(a) / delta_a*(0) in [3, 6]           FAILED on x0 = 1e-8   (4.91, 5.24, 6.70)
    T4  model (b): never halves AND sigma_75 within 1.5x FAILED on all three   (ratios 1.56, 1.41, 1.75;
                                                                               start 2 crosses 0.5 once)

    OUTCOME: BOTTLENECK_AT_START ; LINEAR_RESPONSE_FAILS

Both failure directions are recorded because they are opposite. Model (a):
the measured threshold is LARGER than the first-order prediction — the
system is more robust to x-noise than linear response says — and the gap
grows as the start moves toward the edge (0.84 -> 0.79 -> 0.62). Model (b):
the measured threshold is SMALLER than predicted — less robust to mu-noise
than linear response says — by 1.4-1.75x on every start. One first-order
model, two signs of error, so the failure is not a scale factor.

T4's "never halves" half: start 2's step-1 curve reads P(e0) = 0.455 at
sigma = 1 and the log interpolation puts a 0.5-crossing at sigma = 0.61.
The Wilson interval at that point is [0.387, 0.524] and contains 0.5, so
the crossing is within the step-1 noise; the prediction fails as written
and the asymptote 0.5 is not contradicted by the data. The other two starts
sit at 0.52 and 0.53 at sigma = 1 and never cross.

## What held

    T1  delta_a*(t) minimal at t = 0, monotone over t <= 1      HELD on all three
    T5  delta_a*(0) = edge_mu(0) - x0 within 10%                 HELD (ratio 1.000 on all three)

## The measurement

    start x0      t to e0    delta_a*(0)    I_a          share of I_a    delta_b*(0)   I_b     share of I_b
                                                         from t<=0.25                           t<=1  t<=5  t<=10
    1e-9          27.42      4.942e-8       5.29e13      0.854           0.575         19.5    0.15  0.60  0.87
    10^-8.5       29.91      4.726e-8       5.78e13      0.854           0.409         37.1    0.15  0.61  0.87
    1e-8          37.28      4.042e-8       7.90e13      0.854           0.240        109.1    0.15  0.60  0.84

    delta_a*(t): 4.9e-8 at t=0 -> 1.0e-7 at 0.2 -> 1.9e-6 at 1.0 -> 4.6e-5 at 2.0 -> no kick <= 10 flips from t >= 8
    delta_b*(t): 0.58 at t=0 -> 0.78 at 5 -> 1.31 at 10 -> 2.35 at 15 -> 3.74 at 20   (x0 = 1e-9)

### Answer to the order's question

The start-time edge IS the bottleneck for noise on x, and is NOT the
bottleneck for noise on mu; the question splits by channel.

Model (a), noise on x: delta_a*(t) grows as the trajectory's own gain
x(t)/x0 (1.0 -> 2.2 at t=0.2 -> 41 at t=1 -> 970 at t=2), so 1/delta_a*^2
collapses and 85.4% of the first-order integral I_a sits in t <= 0.25 on
every start (the same 0.854 three times because the linear regime is
self-similar in x0). The 5x: sigma_half(a) / delta_a*(0) = 4.91 at x0=1e-9,
against the analytic 1.4826 sqrt(2 mu0) = 4.19 — the noise competing with
the edge is integrated over ~1/(2 mu0) time units and both signs count after
reflection. The ratio rises to 6.70 at x0 = 1e-8 (edge - x0 = 4.0e-8 against
x = 1e-8, so the kick that matters is 4x the state and first-order in it is
not small), which is where T2 and T3 fail together.

Model (b), noise on mu: delta_b*(t) is minimal at t = 0 but grows only
2.3x over the first 10 time units (a mu kick is integrated by the slow
variable; its leverage decays with eps, not with the gain of x), so I_b
accumulates over the whole approach to the fold: 15% by t = 1, 60% by t = 5,
87% by t = 10. The sensitivity is one-sided (a negative mu kick lowers mu
and helps e0), which is why the step-1 curve asymptotes at 0.5 instead of
halving — predicted and seen on two of three starts, within noise on the
third. The linear prediction for sigma_75 is 1.4-1.75x too high: the
accumulated mu displacement that flips the outcome is not a Gaussian sum
of independent small kicks when the kicks act on a variable whose own
drift is eps(-mu + 3x - 2) and the outcome is decided by a fold.

Scope: deterministic reference trajectories plus a first-order noise model;
one parameter set (a=3, b=2, eps=0.1, mu0=4); bisection on log delta to 50
iterations; kicks above 10 not searched (`none`); the measured comparators
are step 1's and were not re-run. Nothing here changes a step-1 number.
