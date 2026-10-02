# PRE-REGISTRATION — funnel width ALONG the trajectory (step 1 follow-on)

Order (Kavik via Claude, 2026-10-02): "noise: why 5x more robust? Measure
the funnel's width ALONG the trajectory, not just at t = 0. The start-time
edge may not be the bottleneck." Committed before `funnel_traj_width.py`
exists. Identical text in the four repos, as step 1 was. Reference
`singular_funnel_pitchfork.py` unchanged (sha256
466629a741722183a09221a1594c8f75db7d922ce00eede3afd8e7f01c1f9c46).

## Measurement

Deterministic reference trajectory (RK4 on y = ln x, dt = 0.01, the
reference's integrator) from each step-1 start, mu0 = 4, eps = 0.1,
x0 in {1e-9, 10^-8.5, 1e-8}, run to the reference's own e0 exit
(mu < 0 and x < 1e-8). At times t_k on the grid

    t_k = 0, 0.05, ..., 3.0 (step 0.05), then 3.5, 4, ..., until e0

report, per start:

    x(t_k), mu(t_k)
    edge_mu(t_k)   = funnel_edge(mu0 = mu(t_k), eps)  -- the reference's own
                     bisection, started fresh at that mu: the funnel's width
                     in x at that mu, independent of the trajectory
    delta_a*(t_k)  = smallest additive kick delta > 0 applied to x at t_k
                     (x -> x + delta, deterministic thereafter) that flips
                     the outcome e0 -> e2; bisection on log delta over
                     [1e-14, 10], 60 iterations
    delta_b*(t_k)  = the same for a kick on mu (mu -> mu + delta)

Both kick profiles are then integrated into a linear-response threshold:

    I_a = integral dt / delta_a*(t)^2        (trapezoid on the t_k grid)
    I_b = integral dt / delta_b*(t)^2

Under additive white noise of strength sigma, the accumulated outcome-
relevant displacement is Z = sigma * sum_i xi_i sqrt(dt) / delta*(t_i),
so Z ~ N(0, sigma^2 I) to first order, and the outcome flips when |Z| > 1
(model a: a negative kick is REFLECTED at x = 0 and, for |delta| >> x,
acts as a positive kick of the same size, so both signs flip) or when
Z > 1 (model b: a negative mu kick lowers mu and HELPS e0, so only one
sign flips).

    model (a)  P(e0) ~ P(|N(0,1)| < 1/(sigma sqrt I_a))
               P = 0.5 at  sigma_half_pred(a) = 1 / (0.6745 sqrt I_a)
    model (b)  P(e0) ~ Phi(1/(sigma sqrt I_b)) -> 0.5 as sigma -> inf
               never halves; P = 0.75 at sigma_75_pred(b) = 1 / (0.6745 sqrt I_b)

Measured comparators come from step 1's `samples/funnel_noise_results.json`
unchanged: sigma_half(a) per start (2.42e-7, 2.47e-7, 2.71e-7) and, for
(b), the log-interpolated sigma at which P(e0) first falls to <= 0.75.

## Predictions (fixed now)

    T1  bottleneck AT THE START: delta_a*(t) is minimal at t = 0 and
        increases monotonically over the first 1.0 time units (no interior
        minimum). Falsifier: an interior minimum of delta_a* -> the
        bottleneck is later than the start-time edge.
    T2  sigma_half_pred(a) within a factor 1.5 of the measured sigma_half(a),
        for each of the three starts.
    T3  the "5x": sigma_half(a) / delta_a*(0) in [3, 6]. The mechanism
        offered now: the fast growth exp(2 mu0 t) weights the noise toward
        the first ~1/(2 mu0) = 0.125 time units, so the integrated noise
        that competes with the edge is sigma / sqrt(2 mu0), not sigma; and
        the reflection makes both signs count. Analytic form: ratio ~
        1.4826 sqrt(2 mu0) = 4.2.
    T4  model (b): the step-1 curve never halves because the asymptote is
        0.5 (one-sided sensitivity); sigma_75_pred(b) within a factor 1.5
        of the measured sigma_75(b) for each start where the grid crosses
        0.75.
    T5  delta_a*(0) within 10% of edge_mu(0) - x0 (the kick that reaches
        the t = 0 edge is the t = 0 edge).

Outcome vocabulary for this step: BOTTLENECK_AT_START / BOTTLENECK_INTERIOR
(from T1); LINEAR_RESPONSE_HOLDS / LINEAR_RESPONSE_FAILS (from T2 and T4
together, factor 1.5). No parameter is retuned after a result.

Scope: deterministic trajectories plus a first-order noise model; one
parameter set (a=3, b=2, eps=0.1, mu0=4); the measured comparators are
step 1's, not re-run.
