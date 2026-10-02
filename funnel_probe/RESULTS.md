# funnel_probe — results (Geometric-manifold-)

Pre-registration: commit fde6c0e (`PREREGISTRATION.md`). Step 1 (noise):
`NOISE_RESULTS.md`, commit 869030a. This file: step 1 of the repo-specific
part (timescale separation) and the stop decision. Run from a temp
directory: `python3 funnel_probe/timescale_check.py`; trimmed output in
`samples/timescale_check_results.json`.

## What did not hold, first

Nothing in the pre-registered rule was contradicted; the rule fired on its
first branch. One environment fact changed between pre-registration and run:
torch was recorded as not installable and then installed from PyPI (2.14.1),
so measurement 2 (the repo's own simulation) ran in addition to the numpy
twin. The rule and the grid are unchanged.

One observation outside the order's question, recorded because it is in the
same output: over the 100 default steps the repair step moved theta AWAY
from the reference — distance 17.08 -> 18.53, KL to the reference
3.79 -> 39.37, every step at the trust-radius cap. The saddle objective's
minus sign (`task - lambda*safety`, documented as intentional in CLAUDE.md)
makes the step ascend the KL. This is the question `sims/objective_sign/`
already asks; it is not adjudicated here.

## Outcome: NOT_IN_CLASS

    in-loop drift applications per step        0
    repair rate, mean ||delta|| per step        0.050   (cap 0.05; 100% of steps at the cap)
    ratio drift / repair                        0
    rule branch                                 drift_rate_in_loop == 0 -> NOT_IN_CLASS, stop

Why: `theta_drifted = theta_ref + drift_strength * randn` is computed once in
`Environment._setup`. Inside `Controller.run`'s step loop (AST walk,
`code_reading()`), the only assignment to `theta` is the return of
`repair_step`; there is no `randn`, no drift term, no slow variable that
enters `repair_step` as a parameter. Drift is an initial condition and
repair is the only process, so there is one timescale. The paper's class
needs a second one (mu, rate eps) coupled to the fast state.

Also read, not run: `addon_thermodynamic_control/stability.py` uses
`sigma_drift` only inside the proactive penalty's six-sample expectation
(lines 425-438), never to move theta; `sims/rate_induced_escape/` injects
white-noise drift per step in an experiment, and white noise has no eps.

## Measurements (default.yaml, 100 steps)

                                      repo (torch, seed 42)   numpy twin (seed 42, own init)
    initial displacement ||theta_d - theta_ref||     17.070                 16.925
    mean ||delta|| per step                           0.050                  0.050
    steps at the trust-radius cap                     100 / 100              100 / 100
    displacement / repair rate (steps to close)       341                    339
    distance to reference, first -> last              17.08 -> 18.53         16.93 -> 18.43
    KL to reference, first -> last                    3.79 -> 39.37          3.78 -> 39.53
    twin gradient vs torch autograd, same theta       rel err 5.3e-16

The twin is not the same bytes as the repo (different RNG for the init), and
the two agree on every rate to two figures; it is kept because the
pre-registration named it and because it needs no torch.

## Step 2 — NOT RUN (superseded below after the sign fix; left as recorded)

Per the order ("If none -> NOT_IN_CLASS, stop") and the pre-registration.
Stated for the record: with repair off this simulation has no autonomous
flow at all, so every start on the pre-registered 2D slice would settle
where it began, and both KL-ball counts (outside points that return, inside
points that leave) would be 0 of N by construction — a property of the
engine, not a measurement of a basin.

Scope: deterministic; default.yaml (drift_strength 0.3, trust_radius 0.05,
asymmetry_lambda 10, curvature_weight 2, lr 0.01, 100 steps, seed 42);
continuous parameter space.

## Step 2 (literal) and step 2b — RUN after the sign fix

Pre-registration: `PREREGISTRATION_FIX.md` (commit eaf0b14), run on the
fixed objective (`task + lambda*safety`, commit 59031bc; V1-V3 in
`samples/fix/`). Script `slice_step2.py`; output in `samples/slice/`.
Slice: d = unit drift direction, o = orthogonalised Gaussian (seed 1),
theta = theta_ref + s d + u o; dense s,u in [-2R, 2R] (41 x 41 = 1681,
R = 17.07) plus a log set (s = +-10^k, k = -6..1 by half-decades; u in
{0, +-10^k}; 930) = 2611 starts; 100 repair steps each; KL ball epsilon
0.1 to theta_ref; 232 s single-threaded.

### What did not hold, first

Nothing pre-registered was contradicted. One pre-registered check turned
out to be vacuous rather than passed: S2 ("every outside-ball start within
reach 4.0 is REPAIRED") has an empty antecedent. Every start within 4.0 of
theta_ref is ALREADY inside the KL ball at t=0 (the ball's radius in
parameter distance is about 4.5 on this slice), so 0 of 0. The check is
HELD by construction and says nothing about repair.

### Step 2 literal (repair OFF)

    starts inside the KL ball at t=0     843 of 2611
    starts outside                      1768
    outside that return on their own       0
    inside that leave                      0

Degenerate as predicted in the original Step 2 section above: with repair
off there is no flow.

### Step 2b (repair ON, corrected sign)

    class        n
    HELD        843     inside the ball at t=0, still inside at t=100
    REPAIRED    302     outside -> inside
    STRANDED   1466     outside -> outside
    EXPELLED      0     inside -> outside

    band [dist0)   n   REPAIRED  STRANDED  HELD   KL1 median
    [ 0,  1)     601         0         0   601    0.0036
    [ 1,  2)     106         0         0   106    0.0045
    [ 2,  3)       4         0         0     4    0.0080
    [ 3,  4)     118         0         0   118    0.0114
    [ 4,  5)       8         0         0     8    0.0180
    [ 5, 10)      84        78         0     6    0.0371
    [10, 20)     446       224       222     0    0.0996
    [20,100)    1244         0      1244     0    2.6202

    S1 EXPELLED = 0                         HELD (0)
    S2 reach band all REPAIRED              HELD, vacuous (0 of 0)
    S3 KL1 monotone along the four rays     HELD (0 violations)

    OUTCOME step 2b: NOT_FOUND_IN_RANGE
    range: s, u in [-34.1, 34.1] dense 41 x 41 + log set to 10^-6

Reading. The repaired set is the reach-bounded region: 100 steps at the
0.05 cap is a reach of 5.0 in parameter distance, the ball edge sits near
4.5, and the REPAIRED/STRANDED boundary falls in [10, 20) where the
transition zone splits 224 / 222 — the start distance at which 5.0 of
travel is or is not enough to cross the ball edge, modulated by the
direction (the KL ball is not a Euclidean sphere, so the cut is not at one
radius). No start closer than the reach is stranded, no start is
expelled, and KL after 100 steps rises monotonically with start distance
along every ray. That is a single basin with a reach limit, not a funnel:
there is no thin region of starts that fail while their neighbours
succeed. One slice of a 3216-dimensional space; a funnel transverse to
both d and o would not appear here.

Scope: deterministic (torch CPU, seed 42 environment, seed 1 slice
vector), default.yaml, 100 steps, corrected sign; 2D slice only.
