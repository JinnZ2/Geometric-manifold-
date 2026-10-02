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

## Step 2 — NOT RUN

Per the order ("If none -> NOT_IN_CLASS, stop") and the pre-registration.
Stated for the record: with repair off this simulation has no autonomous
flow at all, so every start on the pre-registered 2D slice would settle
where it began, and both KL-ball counts (outside points that return, inside
points that leave) would be 0 of N by construction — a property of the
engine, not a measurement of a basin.

Scope: deterministic; default.yaml (drift_strength 0.3, trust_radius 0.05,
asymmetry_lambda 10, curvature_weight 2, lr 0.01, 100 steps, seed 42);
continuous parameter space.
