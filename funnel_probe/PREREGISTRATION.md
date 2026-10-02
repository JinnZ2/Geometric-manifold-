# PRE-REGISTRATION — funnel_probe/ (Geometric-manifold-)

WORK ORDER: singular-funnel probes, 4 repos (Kavik via Claude, 2026-10-01).
Source under test: Yanchuk, Wieczorek, Jardon-Kojakhmetov, Alkhayuon,
PRL 137, 147202 (2026); arXiv:2601.02001.
Committed BEFORE any probe code exists in this folder. Nothing outside
`funnel_probe/` is edited. Run artifacts go to a temp directory, not here.

Environment facts recorded at pre-registration time: Python 3.11.15; numpy
and scipy installed in-session; torch NOT installable (PyPI and the PyTorch
CPU index both refused from this container). The repo's own simulation
needs torch, so what can run here is a numpy twin of the two-layer model
and a code-level reading of the controller.

## Outcome enum (kept distinct, reported per step)

    FUNNEL_FOUND         a set of starts that reaches the basin the reduced
                         (adiabatic) picture forbids, shrinking with the
                         timescale ratio but non-empty
    NOT_FOUND_IN_RANGE   no such set inside the searched grid (range printed)
    NOT_IN_CLASS         no timescale separation: no slow variable enters the
                         fast dynamics as a parameter
    INCONCLUSIVE         a time cap was hit before classification

## Step 1 — does the simulation separate timescales?

Definition used (fixed now): the paper's class needs a SLOW variable (mu,
rate eps) that enters the FAST dynamics as a parameter and is itself driven
by the fast state. "Drift rate" below means the per-step change applied to
theta by anything other than the repair step, measured INSIDE the loop of
`simulation/controller.py::Controller.run`. "Repair rate" means ||delta||
per step of `ParameterManifold.repair_step`, capped at `trust_radius`.

Rule:
    drift_rate_in_loop == 0      -> ratio = 0 -> NOT_IN_CLASS, stop (per order)
    0 < drift/repair < 0.1       -> separated, proceed to step 2
    drift/repair >= 0.1          -> NOT_IN_CLASS (one timescale), stop

Measurements to report, in this order:
  1. Code reading of `Controller.run`: where `theta_drifted` is produced
     (one-shot, `Environment._setup`) and whether any term moves theta
     inside the loop besides `repair_step`. Count of in-loop drift
     applications per step.
  2. Numpy twin (`funnel_probe/twin.py`): same architecture
     (32-64-16 MLP, ReLU, 3216 params), same losses (CE on task, KL to the
     reference on safety, curvature proxy = softmax variance), same saddle
     objective `task - lambda*weighted_safety`, same trust-region clamp,
     default config values (lr 0.01, trust_radius 0.05, lambda 10,
     curvature_weight 2, drift_strength 0.3, 100 steps). Init: nn.Linear's
     U(-1/sqrt(fan_in), +1/sqrt(fan_in)) with numpy seed 42 — NOT the same
     bytes as torch.manual_seed(42); stated as such. Reports per step:
     ||delta||, KL to reference, distance to reference. Repair rate =
     mean ||delta|| over 100 steps; initial displacement = ||theta_drifted
     - theta_ref||; steps-to-close = displacement / repair_rate.
  3. `addon_thermodynamic_control/stability.py::CoupledDynamicalSystem`:
     code reading of where `sigma_drift` is used (grep says: inside the
     proactive penalty's expectation only, lines 425-438; not applied to
     theta). Reported as a fact, not run (torch).
  4. `sims/rate_induced_escape/` injects white-noise drift per step in an
     EXPERIMENT, not in the simulation; white noise per step has no eps —
     recorded as not the paper's class, not run (torch).

## Step 2 — conditional on step 1 passing (pre-registered so it cannot be
## shaped after the fact; NOT run if step 1 returns NOT_IN_CLASS)

Repair OFF. 2D slice through theta_ref: d = (theta_drifted - theta_ref)
normalized (drift direction), o = a Gaussian vector orthogonalized to d
(seed 1). Dense grid: s, u in linspace(-2R, 2R, 41) with
R = ||theta_drifted - theta_ref||. Log-scaled: s in +-10^k, k = -6..1 in
half-decades, u in {0, +-10^k}. KL ball: epsilon_basin = 0.1 (the addon's
default), KL measured on `safety_inputs` against theta_ref. Each start is
classified by where it settles under the simulation's own autonomous
dynamics with repair off. Two counts, both reported before any statement
about repair: outside-ball starts that return inside on their own;
inside-ball starts that leave.

## Step 1 — reference + noise (shared across the four repos, identical text)

Reference: `singular_funnel_pitchfork.py` vendored UNCHANGED (sha256 recorded
below). Step 0 result on this machine before any repo work: selftest 4/4 PASS,
STATUS REPRODUCED, mu0=4 eps=0.1 edge log10 x0* = -7.30, slope
d(ln x0*)/d(1/eps) = -1.616. Matches the order's expected values.

Noise script `funnel_noise.py` (stdlib only), Euler-Maruyama, dt = 0.01,
T_max = 600, a = 3, b = 2, eps = 0.1, mu0 = 4.

    model (a)  additive on x, reflecting at x = 0:
               x  <- |x + x(mu - x^2) dt + sigma sqrt(dt) xi|
               mu <- mu + eps(-mu + a x - b) dt
    model (b)  additive on mu, x integrated as y = ln x (deterministic):
               y  <- y + (mu - x^2) dt
               mu <- mu + eps(-mu + a x - b) dt + sigma sqrt(dt) xi

Absorbing exits (declared before any run; differ from the reference's because
the e0 test `x < 1e-8` is not meaningful under x-noise of that size):
    e0  : mu < -0.5 and x < 0.1
    e2  : x > 1.0
    INCONCLUSIVE : neither reached by T_max

Grid:
    starts     mu0 = 4, x0 in {1e-9, 10^-8.5, 1e-8}   (all inside the
               deterministic funnel, whose edge at mu0=4, eps=0.1 is 10^-7.30)
    sigma (a)  10^-11, 10^-10.5, ..., 10^-6      (11 values)
    sigma (b)  10^-4, 10^-3.5, ..., 10^0         (9 values)
    runs       200 per (model, start, sigma); seed = 7919*model_idx
               + 1009*start_idx + 101*sigma_idx + run_idx; random.Random(seed)
    readout    P(e0) with a Wilson 95% interval; sigma_half = the log-
               interpolated sigma at which P(e0) first falls to <= 0.5 of
               P(e0) at the smallest sigma on the grid, per (model, start)

Pre-registered predictions (checked, not tuned):
    P1  at the smallest sigma, P(e0) >= 0.95 for every start (noise
        negligible; reproduces the deterministic funnel)
    P2  P(e0) is non-increasing in sigma up to sampling noise (no rise
        larger than 0.10 between adjacent grid points)
    P3  model (a): sigma_half in [1e-9, 1e-7] (noise competes directly
        with a funnel of width ~5e-8 in x)
    P4  model (b): sigma_half in [1e-2, 1] (noise enters x only through
        the time integral of mu; margin to the edge is ~3.9 nats of ln x)
    P5  sigma_half(b) / sigma_half(a) > 1e4
    P6  within one model, sigma_half across the three starts agrees to
        within a factor of 3 (the width, not the depth, sets survival)

Outcome enum for this step:
    FUNNEL_FOUND         P1 holds and sigma_half lies inside the grid
    NOT_FOUND_IN_RANGE   P(e0) never halves inside the grid (range printed)
    NOT_IN_CLASS         not applicable: this is the paper's own system
    INCONCLUSIVE         > 5% of runs at any grid point used for the
                         decision hit T_max unclassified

vendored reference sha256: 466629a741722183a09221a1594c8f75db7d922ce00eede3afd8e7f01c1f9c46

Scope on every result: deterministic unless the step says noisy; the
parameter set named in the step; continuous parameter space (theta).
A null is a result. No parameter is retuned after a result without a
new pre-registration committed first.
