#!/usr/bin/env python3
"""Work order section 4: phase detection on every channel, alignment of drift axes against
the reference channel, and the repair-flag overlay. Every threshold is a [CHOICE] read
from config.json and printed by report.py.

Nothing in here averages across axes. `align()` is only CALLED by report.py when S4 has
licensed it; `phases()` and `overlay()` run regardless, because they are observations.
"""

from __future__ import annotations

import math

# ------------------------------------------------------------------ series helpers


def smooth(xs, half):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        seg = [v for v in xs[lo:hi] if v is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


def series(rows, axis, key="rankme"):
    """(steps, values) for one axis over a run's rows; absent where the axis is absent."""
    steps, vals = [], []
    for r in rows:
        if axis in r and r[axis][key] is not None:
            steps.append(r["step"])
            vals.append(r[axis][key])
    return steps, vals


def a3_series(a3_by_step, axis, key="rankme"):
    items = sorted(((int(t), v) for t, v in a3_by_step.items()), key=lambda kv: kv[0])
    return [t for t, _ in items], [v[axis][key] for _, v in items]


# ------------------------------------------------------------------ phase detection


def phases(steps, vals, half, margin_frac):
    """Three-leg shape test on one series: fall1 (start to a trough), rise (trough to a
    later peak), fall2 (peak to the end). Each leg must clear margin_frac of the smoothed
    range. On the REFERENCE channel the three legs are the shape Li et al. report and may be
    read with their names; on a drift axis they are fall / rise / fall and nothing more
    (order section 7: no import of the phase names onto parameter space). Returns the legs,
    the turning-point steps, the smoothed extrema, and `depth` = start minus trough in the
    series' own units, which the relative margin does not see.

    Also returns `nonmonotone`: whether the smoothed series has at least one interior
    extremum clearing the margin on both sides -- the order's weaker "ANY structure" question.
    """
    if len(vals) < 5:
        return {"detectable": False, "reason": "too few points"}
    s = smooth(vals, half)
    rng = max(s) - min(s)
    m = margin_frac * rng
    # trough first (global minimum, first occurrence), then the peak AFTER it. Taking the
    # global maximum first misses any series that starts at its maximum -- caught by the
    # fall-rise-fall known-answer fixture in selftest.py.
    i_tr = min(range(len(s)), key=s.__getitem__)
    i_pk = i_tr + max(range(len(s) - i_tr), key=lambda k: s[i_tr + k])
    fall1 = s[0] - s[i_tr] > m
    rise = s[i_pk] - s[i_tr] > m
    fall2 = s[i_pk] - s[-1] > m
    interior = False
    for i in range(1, len(s) - 1):
        lo_l, hi_l = min(s[:i]), max(s[:i])
        lo_r, hi_r = min(s[i + 1 :]), max(s[i + 1 :])
        if s[i] <= lo_l and s[i] <= lo_r and hi_l - s[i] > m and hi_r - s[i] > m:
            interior = True
        if s[i] >= hi_l and s[i] >= hi_r and s[i] - lo_l > m and s[i] - lo_r > m:
            interior = True
    return {
        "detectable": True,
        "fall1": fall1,
        "rise": rise,
        "fall2": fall2,
        "three_phase": fall1 and rise and fall2,
        "two_phase": fall1 and rise,
        "depth": s[0] - s[i_tr],
        "nonmonotone": interior,
        "t_start": steps[0],
        "t_trough": steps[i_tr],
        "t_peak": steps[i_pk],
        "t_end": steps[-1],
        "v_start": s[0],
        "v_trough": s[i_tr],
        "v_peak": s[i_pk],
        "v_end": s[-1],
        "range": rng,
        "margin": m,
    }


# ------------------------------------------------------------------ alignment (licensed only)


def _standardize(xs):
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) or 1.0
    return [(x - mu) / sd for x in xs]


def _interp(steps, vals, at):
    """Linear interpolation of (steps, vals) onto `at` (both ascending); clamps at the ends."""
    out, j = [], 0
    for t in at:
        while j + 1 < len(steps) and steps[j + 1] <= t:
            j += 1
        if t <= steps[0]:
            out.append(vals[0])
        elif j + 1 >= len(steps):
            out.append(vals[-1])
        else:
            t0, t1 = steps[j], steps[j + 1]
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            out.append(vals[j] * (1 - w) + vals[j + 1] * w)
    return out


def cross_correlation(ref_steps, ref_vals, ax_steps, ax_vals, half, lag_tol):
    """Max |corr| over lags within +-lag_tol steps, on smoothed standardized series
    interpolated to the reference's steps. Returns (corr_at_best_lag, best_lag, corr_at_zero)."""
    lo, hi = max(ref_steps[0], ax_steps[0]), min(ref_steps[-1], ax_steps[-1])
    grid = [t for t in ref_steps if lo <= t <= hi]
    if len(grid) < 8:
        return None
    r = _standardize(smooth(_interp(ref_steps, ref_vals, grid), half))
    step = min(b - a for a, b in zip(grid, grid[1:])) or 1
    best, zero = (0.0, 0), None
    for lag in range(-lag_tol, lag_tol + 1, step):
        a = _standardize(smooth(_interp(ax_steps, ax_vals, [t + lag for t in grid]), half))
        c = sum(x * y for x, y in zip(r, a)) / len(r)
        if lag == 0:
            zero = c
        if abs(c) > abs(best[0]):
            best = (c, lag)
    return best[0], best[1], zero


def align(ref_steps, ref_vals, ax_steps, ax_vals, half, corr_threshold, lag_tol):
    cc = cross_correlation(ref_steps, ref_vals, ax_steps, ax_vals, half, lag_tol)
    if cc is None:
        return {"verdict": "NOT_COMPUTABLE", "reason": "insufficient overlap"}
    c, lag, c0 = cc
    if c >= corr_threshold:
        v = "LOCKED"
    elif c <= -corr_threshold:
        v = "ANTI-PHASE"
    else:
        v = "DECOUPLED"
    return {"verdict": v, "corr": c, "lag": lag, "corr_lag0": c0}


# ------------------------------------------------------------------ repair-flag overlay


def first_crossing(rows, key, threshold):
    for r in rows:
        if r[key] < threshold:
            return r["step"]
    return None


def overlay(rows, ph, param_threshold, policy_threshold):
    """Where the repo's two flag rules first fire, relative to the reference channel's
    turning points. On a run from init both quantities are monotone in distance, so each
    fires once; the overlay is a placement, not a clustering."""
    t_param = first_crossing(rows, "param_confidence", param_threshold)
    t_policy = first_crossing(rows, "policy_confidence", policy_threshold)

    def place(t):
        if t is None:
            return "never fires"
        if not ph.get("detectable"):
            return f"step {t} (reference phases not detectable)"
        if t <= ph["t_trough"]:
            return f"step {t}: before the REP trough at {ph['t_trough']}"
        if t <= ph["t_peak"]:
            return f"step {t}: between REP trough {ph['t_trough']} and peak {ph['t_peak']}"
        return f"step {t}: after the REP peak at {ph['t_peak']}"

    def zone(t):
        if t is None or not ph.get("detectable"):
            return "none"
        return (
            "before_trough"
            if t <= ph["t_trough"]
            else ("between" if t <= ph["t_peak"] else "after_peak")
        )

    return {
        "param_flag_first_step": t_param,
        "param_flag_placement": place(t_param),
        "param_zone": zone(t_param),
        "policy_flag_first_step": t_policy,
        "policy_flag_placement": place(t_policy),
        "policy_zone": zone(t_policy),
    }
