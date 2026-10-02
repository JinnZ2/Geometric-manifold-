#!/usr/bin/env python3
"""
funnel_traj_width.py -- PREREGISTRATION_TRAJ.md: the funnel's width measured
ALONG the trajectory as a kick sensitivity, and the linear-response threshold
it implies for step 1's two noise models.  stdlib only; imports the vendored
reference unchanged for rhs / classify / funnel_edge.

    delta_a*(t)  smallest additive kick on x at time t that flips e0 -> not e0
    delta_b*(t)  the same for a kick on mu
    I_a, I_b     integral dt / delta*(t)^2  (trapezoid on the time grid)
    model (a)    P(e0) ~ P(|N(0,1)| < 1/(sigma sqrt I_a))  -> sigma_half = 1/(0.6745 sqrt I_a)
    model (b)    P(e0) ~ Phi(1/(sigma sqrt I_b))            -> never below 0.5; sigma_75 = 1/(0.6745 sqrt I_b)

Measured comparators are read from step 1's samples/funnel_noise_results.json.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import singular_funnel_pitchfork as REF   # noqa: E402

A, B, EPS, MU0, DT = 3.0, 2.0, 0.1, 4.0, 0.01
STARTS = [1e-9, 10 ** -8.5, 1e-8]
Z_HALF = 0.6745                          # P(|N(0,1)| < z) = 0.5 ; Phi(z) = 0.75
T_GRID = [round(0.05 * i, 2) for i in range(61)] + [3.5 + 0.5 * i for i in range(60)]


def rk4(y, mu, dt=DT, eps=EPS):
    k1 = REF.rhs(y, mu, eps)
    k2 = REF.rhs(y + dt / 2 * k1[0], mu + dt / 2 * k1[1], eps)
    k3 = REF.rhs(y + dt / 2 * k2[0], mu + dt / 2 * k2[1], eps)
    k4 = REF.rhs(y + dt * k3[0], mu + dt * k3[1], eps)
    return (max(y + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), -690.0),
            mu + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def trajectory(x0):
    """(t, x, mu) at every step until the reference's own e0 exit."""
    y, mu, t = math.log(x0), MU0, 0.0
    out = [(0.0, x0, mu)]
    while t < REF.T_MAX:
        x = math.exp(y)
        if mu < 0.0 and x < 1e-8:
            return out, t, "e0"
        y, mu = rk4(y, mu)
        t += DT
        out.append((round(t, 6), math.exp(y), mu))
    return out, t, "UNCLASSIFIED"


def flips(x, mu):
    return REF.classify(x, mu, EPS) != "e0"


def delta_star(x, mu, on):
    """Bisection on log10 delta in [-14, 1]; smallest kick that flips. None if even 10 does not."""
    lo, hi = -14.0, 1.0
    probe = (lambda d: flips(x + d, mu)) if on == "x" else (lambda d: flips(x, mu + d))
    if not probe(10 ** hi):
        return None
    if probe(10 ** lo):
        return 10 ** lo
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if probe(10 ** mid) else (mid, hi)
    return 10 ** hi


def interp_sigma_at(sigmas, p, level):
    """log-interpolated sigma where p first falls to <= level; None if never."""
    for i in range(1, len(p)):
        if p[i] <= level < p[i - 1]:
            f = (p[i - 1] - level) / (p[i - 1] - p[i])
            return 10 ** (math.log10(sigmas[i - 1]) + f * (math.log10(sigmas[i]) - math.log10(sigmas[i - 1])))
        if p[i] <= level:
            return sigmas[i]
    return None


def main():
    t0 = time.time()
    meas = json.load(open(os.path.join(HERE, "samples", "funnel_noise_results.json")))
    edge0 = math.exp(REF.funnel_edge(MU0, EPS))
    print("funnel width along the trajectory  (mu0=%.1f eps=%.2f, RK4 dt=%.2f); t=0 edge x0* = %.3e" % (MU0, EPS, DT, edge0))
    res = {"edge0": edge0, "starts": []}
    preds = {}
    for si, x0 in enumerate(STARTS):
        traj, t_e0, status = trajectory(x0)
        by_t = {round(t, 2): (x, mu) for (t, x, mu) in traj}
        rows = []
        print("\nstart x0 = %.3e  (reaches e0 at t = %.2f, %s)" % (x0, t_e0, status))
        print("  %6s %10s %8s %10s %10s %10s %10s" % ("t", "x(t)", "mu(t)", "edge(mu)", "delta_a*", "delta_b*", "gain x(t)/x0"))
        for t in T_GRID:
            if t > t_e0:
                break
            x, mu = by_t[round(t, 2)]
            da = delta_star(x, mu, "x"); db = delta_star(x, mu, "mu")
            e = REF.funnel_edge(mu, EPS) if (t in (0.0, 0.5, 1.0, 2.0, 3.0) or t % 2 == 0) else None
            edge = math.exp(e) if e is not None else None
            rows.append({"t": t, "x": x, "mu": mu, "edge_mu": edge, "delta_a": da, "delta_b": db})
            if t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0) or (t >= 3.5 and (t * 2) % 4 == 0):
                print("  %6.2f %10.3e %8.4f %10s %10s %10s %10.3e" % (
                    t, x, mu, ("%.3e" % edge) if edge else "-", ("%.3e" % da) if da else "none",
                    ("%.3e" % db) if db else "none", x / x0))
        # integrals (trapezoid; a None delta contributes 0 -- no flip possible)
        def integral(key):
            I = 0.0
            for i in range(1, len(rows)):
                f0 = 1 / rows[i - 1][key] ** 2 if rows[i - 1][key] else 0.0
                f1 = 1 / rows[i][key] ** 2 if rows[i][key] else 0.0
                I += 0.5 * (f0 + f1) * (rows[i]["t"] - rows[i - 1]["t"])
            return I
        Ia, Ib = integral("delta_a"), integral("delta_b")
        sh_pred_a = 1 / (Z_HALF * math.sqrt(Ia)) if Ia > 0 else None
        s75_pred_b = 1 / (Z_HALF * math.sqrt(Ib)) if Ib > 0 else None
        ma = meas["models"]["a"]["starts"][si]; mb = meas["models"]["b"]["starts"][si]
        sh_meas_a = ma["sigma_half"]
        s75_meas_b = interp_sigma_at(meas["models"]["b"]["sigmas"], mb["p_e0"], 0.75)
        s50_meas_b = interp_sigma_at(meas["models"]["b"]["sigmas"], mb["p_e0"], 0.5)
        da0 = rows[0]["delta_a"]
        early = [r["delta_a"] for r in rows if r["t"] <= 1.0 and r["delta_a"]]
        t1 = (min(range(len(early)), key=lambda i: early[i]) == 0) and all(early[i] <= early[i + 1] * (1 + 1e-9) for i in range(len(early) - 1))
        # where does the first-order integrand live? share of I_a from t <= 0.25
        I_early = 0.0
        for i in range(1, len(rows)):
            if rows[i]["t"] <= 0.25:
                I_early += 0.5 * (1 / rows[i - 1]["delta_a"] ** 2 + 1 / rows[i]["delta_a"] ** 2) * (rows[i]["t"] - rows[i - 1]["t"])
        print("  I_a = %.4e  I_b = %.4e ;  share of I_a from t <= 0.25: %.3f" % (Ia, Ib, I_early / Ia if Ia else float("nan")))
        print("  model (a): sigma_half predicted %.3e  measured %.3e  ratio pred/meas %.3f" % (sh_pred_a, sh_meas_a, sh_pred_a / sh_meas_a))
        print("             sigma_half(meas) / delta_a*(0) = %.2f   (analytic 1.4826 sqrt(2 mu0) = %.2f)" % (sh_meas_a / da0, 1.4826 * math.sqrt(2 * MU0)))
        print("             delta_a*(0) = %.3e vs edge0 - x0 = %.3e  (ratio %.3f)" % (da0, edge0 - x0, da0 / (edge0 - x0)))
        print("  model (b): sigma_75 predicted %.3e  measured %s ; measured sigma_50 %s (asymptote 0.5 predicted)" % (
            s75_pred_b, ("%.3e" % s75_meas_b) if s75_meas_b else "never", ("%.3e" % s50_meas_b) if s50_meas_b else "never"))
        preds.setdefault("T1 bottleneck at the start (delta_a* minimal at t=0, monotone over t<=1)", []).append(t1)
        preds.setdefault("T2 sigma_half_pred(a) within factor 1.5 of measured", []).append(1 / 1.5 <= sh_pred_a / sh_meas_a <= 1.5)
        preds.setdefault("T3 sigma_half(a)/delta_a*(0) in [3, 6]", []).append(3 <= sh_meas_a / da0 <= 6)
        preds.setdefault("T4 model (b): never halves (asymptote 0.5) and sigma_75_pred within factor 1.5 of measured", []).append(
            s50_meas_b is None and s75_meas_b is not None and 1 / 1.5 <= s75_pred_b / s75_meas_b <= 1.5)
        preds.setdefault("T5 delta_a*(0) within 10% of edge0 - x0", []).append(abs(da0 / (edge0 - x0) - 1) <= 0.10)
        res["starts"].append({"x0": x0, "t_e0": t_e0, "rows": rows, "I_a": Ia, "I_b": Ib, "sigma_half_pred_a": sh_pred_a,
                              "sigma_half_meas_a": sh_meas_a, "sigma_75_pred_b": s75_pred_b, "sigma_75_meas_b": s75_meas_b,
                              "sigma_50_meas_b": s50_meas_b, "I_a_share_t_le_0.25": I_early / Ia if Ia else None})
    print("\nPRE-REGISTERED PREDICTIONS (per start: %s)" % ["%.1e" % s for s in STARTS])
    for k, v in preds.items():
        print("  %s %s  %s" % ("HELD  " if all(v) else "FAILED", k, v))
    res["predictions"] = {k: [bool(x) for x in v] for k, v in preds.items()}
    out1 = "BOTTLENECK_AT_START" if all(preds["T1 bottleneck at the start (delta_a* minimal at t=0, monotone over t<=1)"]) else "BOTTLENECK_INTERIOR"
    out2 = "LINEAR_RESPONSE_HOLDS" if (all(preds["T2 sigma_half_pred(a) within factor 1.5 of measured"]) and
                                      all(preds["T4 model (b): never halves (asymptote 0.5) and sigma_75_pred within factor 1.5 of measured"])) else "LINEAR_RESPONSE_FAILS"
    print("OUTCOME: %s ; %s" % (out1, out2))
    print("scope: deterministic trajectories + first-order noise model; a=3 b=2 eps=0.1 mu0=4; comparators are step 1's measured values; %.0f s" % (time.time() - t0))
    res["outcome"] = [out1, out2]
    json.dump(res, open("funnel_traj_width_results.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
