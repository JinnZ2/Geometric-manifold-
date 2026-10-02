#!/usr/bin/env python3
"""
funnel_noise.py  -- CC0, stdlib only.  Step 1 of the singular-funnel work order.

Euler-Maruyama noise on the pitchfork funnel system of singular_funnel_pitchfork.py
(vendored UNCHANGED beside this file; reference values: selftest 4/4, REPRODUCED,
mu0=4 eps=0.1 edge log10 x0* = -7.30, slope d(ln x0*)/d(1/eps) = -1.616).

    dx/dt  = x (mu - x^2)          x >= 0   (fast)
    dmu/dt = eps (-mu + a x - b)            (slow)      a=3, b=2, eps=0.1

Two DECLARED noise models (fixed in PREREGISTRATION.md before any run):
    (a) additive on x, reflecting at x=0:  x <- |x + x(mu-x^2)dt + sigma sqrt(dt) xi|
    (b) additive on mu:                    mu <- mu + eps(-mu+ax-b)dt + sigma sqrt(dt) xi
        with x integrated as y = ln x (deterministic) so x ~ 1e-9 stays exact.

Absorbing exits (declared):  e0: mu < -0.5 and x < 0.1;  e2: x > 1.0;
INCONCLUSIVE at T_MAX.  These differ from the reference's `x < 1e-8` test
because that test is meaningless under x-noise of comparable size.

Readout: P(e0) per (model, start, sigma) with a Wilson 95% interval, and
sigma_half = log-interpolated sigma where P(e0) first falls to <= half of P(e0)
at the smallest sigma on the grid.  A null is a result; no parameter is tuned
after seeing results.

Scope: noisy (Euler-Maruyama, dt=0.01), one parameter set, continuous phase
space.  Nothing here is a statement about any repository's engine.
"""
import json
import math
import random
import sys

A, B, EPS, MU0 = 3.0, 2.0, 0.1, 4.0
DT, T_MAX = 0.01, 600.0
STARTS = [1e-9, 10 ** -8.5, 1e-8]
SIGMA = {
    "a": [10 ** (-11 + 0.5 * i) for i in range(11)],   # 1e-11 .. 1e-6
    "b": [10 ** (-4 + 0.5 * i) for i in range(9)],     # 1e-4  .. 1e0
}
RUNS = 200
MODEL_IDX = {"a": 0, "b": 1}


def seed_for(model, start_idx, sigma_idx, run_idx):
    return 7919 * MODEL_IDX[model] + 1009 * start_idx + 101 * sigma_idx + run_idx


def run_a(x0, sigma, rng):
    x, mu, t = x0, MU0, 0.0
    sq = math.sqrt(DT)
    while t < T_MAX:
        if mu < -0.5 and x < 0.1:
            return "e0"
        if x > 1.0:
            return "e2"
        x_new = x + x * (mu - x * x) * DT + sigma * sq * rng.gauss(0.0, 1.0)
        mu += EPS * (-mu + A * x - B) * DT
        x = abs(x_new)
        t += DT
    return "INCONCLUSIVE"


def run_b(x0, sigma, rng):
    y, mu, t = math.log(x0), MU0, 0.0
    sq = math.sqrt(DT)
    while t < T_MAX:
        x = math.exp(y)
        if mu < -0.5 and x < 0.1:
            return "e0"
        if x > 1.0:
            return "e2"
        y += (mu - x * x) * DT
        mu += EPS * (-mu + A * x - B) * DT + sigma * sq * rng.gauss(0.0, 1.0)
        y = max(y, -690.0)
        t += DT
    return "INCONCLUSIVE"


RUN = {"a": run_a, "b": run_b}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sigma_half(sigmas, probs):
    """Log-interpolated sigma where P first drops to <= 0.5 * P(sigma_min). None if never."""
    p0 = probs[0]
    if p0 <= 0:
        return None
    target = 0.5 * p0
    for i in range(1, len(sigmas)):
        if probs[i] <= target:
            s0, s1 = math.log10(sigmas[i - 1]), math.log10(sigmas[i])
            q0, q1 = probs[i - 1], probs[i]
            if q0 == q1:
                return 10 ** s1
            f = (q0 - target) / (q0 - q1)
            return 10 ** (s0 + f * (s1 - s0))
    return None


def selftest():
    ok = True
    def check(name, cond):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + name)
        ok &= cond
    print("SELFTEST (sigma = 0 reproduces the deterministic funnel under the declared exits)")
    r = random.Random(0)
    check("model a, x0=1e-9 inside funnel -> e0", run_a(1e-9, 0.0, r) == "e0")
    check("model b, x0=1e-9 inside funnel -> e0", run_b(1e-9, 0.0, r) == "e0")
    check("model a, x0=1e-6 outside funnel (edge 10^-7.3) -> e2", run_a(1e-6, 0.0, r) == "e2")
    check("model b, x0=1e-6 outside funnel -> e2", run_b(1e-6, 0.0, r) == "e2")
    check("wilson(0,200) upper < 0.02", wilson(0, 200)[1] < 0.02)
    check("sigma_half interpolates", abs(sigma_half([1, 10, 100], [1.0, 0.75, 0.25]) - 10 ** 1.5) < 1e-9)
    check("sigma_half None when never halves", sigma_half([1, 10], [1.0, 0.9]) is None)
    return ok


def main(argv):
    if not selftest():
        print("SELFTEST FAILED"); return 1
    quick = "--quick" in argv
    runs = 20 if quick else RUNS
    out = {"dt": DT, "t_max": T_MAX, "runs_per_point": runs, "starts": STARTS,
           "models": {}}
    print("\nRUN  P(e0) vs sigma, mu0=%g, %d runs per point" % (MU0, runs))
    for model in ("a", "b"):
        out["models"][model] = {"sigmas": SIGMA[model], "starts": []}
        print("\nmodel (%s)  %s" % (model, "additive on x, reflecting" if model == "a" else "additive on mu"))
        print("  %-10s" % "x0" + "".join("%9.1f" % math.log10(s) for s in SIGMA[model]) + "   log10 sigma")
        for si, x0 in enumerate(STARTS):
            probs, lows, highs, incon = [], [], [], []
            for gi, sigma in enumerate(SIGMA[model]):
                counts = {"e0": 0, "e2": 0, "INCONCLUSIVE": 0}
                for k in range(runs):
                    rng = random.Random(seed_for(model, si, gi, k))
                    counts[RUN[model](x0, sigma, rng)] += 1
                p = counts["e0"] / runs
                lo, hi = wilson(counts["e0"], runs)
                probs.append(p); lows.append(lo); highs.append(hi)
                incon.append(counts["INCONCLUSIVE"] / runs)
            sh = sigma_half(SIGMA[model], probs)
            out["models"][model]["starts"].append({
                "x0": x0, "log10_x0": math.log10(x0), "p_e0": probs,
                "wilson_lo": lows, "wilson_hi": highs, "p_inconclusive": incon,
                "sigma_half": sh,
            })
            print("  %-10.1f" % math.log10(x0) + "".join("%9.3f" % p for p in probs)
                  + "   sigma_half = %s" % ("%.3g" % sh if sh else "none in range"))
            if max(incon) > 0:
                print("  %-10s" % "" + "".join("%9.3f" % q for q in incon) + "   INCONCLUSIVE fraction")
    # outcome per model
    print("\nOUTCOME per model (enum from PREREGISTRATION.md)")
    for model in ("a", "b"):
        rows = out["models"][model]["starts"]
        p1 = all(r["p_e0"][0] >= 0.95 for r in rows)
        halves = [r["sigma_half"] for r in rows]
        inc = max(max(r["p_inconclusive"]) for r in rows)
        if inc > 0.05:
            verdict = "INCONCLUSIVE (%.1f%% runs hit T_MAX)" % (100 * inc)
        elif p1 and all(h is not None for h in halves):
            verdict = "FUNNEL_FOUND"
        elif not p1:
            verdict = "NOT_FOUND_IN_RANGE (deterministic funnel not reproduced at smallest sigma)"
        else:
            verdict = "NOT_FOUND_IN_RANGE (P(e0) never halves for sigma in [%.0e, %.0e])" % (
                SIGMA[model][0], SIGMA[model][-1])
        out["models"][model]["outcome"] = verdict
        print("  model (%s): %s   sigma_half per start: %s" % (
            model, verdict, ["%.3g" % h if h else None for h in halves]))
    # pre-registered predictions
    print("\nPRE-REGISTERED PREDICTIONS")
    ra, rb = out["models"]["a"]["starts"], out["models"]["b"]["starts"]
    def mono(rows):
        return all(all(r["p_e0"][i + 1] - r["p_e0"][i] <= 0.10 for i in range(len(r["p_e0"]) - 1)) for r in rows)
    ha = [r["sigma_half"] for r in ra]; hb = [r["sigma_half"] for r in rb]
    preds = {
        "P1 P(e0)>=0.95 at smallest sigma, every start, both models":
            all(r["p_e0"][0] >= 0.95 for r in ra + rb),
        "P2 P(e0) non-increasing in sigma (no rise > 0.10), both models": mono(ra) and mono(rb),
        "P3 model a: sigma_half in [1e-9, 1e-7], every start":
            all(h is not None and 1e-9 <= h <= 1e-7 for h in ha),
        "P4 model b: sigma_half in [1e-2, 1], every start":
            all(h is not None and 1e-2 <= h <= 1.0 for h in hb),
        "P5 sigma_half(b)/sigma_half(a) > 1e4 (start-matched)":
            all(x is not None and y is not None and y / x > 1e4 for x, y in zip(ha, hb)),
        "P6 within a model, sigma_half across starts within factor 3":
            all(None not in hs and max(hs) / min(hs) < 3.0 for hs in (ha, hb)),
    }
    for k, v in preds.items():
        print("  %s  %s" % ("HELD   " if v else "FAILED ", k))
    out["predictions"] = preds
    with open("funnel_noise_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nscope: noisy (Euler-Maruyama dt=%.3g), T_max=%g, one parameter set (a=3, b=2, eps=0.1, mu0=4),"
          " continuous phase space; results written to funnel_noise_results.json in the cwd" % (DT, T_MAX))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
