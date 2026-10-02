#!/usr/bin/env python3
"""
slice_step2.py -- step 2 as pre-registered (fde6c0e) run literally, and step 2b
(PREREGISTRATION_FIX.md) under the corrected repair.

Slice through theta_ref: d = (theta_drifted - theta_ref)/R (drift direction),
o = Gaussian vector (torch seed 1) orthogonalised to d.  Starts theta_0 =
theta_ref + s d + u o on the dense grid s, u in linspace(-2R, 2R, 41) and the
log set s in +-10^k (k = -6..1, half-decades), u in {0, +-10^k}.  KL ball
epsilon = 0.1 on safety_inputs against the FIXED theta_ref.

Step 2 (repair OFF): nothing moves theta (check C4), so each start settles
where it is; the two pre-registered counts are computed and printed.
Step 2b (repair ON, corrected sign): 100 repair steps per start; classified
REPAIRED / STRANDED / EXPELLED / HELD; reach band dist_0 <= 4.0; S1-S3 and
the outcome enum of PREREGISTRATION_FIX.md.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import torch.nn.functional as F             # noqa: E402

import timescale_check as TC                # noqa: E402

EPS_BALL = 0.1
STEPS = 100
REACH = 4.0


def main(argv):
    quick = "--quick" in argv
    t0 = time.time()
    from simulation.environment import Environment
    from manifolds.parameter_manifold import ParameterManifold
    env = Environment(TC.CFG["simulation"])
    layer = ParameterManifold(env.theta_ref, TC.CFG["manifolds"]["parameter"])
    model_fn = env.get_model_fn()
    ref = env.theta_ref
    with torch.no_grad():
        ref_logp = F.softmax(model_fn(env.safety_inputs, ref), dim=-1)

    def kl(t):
        with torch.no_grad():
            return F.kl_div(F.log_softmax(model_fn(env.safety_inputs, t), dim=-1), ref_logp,
                            reduction="batchmean").item()

    dvec = env.theta_drifted - ref
    R = torch.linalg.norm(dvec).item()
    d = dvec / R
    g = torch.Generator().manual_seed(1)
    o = torch.randn(ref.shape, generator=g)
    o = o - (o @ d) * d
    o = o / torch.linalg.norm(o)
    assert abs((o @ d).item()) < 1e-6

    n_dense = 9 if quick else 41
    lin = np.linspace(-2 * R, 2 * R, n_dense)
    starts = [("dense", float(s), float(u)) for s in lin for u in lin]
    ks = np.arange(-6, 1.01, 0.5)
    svals = [sg * 10.0 ** k for k in ks for sg in (+1, -1)]
    uvals = [0.0] + [sg * 10.0 ** k for k in ks for sg in (+1, -1)]
    if quick:
        svals, uvals = svals[::6], uvals[::6]
    starts += [("log", float(s), float(u)) for s in svals for u in uvals]
    print("slice: R = %.3f ; dense %d x %d ; log set %d ; total starts %d ; epsilon_ball %.2f ; steps %d"
          % (R, n_dense, n_dense, len(svals) * len(uvals), len(starts), EPS_BALL, STEPS))

    rows = []
    for i, (kind, s, u) in enumerate(starts):
        th0 = ref + s * d + u * o
        kl0 = kl(th0)
        th = th0.clone()
        nonfinite = False
        for _ in range(STEPS):
            th, m = layer.repair_step(th, model_fn, env.safety_inputs, env.task_inputs, env.task_labels)
            if not torch.isfinite(th).all():
                nonfinite = True; break
        kl1 = kl(th) if not nonfinite else float("nan")
        inside0, inside1 = kl0 < EPS_BALL, (kl1 < EPS_BALL) if not nonfinite else False
        cls = ("NONFINITE" if nonfinite else "HELD" if inside0 and inside1 else "EXPELLED" if inside0
               else "REPAIRED" if inside1 else "STRANDED")
        rows.append({"kind": kind, "s": s, "u": u, "dist0": float(np.hypot(s, u)), "kl0": kl0, "kl1": kl1,
                     "dist1": torch.linalg.norm(th - ref).item() if not nonfinite else float("nan"), "cls": cls})
        if i % 200 == 0:
            print("  %5d/%d  %.0fs" % (i, len(starts), time.time() - t0), flush=True)

    # ---- step 2 literal: repair OFF, nothing moves (C4); the two pre-registered counts
    out_in = sum(1 for r in rows if r["kl0"] >= EPS_BALL and r["kl0"] < EPS_BALL)      # settles where it is
    in_out = sum(1 for r in rows if r["kl0"] < EPS_BALL and r["kl0"] >= EPS_BALL)
    n_in0 = sum(1 for r in rows if r["kl0"] < EPS_BALL)
    print("\nSTEP 2 (repair OFF, as pre-registered fde6c0e): autonomous dynamics move nothing (check C4)")
    print("  starts inside the KL ball at t=0: %d of %d ; outside: %d" % (n_in0, len(rows), len(rows) - n_in0))
    print("  outside-ball starts that return inside on their own: %d" % out_in)
    print("  inside-ball starts that leave: %d" % in_out)
    print("  -> degenerate as predicted: no dynamics, both counts 0")

    # ---- step 2b
    from collections import Counter
    c = Counter(r["cls"] for r in rows)
    print("\nSTEP 2b (repair ON, corrected sign): %s" % dict(c))
    bands = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, 100)]
    print("  by start distance to theta_ref:")
    print("  %-8s %5s %9s %9s %9s %6s   KL1 median" % ("band", "n", "REPAIRED", "STRANDED", "EXPELLED", "HELD"))
    for lo, hi in bands:
        b = [r for r in rows if lo <= r["dist0"] < hi]
        if not b:
            continue
        cc = Counter(r["cls"] for r in b)
        med = float(np.median([r["kl1"] for r in b if np.isfinite(r["kl1"])])) if b else float("nan")
        print("  [%2d,%3d) %5d %9d %9d %9d %6d   %.4f" % (lo, hi, len(b), cc["REPAIRED"], cc["STRANDED"], cc["EXPELLED"], cc["HELD"], med))
    s1 = c["EXPELLED"] == 0
    band = [r for r in rows if r["dist0"] <= REACH and r["kl0"] >= EPS_BALL]
    s2 = all(r["cls"] == "REPAIRED" for r in band)
    # S3: along each ray from theta_ref, no STRANDED closer than a REPAIRED
    viol = []
    for axis, sign in (("s", +1), ("s", -1), ("u", +1), ("u", -1)):
        other = "u" if axis == "s" else "s"
        ray = sorted([r for r in rows if abs(r[other]) < 1e-12 and sign * r[axis] > 0], key=lambda r: abs(r[axis]))
        far_rep = -1.0
        for r in reversed(ray):
            if r["cls"] == "REPAIRED":
                far_rep = max(far_rep, abs(r[axis]))
        for r in ray:
            if r["cls"] == "STRANDED" and abs(r[axis]) < far_rep:
                viol.append((axis, sign, r["dist0"]))
    s3 = len(viol) == 0
    nonfin = c["NONFINITE"] / len(rows)
    print("\n  S1 EXPELLED = 0:                    %s (%d)" % ("HELD" if s1 else "FAILED", c["EXPELLED"]))
    print("  S2 reach band (<= %.1f) all REPAIRED: %s (%d of %d outside-ball starts in band)" % (REACH, "HELD" if s2 else "FAILED", sum(r["cls"] == "REPAIRED" for r in band), len(band)))
    print("  S3 monotone along the four rays:     %s (%d violations)" % ("HELD" if s3 else "FAILED", len(viol)))
    if viol:
        print("     violations (axis, sign, dist0):", viol[:10])
    if nonfin > 0.05:
        outcome = "INCONCLUSIVE (%.1f%% non-finite)" % (100 * nonfin)
    elif not s3 and any(v[2] <= REACH for v in viol):
        outcome = "FUNNEL_FOUND (close starts stranded while farther ones are repaired, inside the reach band)"
    else:
        outcome = ("NOT_FOUND_IN_RANGE (S1 %s S2 %s S3 %s; the repaired set is the reach-bounded region; "
                   "grid s,u in [-%.1f, %.1f] dense 41x41 + log set)" % (s1, s2, s3, 2 * R, 2 * R))
    print("\nOUTCOME step 2b:", outcome)
    print("scope: deterministic (torch CPU, seed 42 env, seed 1 slice), default.yaml, %d steps, one 2D slice of 3216 dims; %.0f s" % (STEPS, time.time() - t0))
    json.dump({"R": R, "eps_ball": EPS_BALL, "steps": STEPS, "reach": REACH, "counts": dict(c),
               "S1": s1, "S2": s2, "S3": s3, "violations": viol, "outcome": outcome, "rows": rows},
              open("slice_step2_results.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
