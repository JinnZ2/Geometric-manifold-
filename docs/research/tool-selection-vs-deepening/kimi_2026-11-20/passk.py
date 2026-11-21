#!/usr/bin/env python3
"""Unbiased pass@k (Chen et al. 2021) + paired arm comparison.

Usage:
    python passk.py --self-test
    python passk.py --arm A=arm_A.json --arm B=arm_B.json
    python passk.py --arm A=a.json --arm B=b.json --ks 1,8,64,256 --out cmp.json

Arm file format: {problem_id: {"n": int, "c": int}}  (c = correct samples)
Emitted by step1b_passk_from_dump.py --emit-arm-files, or built by hand
for step 2.

Note: pass@k with k > n returns None and the problem is dropped from that
point; the dump from step 1 supports k <= 8 (G=8 samples per config).
pass@256 belongs to step 2's own sampling (n = 256).
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path


def pass_at_k(n, c, k):
    if k > n or n <= 0:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curve(arm, ks):
    """arm: {problem: {n,c}}. Pooled = mean of per-problem pass@k over
    problems with n >= k. The n>=k filter is a real selection effect at
    high k and is reported as n_problems per point."""
    out = {}
    for k in ks:
        vals = [pass_at_k(p["n"], p["c"], k) for p in arm.values()
                if p["n"] >= k]
        out[k] = {"pass": round(sum(vals) / len(vals), 4) if vals else None,
                  "n_problems": len(vals)}
    return out


def paired_delta(A, B, k, n_boot=2000, alpha=0.05, seed=0):
    """B minus A at k, over problems present in both with n>=k.
    Bootstrap resamples problems; CI is problem-level."""
    pairs = []
    for pid, pa in A.items():
        pb = B.get(pid)
        if not pb or pa["n"] < k or pb["n"] < k:
            continue
        va = pass_at_k(pa["n"], pa["c"], k)
        vb = pass_at_k(pb["n"], pb["c"], k)
        if va is None or vb is None:
            continue
        pairs.append((va, vb))
    N = len(pairs)
    if N < 2:
        return {"delta": None, "ci95": None, "n_problems": N}
    point = sum(vb - va for va, vb in pairs) / N
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        boots.append(sum(pairs[i][1] - pairs[i][0] for i in idx) / N)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"delta": round(point, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "n_problems": N}


def load_arm(path):
    raw = json.loads(Path(path).read_text())
    arm = {}
    for pid, v in raw.items():
        if isinstance(v, dict) and "n" in v and "c" in v:
            arm[str(pid)] = {"n": int(v["n"]), "c": int(v["c"])}
    return arm


def self_test():
    # estimator unit checks (pinned before the estimator touches real data)
    assert pass_at_k(10, 10, 1) == 1.0 and pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(1, 1, 1) == 1.0
    assert abs(pass_at_k(5, 2, 1) - 0.4) < 1e-9
    assert pass_at_k(2, 0, 1) == 0.0 and pass_at_k(3, 1, 3) == 1.0
    assert pass_at_k(4, 2, 8) is None  # k > n -> dropped, never guessed

    # paired-delta smoke: planted narrowing (B: +pass@1, -pass@8)
    rng = random.Random(11)
    A, B = {}, {}
    for i in range(120):
        p = rng.uniform(0.1, 0.9)
        ca = sum(rng.random() < p for _ in range(8))
        # collapsed tail: mass on the modal outcome, pass@1 rises slightly,
        # pass@8 falls hard (U(0.1,0.9): pass@1 0.50 -> 0.625, pass@8 ~0.9 -> 0.625)
        cb = 8 * int(p > 0.4)
        A[f"q{i}"] = {"n": 8, "c": ca}
        B[f"q{i}"] = {"n": 8, "c": cb}
    d1 = paired_delta(A, B, 1)
    d8 = paired_delta(A, B, 8)
    assert d1["delta"] > 0 and d1["ci95"][0] > 0, d1
    assert d8["delta"] < 0 and d8["ci95"][1] < 0, d8
    print(json.dumps({"self_test": "PASS",
                      "planted_narrowing": {"d_pass1": d1["delta"],
                                            "d_pass8": d8["delta"]}},
                     indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="LABEL=path.json  (repeatable)")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--out", default="passk_compare.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            print(f"error: --arm wants LABEL=path, got {spec}", file=sys.stderr)
            return 2
        label, path = spec.split("=", 1)
        arms[label] = load_arm(path)
    if len(arms) < 1:
        print("error: at least one --arm required", file=sys.stderr)
        return 2

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out = {"arms": {}, "comparisons": {}}
    for label, arm in arms.items():
        out["arms"][label] = {"n_problems": len(arm),
                              "curve": {str(k): v for k, v in
                                        curve(arm, ks).items()}}
    labels = sorted(arms)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            cmp_ = {}
            for k in ks:
                cmp_[str(k)] = paired_delta(arms[a], arms[b], k)
            out["comparisons"][f"{b}_minus_{a}"] = cmp_

    Path(args.out).write_text(json.dumps(out, indent=2))

    for label, v in out["arms"].items():
        pts = "  ".join(f"@{k}={v['curve'][str(k)]['pass']}"
                        for k in ks if v["curve"][str(k)]["pass"] is not None)
        print(f"[{label}] n_problems={v['n_problems']}\n  {pts}")
    for name, cmp_ in out["comparisons"].items():
        print(f"\n[{name}]  (delta, ci95, n_problems)")
        for k in ks:
            c = cmp_[str(k)]
            if c["delta"] is not None:
                print(f"  k={k:>3}  {c['delta']:+.4f}  "
                      f"{c['ci95']}  n={c['n_problems']}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
