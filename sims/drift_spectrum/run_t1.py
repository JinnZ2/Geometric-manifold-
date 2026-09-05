#!/usr/bin/env python3
"""Tier 1 sweep: skewed and control conditions, N seeds, per-step loss + RankMe + alpha-ReQ
on the reference channel and on every drift axis. Writes results/<stamp>/raw.json.

The AXIS is on every spectrum record (spectrum.Spectrum refuses otherwise). Nothing here
averages across axes. Stdlib only; ~2-4 minutes at the shipped config, --quick for less.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import drift as D  # noqa: E402

LN2 = math.log(2.0)


def load_config(path):
    return json.loads(Path(path).read_text())


def measure_steps(cfg):
    T = cfg["train"]["steps"]
    dense, every = cfg["measure"]["dense_until"], cfg["measure"]["every"]
    return [t for t in range(T + 1) if t < dense or t % every == 0]


def js_confidence(p_rows, q_rows):
    """The repo's PolicyManifold.trajectory_confidence: 1 - JS/ln2, batch-mean JS."""
    tot = 0.0
    for p, q in zip(p_rows, q_rows):
        js = 0.0
        for a, b in zip(p, q):
            m = 0.5 * (a + b)
            if a > 0:
                js += 0.5 * a * math.log(a / m)
            if b > 0:
                js += 0.5 * b * math.log(b / m)
        tot += js
    return max(0.0, 1.0 - (tot / len(p_rows)) / LN2)


def curvature_proxy(prob_rows):
    """The repo's ParameterManifold.curvature_proxy: mean over rows of var(softmax)."""
    tot = 0.0
    for p in prob_rows:
        mu = sum(p) / len(p)
        tot += sum((v - mu) ** 2 for v in p) / (len(p) - 1)
    return tot / len(prob_rows)


def run_condition(cfg, skew, seed, steps_to_measure):
    mc, dc, tc = cfg["model"], cfg["data"], cfg["train"]
    rng = random.Random(seed)
    protos = D.make_prototypes(rng, mc["n_classes"], mc["n_in"])
    freqs = D.class_frequencies(mc["n_classes"], skew)
    prng = random.Random(seed + 1000)
    if dc["probe_distribution"] == "uniform":
        px, py = D.probe_set(prng, protos, dc["probe_per_class"], dc["noise"])
    else:
        px, py = D.sample_batch(
            prng, protos, freqs, dc["probe_per_class"] * mc["n_classes"], dc["noise"]
        )
    model = D.MLP(random.Random(seed), mc["n_in"], mc["d"], mc["n_classes"], mc["init_scale"])
    theta0 = model.theta()
    probs0 = [D.MLP.softmax(model.forward(x)[1]) for x in px]

    W = cfg["axes"]["A1_window"]
    lam = cfg["repair_flags"]["lambda_curv"]
    window = []
    prev = theta0
    rows = []
    theta_at = {}
    measure = set(steps_to_measure)
    loss = None
    for t in range(tc["steps"] + 1):
        if t in measure:
            theta = model.theta()
            rec = {"step": t, "loss": loss, "acc_probe": model.accuracy(px, py)}
            rep = D.rep_spectrum(model, px)
            rec["REP"] = rep.as_dict()
            if len(window) >= W:
                rec["A1/TIME"] = D.a1_time(window[-W:]).as_dict()
            if window:
                l1, l2 = D.a2_unit(window[-1], model)  # the most recent per-step delta
                rec["A2/UNIT-L1"] = l1.as_dict()
                rec["A2/UNIT-L2"] = l2.as_dict()
            # the repo's two flag rules, reference := theta_0
            probs = [D.MLP.softmax(model.forward(x)[1]) for x in px]
            dist = D.norm(D.sub(theta, theta0))
            curv = curvature_proxy(probs)
            rec["dist_to_ref"] = dist
            rec["param_confidence"] = math.exp(-lam * curv - dist)
            rec["policy_confidence"] = js_confidence(probs, probs0)
            rows.append(rec)
            theta_at[t] = theta
        xs, ys = D.sample_batch(rng, protos, freqs, dc["batch"], dc["noise"])
        loss = model.sgd_step(xs, ys, tc["lr"], tc["weight_decay"])
        theta = model.theta()
        window.append(D.sub(theta, prev))
        if len(window) > W:
            window.pop(0)
        prev = theta
    return {
        "skew": skew,
        "seed": seed,
        "rows": rows,
        "theta_at": theta_at,
        "theta0": theta0,
        "model": model,
    }


def a3_across_seeds(cfg, runs, steps_to_measure):
    """A3: at each matched step, rows = theta_t - theta_0 per seed. Raw, and aligned by
    permuting each seed's hidden units to seed 0's at the FINAL checkpoint (one permutation
    per seed, applied at every step, so the alignment cannot itself vary over time)."""
    mc = cfg["model"]
    d, n_in, n_cls = mc["d"], mc["n_in"], mc["n_classes"]
    ref = runs[0]["model"]
    perms = [D.align_hidden(r["model"], ref) for r in runs]
    out = {}
    for t in steps_to_measure:
        if t == 0:
            continue
        raw = [D.sub(r["theta_at"][t], r["theta0"]) for r in runs]
        aligned = [D.permute_theta(v, p, d, n_in, n_cls) for v, p in zip(raw, perms)]
        out[t] = {
            "A3/SEED-raw": D.a3_seed(raw, aligned=False).as_dict(),
            "A3/SEED-aligned": D.a3_seed(aligned, aligned=True).as_dict(),
        }
    return out, perms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true", help="2 seeds, 400 steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["train"]["steps"] = min(cfg["train"]["steps"], 400)
    steps = measure_steps(cfg)
    t0 = time.time()
    result = {"config": cfg, "quick": args.quick, "measure_steps": steps, "conditions": {}}
    for label, skew in (("skewed", cfg["data"]["skew"]), ("control", cfg["data"]["control_skew"])):
        runs = []
        for seed in cfg["seeds"]:
            r = run_condition(cfg, skew, seed, steps)
            runs.append(r)
            print(
                f"  {label} seed {seed}: final loss {r['rows'][-1]['loss']:.3f} "
                f"acc {r['rows'][-1]['acc_probe']:.2f} REP RankMe {r['rows'][-1]['REP']['rankme']:.2f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        a3, perms = a3_across_seeds(cfg, runs, steps)
        result["conditions"][label] = {
            "skew": skew,
            "per_seed": [{"seed": r["seed"], "rows": r["rows"]} for r in runs],
            "A3_by_step": a3,
            "hidden_permutations_to_seed0": perms,
        }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    result["stamp"] = stamp
    result["runtime_s"] = round(time.time() - t0, 1)
    result["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    out = Path(args.out) if args.out else HERE / "results" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(result))
    print(f"wrote {out / 'raw.json'} in {result['runtime_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
