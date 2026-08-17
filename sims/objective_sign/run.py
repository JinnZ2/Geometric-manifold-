#!/usr/bin/env python3
"""Does the minus sign in `task_loss - lambda*safety` do what the docs claim?

The repo's governing rule is the scientific method with physics laws as an a priori base
held defeasibly. So CLAUDE.md's "Do Not change the sign" is a claim under test here, and
the physics prior is stated in REFUTE.md so that it can lose too.

Prior: KL(f_theta || f_theta_ref) is a potential well centred on the reference -- zero
there, non-negative, Fisher as its local metric. Descending it restores; ascending it makes
the reference repulsive. The standard saddle-point form the terminology map already cites
(Altman; CPO) is min_theta max_lambda [task + lambda*violation]: theta descends, lambda
ascends, and the safety term enters theta's objective with a PLUS.

Documented claim, given a fair test: the minus sign "ensures the repair explores the loss
landscape rather than collapsing to a local minimum". Exploration that means anything must
arrive somewhere better, so the minus arm wins if it reaches lower final safety KL at
matched-or-better task loss.

Both arms run the identical operator with only the sign flipped. Tier 2 (torch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.linalg as LA
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from simulation.environment import Environment  # noqa: E402


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def repair_step(theta, model_fn, theta_ref, safety_inputs, task_inputs, task_labels,
                cfg, lam, sign):
    """The parameter-manifold step, with the objective's sign as the only variable.

    Reproduces manifolds/parameter_manifold.py exactly -- same curvature proxy, same
    Euclidean trust-region clamp -- so the arms differ in one character and nothing else.
    """
    theta = theta.detach().requires_grad_(True)
    task_out = model_fn(task_inputs, theta)
    task_loss = F.cross_entropy(task_out, task_labels)

    with torch.no_grad():
        ref_out = model_fn(safety_inputs, theta_ref)
    safety_out = model_fn(safety_inputs, theta)
    kl_loss = F.kl_div(F.log_softmax(safety_out, dim=-1),
                       F.softmax(ref_out, dim=-1), reduction="batchmean")

    curv = torch.var(F.softmax(safety_out, dim=-1), dim=-1).mean()
    weighted_safety = kl_loss * (1.0 + cfg["curvature_weight"] * curv)

    total = task_loss - lam * weighted_safety if sign == "minus" \
        else task_loss + lam * weighted_safety
    total.backward()

    with torch.no_grad():
        delta = -cfg["lr"] * theta.grad
        norm = LA.norm(delta)
        if norm > cfg["trust_radius"]:
            delta = delta * (cfg["trust_radius"] / norm)
        theta_new = theta + delta

    return theta_new.detach(), float(kl_loss.item()), float(task_loss.item())


def run_arm(seed, drift, lam, sign, cfg, steps, start_at_ref=False):
    env = Environment({"drift_strength": drift, "seed": seed})
    model_fn = env.get_model_fn()
    ref = env.theta_ref
    theta = ref.clone() if start_at_ref else env.theta_drifted.clone()

    kl0 = task0 = None
    kl = task = float("nan")
    for i in range(steps):
        theta, kl, task = repair_step(theta, model_fn, ref, env.safety_inputs,
                                      env.task_inputs, env.task_labels, cfg, lam, sign)
        if i == 0:
            kl0, task0 = kl, task
    return {"seed": seed, "drift": drift, "lam": lam, "sign": sign,
            "kl_first": kl0, "kl_final": kl, "task_first": task0, "task_final": task,
            "dist_final": float(LA.norm(theta - ref).item())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))
    seeds = cfg["seeds"]
    drifts = cfg["sweeps"]["drift_strength"]
    lams = cfg["sweeps"]["asymmetry_lambda"]
    steps = cfg["steps"]
    if args.quick:
        seeds, drifts, lams, steps = seeds[:2], drifts[:1], lams[:1], 15
    pcfg = cfg["parameter"]

    # --- null: start at theta_ref, where there is nothing to repair -----------
    print("[sign] NULL — started at theta_ref (KL = 0, nothing to repair)")
    null_rows = []
    for sign in cfg["sweeps"]["sign"]:
        for s in seeds[:4]:
            null_rows.append(run_arm(s, 0.0, lams[0], sign, pcfg, steps, start_at_ref=True))
    null = {}
    for sign in cfg["sweeps"]["sign"]:
        rs = [r for r in null_rows if r["sign"] == sign]
        null[sign] = {"kl_final": median([r["kl_final"] for r in rs]),
                      "dist_final": median([r["dist_final"] for r in rs])}
        print(f"   {sign:<6} final KL from zero = {null[sign]['kl_final']:.6f}  "
              f"dist from ref = {null[sign]['dist_final']:.4f}")
    # H1 gate CORRECTED after the first run, and the correction is disclosed in
    # FINDING.md rather than quietly applied. The original test demanded the plus arm
    # hold KL <= 1e-6 from a start at theta_ref. That is wrong on physics grounds, not
    # on results grounds: theta_ref minimises the SAFETY term only, and the composite
    # objective also carries task_loss, whose minimum is elsewhere. No composite
    # objective can be stationary at theta_ref, so the absolute gate was unmeetable by
    # construction. The comparative form is what the null was for: does the plus arm
    # stay markedly closer to the basin than the minus arm?
    plus_restorative = (null["plus"]["kl_final"]
                        <= 0.1 * max(null["minus"]["kl_final"], 1e-12))

    # --- drift sweep ---------------------------------------------------------
    rows = []
    for drift in drifts:
        for lam in lams:
            for sign in cfg["sweeps"]["sign"]:
                rs = [run_arm(s, drift, lam, sign, pcfg, steps) for s in seeds]
                rows.extend(rs)
                print(f"  drift={drift:<4} lam={lam:<5} {sign:<6} "
                      f"KL {median([r['kl_first'] for r in rs]):.4f} -> "
                      f"{median([r['kl_final'] for r in rs]):.4f}   "
                      f"task {median([r['task_final'] for r in rs]):.4f}   "
                      f"dist {median([r['dist_final'] for r in rs]):.3f}")

    # --- H2: does minus reach a better basin at matched-or-better task loss? --
    cells, minus_wins = [], 0
    for drift in drifts:
        for lam in lams:
            for s in seeds:
                m = next(r for r in rows if r["sign"] == "minus" and r["seed"] == s
                         and r["drift"] == drift and r["lam"] == lam)
                p = next(r for r in rows if r["sign"] == "plus" and r["seed"] == s
                         and r["drift"] == drift and r["lam"] == lam)
                win = m["kl_final"] < p["kl_final"] and m["task_final"] <= p["task_final"]
                minus_wins += win
                cells.append({"seed": s, "drift": drift, "lam": lam,
                              "kl_minus": m["kl_final"], "kl_plus": p["kl_final"],
                              "task_minus": m["task_final"], "task_plus": p["task_final"],
                              "minus_better_basin": win})
    frac = minus_wins / len(cells) if cells else 0.0

    # --- H3: does doubling lambda increase divergence for the minus arm? ------
    lam_effect = {}
    if len(lams) > 1:
        for lam in lams:
            rs = [r for r in rows if r["sign"] == "minus" and r["lam"] == lam]
            lam_effect[str(lam)] = median([r["kl_final"] for r in rs])
        lo, hi = str(min(lams)), str(max(lams))
        h3 = lam_effect[hi] > lam_effect[lo]
    else:
        h3 = None

    if not plus_restorative:
        verdict = "PRIOR REFUTED"
        reason = (f"the plus arm's KL growth from theta_ref ({null['plus']['kl_final']:.2e}) "
                  f"is not markedly smaller than the minus arm's "
                  f"({null['minus']['kl_final']:.2e}); the KL term is not acting as a "
                  "potential well and nothing else here is interpretable")
    elif frac >= 0.8:
        verdict = "PRIOR REFUTED"
        reason = (f"the documented exploration claim holds: the minus arm reached a better "
                  f"basin at matched-or-better task loss in {frac:.0%} of cells")
    else:
        verdict = "DOCUMENTED CLAIM REFUTED"
        reason = (f"the minus arm reached a better basin at matched-or-better task loss in "
                  f"only {frac:.0%} of cells; 'explores rather than collapsing' is motion "
                  "without arrival")

    metrics = {
        "name": cfg["name"], "config_hash": config_hash, "quick_mode": bool(args.quick),
        "seeds": seeds, "drifts": drifts, "lambdas": lams, "steps": steps,
        "null": null, "null_rows": null_rows,
        "plus_is_restorative": plus_restorative,
        "rows": rows, "cells": cells,
        "frac_cells_minus_better_basin": frac,
        "h3_lambda_increases_divergence": h3, "minus_kl_by_lambda": lam_effect,
        "grading": {"verdict": verdict, "reason": reason},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Objective sign — run summary", "", f"Run at: {stamp}",
        f"Verdict: **{verdict}** — {reason}", "",
        "## Null: started at theta_ref, where KL = 0", "",
        "| sign | final KL | dist from ref |", "|---|---|---|",
    ]
    for sign in cfg["sweeps"]["sign"]:
        lines.append(f"| {sign} | {null[sign]['kl_final']:.6f} "
                     f"| {null[sign]['dist_final']:.4f} |")
    lines += ["", "## Drift sweep (median over seeds)", "",
              "| drift | lambda | sign | KL first | KL final | task final | dist final |",
              "|---|---|---|---|---|---|---|"]
    for drift in drifts:
        for lam in lams:
            for sign in cfg["sweeps"]["sign"]:
                rs = [r for r in rows if r["sign"] == sign and r["drift"] == drift
                      and r["lam"] == lam]
                lines.append(
                    f"| {drift} | {lam} | {sign} | {median([r['kl_first'] for r in rs]):.4f} "
                    f"| {median([r['kl_final'] for r in rs]):.4f} "
                    f"| {median([r['task_final'] for r in rs]):.4f} "
                    f"| {median([r['dist_final'] for r in rs]):.3f} |")
    lines += ["", f"Minus arm reached a better basin at matched-or-better task loss in "
              f"**{frac:.0%}** of cells (needs >= 80% to support the documentation).",
              f"H3 (doubling lambda increases minus-arm divergence): {h3} — "
              f"median final KL by lambda: {lam_effect}",
              "", "Generated by run.py; do not edit."]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE", "sim": cfg["name"],
        "claim": ("CLAUDE.md: the minus sign in task_loss - lambda*safety ensures the "
                  "repair explores the loss landscape rather than collapsing to a local "
                  "minimum"),
        "refute_if": cfg["refute_if"], "verdict": verdict, "reason": reason,
        "frac_cells_minus_better_basin": frac,
        "plus_is_restorative": plus_restorative,
        "metrics_hash": metrics_hash, "config_hash": config_hash,
        "seeds": len(seeds), "null_model": cfg["null_model"],
        "exploratory": bool(args.quick), "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print(f"\n[sign] VERDICT: {verdict} — {reason}")
    print(f"[sign] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
