#!/usr/bin/env python3
"""E-P8 snap latency — does a snap event decode its own initial condition?

Harness-conformant rebuild of the E-P8 test. The previous sim in this lineage,
`experiments/snap_information_sim.py`, never ran a snap at all: it launched from the
stable well minimum with zero velocity and no ramp, so its trajectories did not move
(measured range 3.19e-10) and its printed conclusions were unsupported by its own output.

The fix is not a displaced launch — it is that the load must actually ramp. Here the
system starts on the stable branch, exactly as a real strut sits at rest, and a constant
compression ramp carries it through the fold. The snap is a consequence of the protocol
rather than of the initial condition.

Model: the saddle-node (fold) normal form, overdamped,

    dx/dt = K*(c(t) - c_snap) + x^2,      c(t) = eps0 + rate*t

with the trajectory started on the stable branch x = -sqrt(-mu_0). Linearizing about that
branch gives a relaxation rate 2*sqrt(-mu), i.e. tau proportional to
1/sqrt(1 - c/c_snap) — the same stiffness law the rest of the ecosystem uses. Time units
are arbitrary (K = 1); neither graded quantity depends on them, since the exponent is
dimensionless and the decoder RMSE is in compression units.

Conforms to docs/research/HARNESS.md: config-driven, >=5 seeds, a swept parameter, a
named null, a pre-committed refutation condition transcribed from notes/15, a verdict
graded against the data rather than asserted, and a ledger entry.

Stdlib only (Tier 0). Usage: python3 run.py [--config config.json] [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def canonical(obj) -> str:
    """Canonical JSON so hashes are reproducible across runs and machines."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares; returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def rmse(pred: list[float], true: list[float]) -> float:
    n = len(true)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / n)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def snap_time(eps0: float, rate: float, c_snap: float, K: float,
              dt: float, x_esc: float, t_max: float) -> float | None:
    """Integrate the ramped fold normal form; return the time x crosses x_esc.

    The trajectory starts on the stable branch, so any escape is driven by the ramp
    carrying the system through the saddle-node — not by the launch condition.
    Crossing is linearly interpolated within the final step so the result is not
    quantized to dt.
    """
    mu0 = K * (eps0 - c_snap)
    if mu0 >= 0.0:
        return 0.0  # already past threshold; nothing to ramp through
    x = -math.sqrt(-mu0)
    t = 0.0

    def f(tt: float, xx: float) -> float:
        return K * (eps0 + rate * tt - c_snap) + xx * xx

    while t < t_max:
        k1 = f(t, x)
        k2 = f(t + dt / 2.0, x + dt * k1 / 2.0)
        k3 = f(t + dt / 2.0, x + dt * k2 / 2.0)
        k4 = f(t + dt, x + dt * k3)
        x_new = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if x_new >= x_esc:
            frac = (x_esc - x) / (x_new - x)
            return t + frac * dt
        x = x_new
        t += dt
    return None


# ---------------------------------------------------------------------------
# one (seed, ramp_rate) cell
# ---------------------------------------------------------------------------

def run_cell(seed: int, rate: float, cfg: dict) -> dict:
    """Run every eps0 level x repeat for one seed and ramp rate, then grade it."""
    m = cfg["model"]
    p = cfg["protocol"]
    nz = cfg["noise"]
    c_snap = m["c_snap"]
    rng = random.Random((seed << 8) ^ hash(round(rate * 1e6)) & 0xFFFFFFFF)

    t_max = (c_snap - min(p["eps0_levels"])) / rate * 3.0 + m["t_max_slack"]

    eps0_obs: list[float] = []
    t_obs: list[float] = []
    for eps0 in p["eps0_levels"]:
        for _ in range(p["repeats_per_level"]):
            # PETG fatigue: the true threshold wanders trial to trial. The experimenter
            # does not know the wandered value and fits against the nominal c_snap.
            c_eff = c_snap + rng.gauss(0.0, nz["threshold_wander_sd"])
            t = snap_time(eps0, rate, c_eff, m["K"], m["dt"], m["x_esc"], t_max)
            if t is None or t <= 0.0:
                continue
            t_noisy = t * (1.0 + rng.gauss(0.0, nz["timing_noise_frac"]))
            if t_noisy <= 0.0:
                continue
            eps0_obs.append(eps0)
            t_obs.append(t_noisy)

    n = len(t_obs)
    if n < 10:
        return {"seed": seed, "ramp_rate": rate, "n_trials": n, "usable": False}

    # --- exponent fits -----------------------------------------------------
    # Primary regressor follows the protocol text: log t_snap vs log(eps-distance).
    log_t = [math.log(t) for t in t_obs]
    log_dist = [math.log(c_snap - e) for e in eps0_obs]
    exp_distance, _ = ols(log_dist, log_t)
    # Secondary: the regressor the claim sentence names (log eps_0). Reported, not graded.
    log_eps0 = [math.log(e) for e in eps0_obs]
    exp_eps0, _ = ols(log_eps0, log_t)

    # --- decoder: t_snap -> eps0, held-out ---------------------------------
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(2, int(round(p["holdout_frac"] * n)))
    test_i, train_i = idx[:n_test], idx[n_test:]

    def fit_and_score(pairs_t: list[float], pairs_e: list[float]) -> float:
        tr_t = [pairs_t[i] for i in train_i]
        tr_e = [pairs_e[i] for i in train_i]
        b, a = ols(tr_t, tr_e)
        pred = [a + b * pairs_t[i] for i in test_i]
        return rmse(pred, [pairs_e[i] for i in test_i])

    rmse_decoder = fit_and_score(t_obs, eps0_obs)

    train_mean = sum(eps0_obs[i] for i in train_i) / len(train_i)
    rmse_baseline = rmse([train_mean] * n_test, [eps0_obs[i] for i in test_i])

    # Null: destroy only the eps0 <-> t_snap correspondence, keep both marginals.
    shuffled_t = list(t_obs)
    rng.shuffle(shuffled_t)
    rmse_null = fit_and_score(shuffled_t, eps0_obs)

    # --- grade against the pre-committed criteria --------------------------
    exponent_in_band = -0.65 <= exp_distance <= -0.35
    decoder_rmse_ok = rmse_decoder <= 0.02
    beats_baseline = rmse_decoder < rmse_baseline
    null_is_clean = rmse_null >= 0.9 * rmse_baseline

    return {
        "seed": seed,
        "ramp_rate": rate,
        "n_trials": n,
        "usable": True,
        "exponent_vs_distance": exp_distance,
        "exponent_vs_eps0": exp_eps0,
        "rmse_decoder": rmse_decoder,
        "rmse_baseline": rmse_baseline,
        "rmse_null": rmse_null,
        "mean_t_snap": sum(t_obs) / n,
        "exponent_in_band": exponent_in_band,
        "decoder_rmse_ok": decoder_rmse_ok,
        "beats_baseline": beats_baseline,
        "null_is_clean": null_is_clean,
        "cell_passes": exponent_in_band and decoder_rmse_ok and beats_baseline,
        "cell_refutes": (not exponent_in_band) or (not beats_baseline),
    }


# ---------------------------------------------------------------------------
# numerics convergence check
# ---------------------------------------------------------------------------

def numerics_check(cfg: dict) -> dict:
    """Re-integrate a few trials at dt/4; large deviation invalidates the run."""
    nc = cfg["numerics_check"]
    if not nc.get("enabled", True):
        return {"enabled": False}
    m = cfg["model"]
    rate = cfg["sweeps"]["ramp_rate"][len(cfg["sweeps"]["ramp_rate"]) // 2]
    c_snap = m["c_snap"]
    t_max = (c_snap - min(cfg["protocol"]["eps0_levels"])) / rate * 3.0 + m["t_max_slack"]
    worst = 0.0
    rows = []
    for eps0 in nc["sample_levels"]:
        coarse = snap_time(eps0, rate, c_snap, m["K"], m["dt"], m["x_esc"], t_max)
        fine = snap_time(eps0, rate, c_snap, m["K"], nc["refined_dt"], m["x_esc"], t_max)
        rel = abs(coarse - fine) / fine if fine else float("nan")
        worst = max(worst, rel)
        rows.append({"eps0": eps0, "t_dt": coarse, "t_dt_refined": fine, "rel_dev": rel})
    return {
        "enabled": True,
        "ramp_rate": rate,
        "rows": rows,
        "max_rel_deviation": worst,
        "converged": worst < 1e-3,
    }


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def grade(cells: list[dict], rates: list[float]) -> dict:
    usable = [c for c in cells if c["usable"]]
    if not usable:
        return {"verdict": "INCONCLUSIVE", "reason": "no usable cells"}

    n = len(usable)
    frac_pass = sum(c["cell_passes"] for c in usable) / n
    frac_refute = sum(c["cell_refutes"] for c in usable) / n
    nulls_clean = all(c["null_is_clean"] for c in usable)

    per_rate_pass = {}
    for r in rates:
        sub = [c for c in usable if c["ramp_rate"] == r]
        per_rate_pass[str(r)] = (sum(c["cell_passes"] for c in sub) / len(sub)) if sub else 0.0

    supported = frac_pass >= 0.8 and all(v >= 0.8 for v in per_rate_pass.values()) and nulls_clean
    if supported:
        verdict, reason = "SUPPORTED", f"pass condition held at {frac_pass:.0%} of cells"
    elif frac_refute >= 0.8:
        verdict = "REFUTED"
        n_band = sum(not c["exponent_in_band"] for c in usable)
        n_base = sum(not c["beats_baseline"] for c in usable)
        reason = (f"refutation condition held at {frac_refute:.0%} of cells "
                  f"(exponent outside band in {n_band}/{n}, "
                  f"decoder failed to beat baseline in {n_base}/{n})")
    else:
        verdict = "INCONCLUSIVE"
        reason = f"pass at {frac_pass:.0%}, refute at {frac_refute:.0%} of cells"

    return {
        "verdict": verdict,
        "reason": reason,
        "frac_cells_pass": frac_pass,
        "frac_cells_refute": frac_refute,
        "per_rate_pass_fraction": per_rate_pass,
        "nulls_clean": nulls_clean,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def write_summary(path: Path, cfg: dict, metrics: dict, stamp: str) -> None:
    g = metrics["grading"]
    usable = [c for c in metrics["cells"] if c["usable"]]
    lines = [
        "# E-P8 snap latency — run summary",
        "",
        f"Run at: {stamp}",
        f"Verdict: **{g['verdict']}** — {g['reason']}",
        "",
        "## Claim",
        "",
        "t_snap encodes the initial distance-from-threshold via a fold law",
        "t_snap ~ eps_0^(-1/2), so snap latency decodes the initial condition.",
        "",
        "## Pre-committed criteria (notes/15 E-P8, verbatim)",
        "",
        "- Pass: exponent in [-0.65, -0.35] AND decoder RMSE <= 0.02 compression.",
        "- Refuted if: exponent outside band OR decoder no better than mean-eps0 baseline.",
        "",
        "## Measured",
        "",
        "| ramp_rate | exponent vs distance | exponent vs eps_0 | RMSE decoder | RMSE baseline | RMSE null |",
        "|---|---|---|---|---|---|",
    ]
    for r in cfg["sweeps"]["ramp_rate"]:
        sub = [c for c in usable if c["ramp_rate"] == r]
        if not sub:
            continue
        lines.append(
            f"| {r} | {mean([c['exponent_vs_distance'] for c in sub]):+.3f} "
            f"| {mean([c['exponent_vs_eps0'] for c in sub]):+.3f} "
            f"| {mean([c['rmse_decoder'] for c in sub]):.4f} "
            f"| {mean([c['rmse_baseline'] for c in sub]):.4f} "
            f"| {mean([c['rmse_null'] for c in sub]):.4f} |"
        )
    nc = metrics["numerics_check"]
    lines += [
        "",
        f"Cells passing: {g['frac_cells_pass']:.0%} · cells meeting refutation "
        f"condition: {g['frac_cells_refute']:.0%} · nulls clean: {g['nulls_clean']}",
        "",
        "## Numerics",
        "",
        f"Max relative deviation at dt/4: {nc.get('max_rel_deviation', float('nan')):.2e} "
        f"(converged: {nc.get('converged')})",
        "",
        "Seeds: " + ", ".join(str(s) for s in cfg["seeds"]),
        "",
        "Generated by run.py; do not edit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds and 1 ramp rate, for a fast smoke run (marked in output)")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = json.loads(cfg_text)
    config_hash = sha256_str(canonical(cfg))

    seeds = cfg["seeds"]
    rates = cfg["sweeps"]["ramp_rate"]
    if args.quick:
        seeds, rates = seeds[:2], rates[1:2]

    print(f"[ep8] {len(seeds)} seeds x {len(rates)} ramp rates "
          f"x {len(cfg['protocol']['eps0_levels'])} levels "
          f"x {cfg['protocol']['repeats_per_level']} repeats")

    cells = []
    for r in rates:
        for s in seeds:
            cell = run_cell(s, r, cfg)
            cells.append(cell)
            if cell["usable"]:
                print(f"  rate={r:<6} seed={s}  exponent={cell['exponent_vs_distance']:+.3f}  "
                      f"rmse={cell['rmse_decoder']:.4f} (baseline {cell['rmse_baseline']:.4f}, "
                      f"null {cell['rmse_null']:.4f})  pass={cell['cell_passes']}")
            else:
                print(f"  rate={r:<6} seed={s}  UNUSABLE ({cell['n_trials']} trials)")

    nc = numerics_check(cfg)
    grading = grade(cells, rates)

    metrics = {
        "name": cfg["name"],
        "config_hash": config_hash,
        "quick_mode": bool(args.quick),
        "seeds": seeds,
        "ramp_rates": rates,
        "cells": cells,
        "numerics_check": nc,
        "grading": grading,
    }
    # NOTE: no timestamp inside metrics.json, so metrics_hash is a true content hash and
    # an unchanged run reproduces an unchanged hash.
    metrics_text = canonical(metrics)
    metrics_hash = sha256_str(metrics_text)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary(outdir / "summary.md", cfg, metrics, stamp)

    ledger = {
        "type": "MEASURE",
        "sim": cfg["name"],
        "claim": ("t_snap encodes initial distance-from-threshold via t_snap ~ eps_0^(-1/2); "
                  "snap latency decodes the initial condition (E-P8, notes/15)"),
        "refute_if": cfg["refute_if"],
        "verdict": grading["verdict"],
        "reason": grading["reason"],
        "metrics_hash": metrics_hash,
        "config_hash": config_hash,
        "seeds": len(seeds),
        "null_model": cfg["null_model"],
        "numerics_converged": nc.get("converged"),
        "exploratory": bool(args.quick),
        "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print(f"\n[ep8] VERDICT: {grading['verdict']} — {grading['reason']}")
    print(f"[ep8] numerics converged: {nc.get('converged')} "
          f"(max rel deviation {nc.get('max_rel_deviation', float('nan')):.2e})")
    print(f"[ep8] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
