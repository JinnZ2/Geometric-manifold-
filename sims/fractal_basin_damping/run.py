#!/usr/bin/env python3
"""Fractal basin boundary vs damping — HARNESS retrofit of fractal_basin_sim.py.

`experiments/fractal_basin_sim.py` measured the uncertainty exponent alpha at a single
damping (gamma = 0.25). `docs/research/HARNESS.md` lists that as retrofit queue item 4,
with the deficiency stated plainly: "alpha measured at single damping — sweep gamma
mandatory." gamma was a parameter of the original basin_grid() but was never varied.

This sweeps it, adds the smooth-boundary null the original had no way to fail, and grades
itself against the criteria in REFUTE.md.

alpha is the uncertainty exponent of the basin boundary (Grebogi/McDonald/Ott/Yorke):
f(eps) ~ eps^alpha, where f is the fraction of eps-separated initial-condition pairs
landing in different basins. alpha = 1 is a smooth boundary; alpha -> 0 is maximally
fractal. Boundary dimension D_b = 2 - alpha on this 2-D section. Physically alpha is an
exchange rate: doubling measurement precision buys 2^alpha times the outcome certainty.

Tier 1 (numpy): the grid integration is 200x200 initial conditions x 2400 steps, which is
not reachable in pure stdlib. The harness scaffolding around it is stdlib.

Usage: python3 run.py [--config config.json] [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# dynamics + estimators (behaviour preserved from the original sim)
# ---------------------------------------------------------------------------

def basin_grid(centers, N, xr, vr, dt, T, gamma):
    """Integrate every initial condition at once; return the basin label field."""
    def F(x):
        h = 1e-5

        def E(z):
            return np.prod([(z - c) ** 2 for c in centers], axis=0)
        return -(E(x + h) - E(x - h)) / (2 * h)

    xs = np.linspace(*xr, N)
    vs = np.linspace(*vr, N)
    X, V = np.meshgrid(xs, vs)
    for _ in range(int(T / dt)):
        V += dt * (F(X) - gamma * V)
        X += dt * V
    G = np.argmin(np.abs(X[..., None] - np.array(centers)), axis=-1)
    return G.astype(int), xs


def uncertainty_exponent(G, xs, n_probe, n_scales, rng):
    """Fit f(eps) ~ eps^alpha over dyadic scales; returns (alpha, r_squared)."""
    N = G.shape[0]
    dx = xs[1] - xs[0]
    scales = []
    fs = []
    for eps in dx * 2.0 ** np.arange(0, n_scales):
        dj = max(1, int(round(eps / dx)))
        if N - 2 - dj <= 2:
            break  # scale exceeds the grid; drop it rather than sampling an empty range
        i = rng.integers(2, N - 2, n_probe)
        j = rng.integers(2, N - 2 - dj, n_probe)
        scales.append(eps)
        fs.append(np.mean(G[i, j] != G[i, j + dj]))
    epss = np.array(scales)
    fs = np.array(fs)
    m = fs > 0
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(epss[m]), np.log(fs[m])
    alpha, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (alpha * lx + intercept)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(r2)


def wada_fraction(G, rad):
    N = G.shape[0]
    tot = wada = 0
    for i in range(rad, N - rad):
        row = G[i - rad:i + rad + 1]
        for j in range(rad, N - rad):
            u = np.unique(row[:, j - rad:j + rad + 1])
            if len(u) > 1:
                tot += 1
                wada += (len(u) == 3)
    return wada / max(tot, 1), tot


def smooth_control_field(N, xs):
    """Null: labels from a smooth analytic split, no dynamics. True alpha = 1."""
    x_mid = 0.5 * (xs[0] + xs[-1])
    row = (xs >= x_mid).astype(int)
    return np.tile(row, (N, 1))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def kendall_tau(xs, ys) -> float:
    n = len(xs)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 0.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true",
                    help="coarse grid and 2 gammas, for a fast smoke run (marked in output)")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))

    grid = dict(cfg["grid"])
    gammas = list(cfg["sweeps"]["gamma"])
    seeds = list(cfg["seeds"])
    est = cfg["estimator"]
    if args.quick:
        grid["N"] = 60
        grid["T"] = 40.0
        gammas = [gammas[0], gammas[-1]]
        seeds = seeds[:3]

    print(f"[fractal] {len(gammas)} gammas x {len(cfg['systems'])} systems, "
          f"N={grid['N']} T={grid['T']}, {len(seeds)} probe seeds")

    fields = {}
    rows = []
    for sysname, sc in cfg["systems"].items():
        for g in gammas:
            G, xs = basin_grid(sc["centers"], grid["N"], tuple(sc["xr"]), tuple(sc["vr"]),
                               grid["dt"], grid["T"], g)
            fields[(sysname, g)] = (G, xs)
            for s in seeds:
                rng = np.random.default_rng(s)
                a, r2 = uncertainty_exponent(G, xs, est["n_probe"], est["n_scales"], rng)
                rows.append({"system": sysname, "gamma": g, "seed": s,
                             "alpha": a, "fit_r2": r2, "d_boundary": 2.0 - a})
            amean = float(np.mean([r["alpha"] for r in rows
                                   if r["system"] == sysname and r["gamma"] == g]))
            print(f"  {sysname:<7} gamma={g:<5} alpha={amean:.3f}  D_b={2 - amean:.3f}")

    # Wada fraction (deterministic per field; triple-well only — needs 3 basins)
    wada_rows = []
    if cfg["wada"]["enabled"]:
        for g in gammas:
            G, _ = fields[("triple", g)]
            wf, tot = wada_fraction(G, cfg["wada"]["radius"])
            wada_rows.append({"gamma": g, "wada_fraction": wf, "boundary_cells": tot})
            print(f"  wada    gamma={g:<5} {wf * 100:.1f}% of {tot} boundary cells")

    # --- null: smooth control field -----------------------------------------
    any_sys = next(iter(cfg["systems"]))
    _, xs_ref = fields[(any_sys, gammas[0])]
    Gnull = smooth_control_field(grid["N"], xs_ref)
    null_alphas = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        a, _ = uncertainty_exponent(Gnull, xs_ref, est["n_probe"], est["n_scales"], rng)
        null_alphas.append(a)
    null_alpha = float(np.mean(null_alphas))
    null_ok = null_alpha >= 0.90
    print(f"  NULL    smooth boundary alpha={null_alpha:.3f} (need >= 0.90) -> ok={null_ok}")

    # --- C1 regression check against notes/17 --------------------------------
    rc = cfg["reproduction_check"]
    c1 = {"gamma": rc["gamma"], "checked": rc["gamma"] in gammas, "tolerance": rc["tolerance"]}
    if c1["checked"]:
        for sysname, key in (("double", "expected_alpha_double"),
                             ("triple", "expected_alpha_triple")):
            got = float(np.mean([r["alpha"] for r in rows
                                 if r["system"] == sysname and r["gamma"] == rc["gamma"]]))
            c1[sysname] = {"expected": rc[key], "measured": got,
                           "within_tolerance": abs(got - rc[key]) <= rc["tolerance"]}
        if wada_rows:
            wg = [w for w in wada_rows if w["gamma"] == rc["gamma"]]
            if wg:
                c1["wada"] = {"expected": rc["expected_wada_fraction"],
                              "measured": wg[0]["wada_fraction"],
                              "within_tolerance":
                                  abs(wg[0]["wada_fraction"] - rc["expected_wada_fraction"])
                                  <= rc["tolerance"]}
        c1["passes"] = all(v["within_tolerance"] for k, v in c1.items()
                           if isinstance(v, dict) and "within_tolerance" in v)
    else:
        c1["passes"] = None

    # --- C2 sweep claim: alpha rises with gamma ------------------------------
    c2_per_seed = []
    for sysname in cfg["systems"]:
        for s in seeds:
            pts = sorted([(r["gamma"], r["alpha"]) for r in rows
                          if r["system"] == sysname and r["seed"] == s])
            gs = [p[0] for p in pts]
            als = [p[1] for p in pts]
            tau = kendall_tau(gs, als)
            span = als[-1] - als[0]
            c2_per_seed.append({"system": sysname, "seed": s, "kendall_tau": tau,
                                "alpha_span": span,
                                "supports": tau > 0 and span >= 0.10})
    frac_support = sum(c["supports"] for c in c2_per_seed) / len(c2_per_seed)

    if not null_ok:
        verdict = "VOID"
        reason = (f"smooth-boundary null returned alpha={null_alpha:.3f} < 0.90; "
                  "the estimator reports fractality where none exists")
    elif frac_support >= 0.8:
        verdict = "SUPPORTED"
        reason = f"alpha rises with damping at {frac_support:.0%} of (system, seed) cells"
    elif frac_support <= 0.2:
        verdict = "REFUTED"
        reason = (f"alpha does not rise with damping — the pre-committed trend held at only "
                  f"{frac_support:.0%} of cells")
    else:
        verdict = "INCONCLUSIVE"
        reason = f"trend held at {frac_support:.0%} of cells, between the 20% and 80% gates"

    metrics = {
        "name": cfg["name"],
        "config_hash": config_hash,
        "quick_mode": bool(args.quick),
        "gammas": gammas,
        "seeds": seeds,
        "alpha_rows": rows,
        "wada_rows": wada_rows,
        "null": {"alpha": null_alpha, "per_seed": null_alphas, "ok": null_ok},
        "c1_reproduction": c1,
        "c2_per_cell": c2_per_seed,
        "grading": {"verdict": verdict, "reason": reason,
                    "frac_cells_supporting": frac_support},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fractal basin boundary vs damping — run summary",
        "", f"Run at: {stamp}", f"Verdict: **{verdict}** — {reason}", "",
        "## alpha vs damping", "",
        "| system | gamma | alpha (mean over seeds) | D_boundary | fit R^2 |",
        "|---|---|---|---|---|",
    ]
    for sysname in cfg["systems"]:
        for g in gammas:
            sel = [r for r in rows if r["system"] == sysname and r["gamma"] == g]
            if sel:
                lines.append(f"| {sysname} | {g} | {np.mean([r['alpha'] for r in sel]):.3f} "
                             f"| {np.mean([r['d_boundary'] for r in sel]):.3f} "
                             f"| {np.mean([r['fit_r2'] for r in sel]):.3f} |")
    if wada_rows:
        lines += ["", "## Wada fraction (triple well)", "",
                  "| gamma | Wada fraction | boundary cells |", "|---|---|---|"]
        lines += [f"| {w['gamma']} | {w['wada_fraction'] * 100:.1f}% | {w['boundary_cells']} |"
                  for w in wada_rows]
    lines += [
        "", "## Null and regression", "",
        f"- Smooth-boundary null: alpha = {null_alpha:.3f} (needs >= 0.90) -> ok = {null_ok}",
        f"- notes/17 reproduction at gamma={rc['gamma']}: passes = {c1.get('passes')}",
        "", "Generated by run.py; do not edit.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE",
        "sim": cfg["name"],
        "claim": ("basin-boundary uncertainty exponent alpha rises with damping gamma "
                  "(HARNESS retrofit queue item 4: single-damping alpha needs a sweep)"),
        "refute_if": cfg["refute_if"],
        "verdict": verdict,
        "reason": reason,
        "metrics_hash": metrics_hash,
        "config_hash": config_hash,
        "seeds": len(seeds),
        "null_model": cfg["null_model"],
        "null_alpha": null_alpha,
        "reproduces_notes17": c1.get("passes"),
        "exploratory": bool(args.quick),
        "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print(f"\n[fractal] VERDICT: {verdict} — {reason}")
    print(f"[fractal] notes/17 reproduction: {c1.get('passes')}")
    print(f"[fractal] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
