#!/usr/bin/env python3
"""IP-18 — is kappa_eff a leading indicator of basin breach, and does it beat free?

Turns the informal check 3 in addon_thermodynamic_control/experiment_stability.py into a
pre-registered kill test: Theory A (kappa_eff leads the KL breach and beats a trivial
baseline) against Theory B (coincident/lagging, or no better than free).

The existing check computes its alarm as the run's own 90th percentile of kappa_eff. That
uses future data, always fires (10% of steps exceed it by construction), and is never
scored against a null. Every criterion here is causal -- thresholds are fixed on a warm-up
window and applied forward -- swept across seven settings per config.json, and scored
against both a no-drift null arm and the free ||theta - theta_ref|| baseline.

Scenario note, expanded in REFUTE.md: the framework cannot produce a breach on its own
(basin_kl either starts below epsilon and stays, or starts far above it), so per-step
drift is injected to create the walk-out the claim presupposes.

Tier 2 (torch). Usage: python3 run.py [--config config.json] [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from addon_thermodynamic_control.stability import CoupledDynamicalSystem  # noqa: E402
from simulation.environment import Environment  # noqa: E402


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def kendall_tau(ys) -> float:
    n = len(ys)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (j - i) * (ys[j] - ys[i])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 0.0


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------

def _safety_kl(model_fn, theta, theta_ref, unsafe_inputs):
    curr = F.log_softmax(model_fn(unsafe_inputs, theta), dim=-1)
    ref = F.softmax(model_fn(unsafe_inputs, theta_ref).detach(), dim=-1)
    return F.kl_div(curr, ref, reduction="batchmean")


def adversarial_direction(model_fn, theta, theta_ref, unsafe_inputs, iters: int):
    """Unit top eigenvector of the safety Hessian, oriented to INCREASE safety KL.

    Orientation is not cosmetic. Power iteration returns an arbitrary sign, so an
    unoriented direction alternates and the drift random-walks instead of
    accumulating -- the model then never leaves the basin, which reads as "the
    repair loop handles adversarial drift well" when it is purely an artifact.
    """
    t = theta.detach().requires_grad_(True)
    grad0 = torch.autograd.grad(
        _safety_kl(model_fn, t, theta_ref, unsafe_inputs), t)[0].detach()

    v = torch.randn_like(theta)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        t = theta.detach().requires_grad_(True)
        g = torch.autograd.grad(
            _safety_kl(model_fn, t, theta_ref, unsafe_inputs), t, create_graph=True)[0]
        hv = torch.autograd.grad(g, t, grad_outputs=v)[0].detach()
        n = hv.norm()
        if n < 1e-12:
            break
        v = hv / n
    if torch.dot(v.flatten(), grad0.flatten()) < 0:
        v = -v
    return v


def run_trajectory(seed: int, sigma: float, mode: str, cfg: dict) -> dict:
    """One run: walk theta out of the basin (sigma>0) or hold it near (sigma=0)."""
    sc = cfg["scenario"]
    env = Environment({"drift_strength": sc["initial_drift_strength"], "seed": seed})
    model_fn = env.get_model_fn()
    system = CoupledDynamicalSystem(model_fn, env.theta_ref,
                                    env.task_inputs, cfg["thermo"])
    theta = env.theta_drifted.clone()
    gen = torch.Generator().manual_seed(10_000 + seed)

    series = {"kappa_eff": [], "theta_dist": [], "repair_energy": [], "basin_kl": []}
    for _ in range(sc["steps"]):
        if sigma > 0:
            if mode == "adversarial":
                step_dir = adversarial_direction(
                    model_fn, theta, env.theta_ref, env.safety_inputs,
                    sc["adversarial_power_iters"])
            else:
                step_dir = torch.randn(theta.shape, generator=gen)
            theta = theta + sigma * step_dir
        theta, st = system.step(theta, env.safety_inputs, env.task_inputs, env.task_labels)
        series["kappa_eff"].append(st.kappa_eff)
        series["theta_dist"].append(st.theta_norm)
        series["repair_energy"].append(st.repair_energy_step)
        series["basin_kl"].append(st.basin_kl)

    eps = cfg["thermo"]["epsilon_basin"]
    breach = next((i for i, kl in enumerate(series["basin_kl"]) if kl > eps), None)
    return {"seed": seed, "sigma": sigma, "mode": mode,
            "series": series, "breach_step": breach}


# ---------------------------------------------------------------------------
# causal alarm criteria
# ---------------------------------------------------------------------------

def smooth_causal_median(values: list[float], window: int) -> list[float]:
    """Trailing rolling median. Uses only past and present, never future.

    Window 1 returns the series unchanged, which makes "no smoothing" an ordinary
    point in the sweep rather than a separate code path.
    """
    if window <= 1:
        return list(values)
    out = []
    for t in range(len(values)):
        seg = sorted(values[max(0, t - window + 1):t + 1])
        out.append(seg[len(seg) // 2])
    return out


def alarm_step(values: list[float], crit: dict, warmup: int) -> int | None:
    """First step after warm-up at which the criterion fires, using only past data."""
    base = values[:warmup]
    if not base:
        return None
    kind = crit["kind"]

    if kind in ("ratio", "z"):
        srt = sorted(base)
        med = srt[len(srt) // 2]
        mu = sum(base) / len(base)
        var = sum((b - mu) ** 2 for b in base) / max(1, len(base) - 1)
        sd = var ** 0.5
        thresh = crit["k"] * med if kind == "ratio" else mu + crit["k"] * sd
        for t in range(warmup, len(values)):
            if values[t] > thresh:
                return t
        return None

    if kind == "tau":
        w = crit["window"]
        for t in range(max(warmup, w - 1), len(values)):
            if kendall_tau(values[t - w + 1:t + 1]) > crit["tau_star"]:
                return t
        return None

    raise ValueError(f"unknown criterion kind {kind!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, 1 sigma, fewer steps (marked in output)")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))

    seeds = list(cfg["seeds"])
    sigmas = list(cfg["sweeps"]["drift_sigma"])
    modes = list(cfg["sweeps"]["drift_mode"])
    windows = list(cfg["sweeps"]["smoothing_window"])
    if args.quick:
        seeds, sigmas, windows = seeds[:2], sigmas[1:2], windows[:2]
        cfg["scenario"] = {**cfg["scenario"], "steps": 20}

    warmup = cfg["scenario"]["warmup"]
    signals = cfg["signals"]
    baseline_sig = cfg["trivial_baseline"]

    print(f"[ip18] drift arm: {len(seeds)} seeds x {len(sigmas)} sigmas x "
          f"{len(modes)} modes; null arm: {len(seeds)} seeds; "
          f"{cfg['scenario']['steps']} steps each")

    drift_runs = []
    for md in modes:
        for sg in sigmas:
            for s in seeds:
                r = run_trajectory(s, sg, md, cfg)
                drift_runs.append(r)
                kl = r["series"]["basin_kl"]
                ke = r["series"]["kappa_eff"]
                print(f"  {md:<11} sigma={sg:<6} seed={s}  breach@{r['breach_step']}  "
                      f"kl {kl[0]:.3f}->{kl[-1]:.3f}  "
                      f"kappa {min(ke):.4f}-{max(ke):.4f}")

    # sigma=0 makes the mode irrelevant, so one null arm serves both.
    null_runs = [run_trajectory(s, 0.0, "isotropic", cfg) for s in seeds]
    n_null_breached = sum(r["breach_step"] is not None for r in null_runs)
    print(f"  null arm: {n_null_breached}/{len(null_runs)} runs breached "
          f"(expected 0; any breach invalidates the null)")

    # --- score every (criterion, signal) ------------------------------------
    max_fp = cfg["grading"]["max_null_false_alarm_rate"]
    gate = cfg["grading"]["cell_fraction_gate"]
    usable = [r for r in drift_runs
              if r["breach_step"] is not None and r["breach_step"] > warmup]

    per_criterion = []
    for win in windows:
      for crit in cfg["criteria"]:
        # Smoothing is applied to every signal, baseline included; smoothing the
        # candidate alone would be an unfair comparison rather than a measurement.
        def sm(series):
            return smooth_causal_median(series, win)

        fp = {}
        for sig in signals:
            fired = sum(alarm_step(sm(r["series"][sig]), crit, warmup) is not None
                        for r in null_runs)
            fp[sig] = fired / len(null_runs)
        viable = fp["kappa_eff"] <= max_fp and fp[baseline_sig] <= max_fp

        cells = []
        for r in usable:
            b = r["breach_step"]
            rec = {"seed": r["seed"], "sigma": r["sigma"], "mode": r["mode"],
                   "breach_step": b}
            for sig in signals:
                a = alarm_step(sm(r["series"][sig]), crit, warmup)
                rec[f"alarm_{sig}"] = a
                rec[f"lead_{sig}"] = (b - a) if a is not None else None
            lk, lb = rec["lead_kappa_eff"], rec[f"lead_{baseline_sig}"]
            rec["kappa_leads"] = lk is not None and lk > 0
            rec["kappa_beats_baseline"] = (
                lk is not None and lk > 0 and (lb is None or lk > lb)
            )
            rec["cell_supports_A"] = rec["kappa_leads"] and rec["kappa_beats_baseline"]
            cells.append(rec)

        n = len(cells)
        frac_lead = sum(c["kappa_leads"] for c in cells) / n if n else 0.0
        frac_A = sum(c["cell_supports_A"] for c in cells) / n if n else 0.0
        by_mode = {}
        for md in modes:
            sub = [c for c in cells if c["mode"] == md]
            by_mode[md] = {
                "n_cells": len(sub),
                "frac_kappa_leads": (sum(c["kappa_leads"] for c in sub) / len(sub)) if sub else 0.0,
                "frac_supports_A": (sum(c["cell_supports_A"] for c in sub) / len(sub)) if sub else 0.0,
            }
        per_criterion.append({
            "criterion": crit["id"], "spec": crit, "smoothing_window": win,
            "null_false_alarm": fp, "viable": viable,
            "frac_cells_kappa_leads": frac_lead,
            "frac_cells_support_theory_A": frac_A,
            "by_mode": by_mode,
            "supports_A": viable and frac_A >= gate,
            "mean_lead_kappa": mean([c["lead_kappa_eff"] for c in cells]),
            "mean_lead_baseline": mean([c[f"lead_{baseline_sig}"] for c in cells]),
            "cells": cells,
        })
        modes_txt = "  ".join(
            f"{md[:5]}: leads {by_mode[md]['frac_kappa_leads']:.0%}/A "
            f"{by_mode[md]['frac_supports_A']:.0%}" for md in modes)
        print(f"  w={win} {crit['id']:<12} viable={str(viable):<5} "
              f"FP(k)={fp['kappa_eff']:.2f} FP(th)={fp[baseline_sig]:.2f}  "
              f"all: leads {frac_lead:.0%}/A {frac_A:.0%}   {modes_txt}")

    # C1: does smoothing do what smoothing is for? Median null FP per window.
    fp_by_window = {}
    for win in windows:
        sub = [p["null_false_alarm"]["kappa_eff"] for p in per_criterion
               if p["smoothing_window"] == win]
        srt = sorted(sub)
        fp_by_window[str(win)] = srt[len(srt) // 2] if srt else None
    ordered = [fp_by_window[str(w)] for w in windows]
    c1_fp_falls = all(b <= a for a, b in zip(ordered, ordered[1:])) and ordered[-1] < ordered[0]

    viable_crits = [p for p in per_criterion if p["viable"]]
    if not usable:
        verdict, reason = "VOID", "no usable cells: breach never occurred after warm-up"
    elif n_null_breached:
        verdict = "VOID"
        reason = f"null arm breached in {n_null_breached} runs; it is not a null"
    elif not viable_crits:
        verdict = "INCONCLUSIVE"
        reason = (f"no criterion kept its null false-alarm rate at or below {max_fp}; "
                  "nothing is left to grade")
    elif any(p["supports_A"] for p in viable_crits):
        winners = [f"{p['criterion']}@w{p['smoothing_window']}"
                   for p in viable_crits if p["supports_A"]]
        verdict = "SUPPORTED"
        reason = (f"Theory A holds under viable criteria: {', '.join(winners)} "
                  f"({len(winners)} of {len(per_criterion)} combinations tried)")
    elif all(p["frac_cells_support_theory_A"] <= (1 - gate) for p in viable_crits):
        verdict = "REFUTED"
        reason = (f"Theory B: under all {len(viable_crits)} viable criteria, kappa_eff "
                  f"failed to both lead and beat the free {baseline_sig} baseline at "
                  f">={gate:.0%} of cells")
    else:
        verdict = "INCONCLUSIVE"
        reason = "viable criteria disagree; see the per-criterion table"

    metrics = {
        "name": cfg["name"], "config_hash": config_hash, "quick_mode": bool(args.quick),
        "seeds": seeds, "sigmas": sigmas, "modes": modes, "windows": windows,
        "n_usable_cells": len(usable), "n_drift_runs": len(drift_runs),
        "null_arm": {"n_runs": len(null_runs), "n_breached": n_null_breached,
                     "breach_steps": [r["breach_step"] for r in null_runs]},
        "breach_steps": [{"seed": r["seed"], "sigma": r["sigma"], "mode": r["mode"],
                          "breach_step": r["breach_step"]} for r in drift_runs],
        "per_criterion": per_criterion,
        "smoothing_c1": {"median_null_fp_kappa_by_window": fp_by_window,
                         "falls_with_window": c1_fp_falls},
        "grading": {"verdict": verdict, "reason": reason,
                    "n_viable_criteria": len(viable_crits),
                    "n_combinations": len(per_criterion)},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# IP-18 kappa_eff leading-indicator kill test — run summary", "",
        f"Run at: {stamp}", f"Verdict: **{verdict}** — {reason}", "",
        "## Theory A vs Theory B", "",
        "- A: kappa_eff alarms before the basin breach AND beats the free theta-distance baseline.",
        "- B: coincident/lagging, or no better than free.", "",
        "## Criterion sweep (the point of this sim)", "",
        "| w | criterion | null FP (kappa) | null FP (theta) | viable | kappa leads | supports A | mean lead kappa | mean lead theta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in per_criterion:
        mk = p["mean_lead_kappa"]
        mb = p["mean_lead_baseline"]
        lines.append(
            f"| {p['smoothing_window']} | {p['criterion']} "
            f"| {p['null_false_alarm']['kappa_eff']:.2f} "
            f"| {p['null_false_alarm'][baseline_sig]:.2f} | {p['viable']} "
            f"| {p['frac_cells_kappa_leads']:.0%} | {p['frac_cells_support_theory_A']:.0%} "
            f"| {'n/a' if mk is None else f'{mk:.1f}'} "
            f"| {'n/a' if mb is None else f'{mb:.1f}'} |"
        )
    lines += [
        "", "## C1: median null FP for kappa_eff by smoothing window", "",
        "  " + " · ".join(f"w={w}: {fp_by_window[str(w)]:.2f}" for w in windows),
        f"  falls with window: {c1_fp_falls}",
        "", f"Usable cells: {len(usable)}/{len(drift_runs)} · "
        f"null runs breaching: {n_null_breached}/{len(null_runs)} (must be 0)",
        "", "Generated by run.py; do not edit.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE", "sim": cfg["name"],
        "claim": ("kappa_eff spikes before basin_kl exceeds epsilon and beats a trivial "
                  "baseline (IP-18; stability.py line 32; experiment_stability.py check 3)"),
        "refute_if": cfg["refute_if"], "verdict": verdict, "reason": reason,
        "metrics_hash": metrics_hash, "config_hash": config_hash,
        "seeds": len(seeds), "null_model": cfg["null_model"],
        "n_viable_criteria": len(viable_crits),
        "n_combinations_swept": len(per_criterion),
        "smoothing_reduces_false_alarms": c1_fp_falls,
        "exploratory": bool(args.quick), "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print("\n[ip18] C1 median null FP(kappa) by window: " +
          ", ".join(f"w={w}:{fp_by_window[str(w)]:.2f}" for w in windows) +
          f"  falls={c1_fp_falls}")
    print(f"[ip18] VERDICT: {verdict} — {reason}")
    print(f"[ip18] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
