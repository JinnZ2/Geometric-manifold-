#!/usr/bin/env python3
"""Is basin escape rate-induced (control saturation) rather than curvature-induced?

Three rounds of sims/kappa_eff_leading/ refuted kappa_eff as a leading indicator. Looking
at why turned up a structural reason: CoupledDynamicalSystem caps every repair step at
trust_r = lr / (1 + mu * max(fisher)) <= lr, and in the basin coordinate that cap makes the
repair remove a roughly constant amount of KL per step no matter how hard the system is
driven, while injected drift adds KL in proportion to sigma^2.

A capped corrector against quadratically growing forcing gives a parameter-free critical
rate:

    dKL/step(net) = k * sigma^2 - repair_cap        =>   sigma_crit = sqrt(repair_cap / k)

k and repair_cap are measured from one calibration run at a single sigma, so the predicted
crossing is a real prediction and can be wrong. This sim measures the actual crossing and
grades it against that prediction per REFUTE.md.

The cross-domain reading is in docs/research/DOMAIN_PHYSICS.md: this is Ashby's requisite
variety, MCPM's drag ratio L/A, and R-tipping, which all compare a disturbance rate to a
maximum correction rate rather than examining landscape curvature.

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
    return sum(xs) / len(xs) if xs else float("nan")


def thermo_for(cfg: dict, mu_mode: str) -> dict:
    """Frozen mode pins mu by setting mu_max = mu_repair, so trust_r cannot shrink."""
    t = dict(cfg["thermo"])
    if mu_mode == "frozen":
        t["mu_max"] = t["mu_repair"]
    return t


def measure_rates(seed: int, sigma: float, cfg: dict, mu_mode: str = "adaptive") -> dict:
    """Per-step KL added by drift and removed by repair, measured separately.

    The two are measured on the same trajectory by evaluating KL three times per step:
    before drift, after drift, after repair. That attributes each contribution without
    needing two counterfactual runs.
    """
    sc = cfg["scenario"]
    env = Environment({"drift_strength": sc["initial_drift_strength"], "seed": seed})
    model_fn = env.get_model_fn()
    ref = env.theta_ref
    system = CoupledDynamicalSystem(model_fn, ref, env.task_inputs,
                                    thermo_for(cfg, mu_mode))
    theta = env.theta_drifted.clone()
    gen = torch.Generator().manual_seed(20_000 + seed)

    def kl(t):
        with torch.no_grad():
            return F.kl_div(F.log_softmax(model_fn(env.safety_inputs, t), dim=-1),
                            F.softmax(model_fn(env.safety_inputs, ref), dim=-1),
                            reduction="batchmean").item()

    d_drift, d_repair, repair_steps = [], [], []
    for _ in range(sc["steps"]):
        k0 = kl(theta)
        if sigma > 0:
            theta = theta + sigma * torch.randn(theta.shape, generator=gen)
        k1 = kl(theta)
        pre = theta.clone()
        theta, _ = system.step(theta, env.safety_inputs, env.task_inputs, env.task_labels)
        k2 = kl(theta)
        d_drift.append(k1 - k0)
        d_repair.append(k2 - k1)
        repair_steps.append((theta - pre).norm().item())

    return {
        "seed": seed, "sigma": sigma, "mu_mode": mu_mode,
        "dkl_drift": mean(d_drift),
        "dkl_repair": mean(d_repair),
        "dkl_net": mean(d_drift) + mean(d_repair),
        "repair_step_norm": mean(repair_steps),
        "final_kl": kl(theta),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))
    seeds = list(cfg["seeds"])
    sigmas = list(cfg["sweeps"]["drift_sigma"])
    mu_modes = list(cfg["sweeps"]["mu_mode"])
    if args.quick:
        seeds, sigmas = seeds[:2], sigmas[::3]
        cfg["scenario"] = {**cfg["scenario"], "steps": 10}

    print(f"[rate] {len(sigmas)} sigmas x {len(seeds)} seeds, "
          f"{cfg['scenario']['steps']} steps each")

    # --- null arm: no forcing ------------------------------------------------
    null_rows = [measure_rates(s, 0.0, cfg) for s in seeds]
    null_net = mean([r["dkl_net"] for r in null_rows])
    null_ok = null_net <= 0.0
    print(f"  null (sigma=0): net dKL/step = {null_net:+.6f}  ok={null_ok}")

    # --- calibration: one sigma fixes k and repair_cap, no free parameters ---
    cal_sigma = cfg["scenario"]["calibration_sigma"]
    cal_rows = [measure_rates(s, cal_sigma, cfg) for s in seeds]
    k = mean([r["dkl_drift"] for r in cal_rows]) / (cal_sigma ** 2)
    repair_cap = abs(mean([r["dkl_repair"] for r in cal_rows]))
    predicted = (repair_cap / k) ** 0.5 if k > 0 else float("nan")
    print(f"  calibration @ sigma={cal_sigma}: k={k:.3f}  repair_cap={repair_cap:.6f}")
    print(f"  PREDICTED sigma_crit = sqrt(repair_cap/k) = {predicted:.5f}  (0 free parameters)")

    # --- sweep, per mu_mode --------------------------------------------------
    def crossing(by_sigma: dict) -> float | None:
        ordered = sorted(by_sigma.items())
        for (s_lo, n_lo), (s_hi, n_hi) in zip(ordered, ordered[1:]):
            if n_lo <= 0.0 < n_hi:
                # interpolate in sigma^2, the variable the rate is quadratic in
                f = (0.0 - n_lo) / (n_hi - n_lo)
                return (s_lo ** 2 + f * (s_hi ** 2 - s_lo ** 2)) ** 0.5
        return None

    rows = []
    per_mu = {}
    for mm in mu_modes:
        by_sigma = {}
        for sg in sigmas:
            rs = [measure_rates(s, sg, cfg, mm) for s in seeds]
            rows.extend(rs)
            by_sigma[sg] = mean([r["dkl_net"] for r in rs])
            print(f"  [{mm:<8}] sigma={sg:<8} net dKL/step = {by_sigma[sg]:+.6f}  "
                  f"(drift {mean([r['dkl_drift'] for r in rs]):+.6f}, "
                  f"repair {mean([r['dkl_repair'] for r in rs]):+.6f})")
        nets = [by_sigma[s] for s in sorted(by_sigma)]
        per_mu[mm] = {
            "net_by_sigma": {str(k): v for k, v in by_sigma.items()},
            "sigma_crit": crossing(by_sigma),
            "monotone": all(b >= a for a, b in zip(nets, nets[1:])),
            "mean_repair_cap": abs(mean([r["dkl_repair"] for r in rows
                                         if r["mu_mode"] == mm])),
        }
        print(f"  [{mm:<8}] measured sigma_crit = {per_mu[mm]['sigma_crit']}")

    primary = per_mu[mu_modes[0]]
    by_sigma = {float(k): v for k, v in primary["net_by_sigma"].items()}
    measured = primary["sigma_crit"]
    monotone = primary["monotone"]

    tol = cfg["prediction"]["tolerance_factor"]
    if not null_ok:
        verdict = "VOID"
        reason = f"null arm shows net dKL/step = {null_net:+.6f} > 0; the repair loop does not hold an unforced basin"
    elif measured is None:
        verdict = "INCONCLUSIVE"
        reason = "net dKL/step does not cross zero inside the swept range"
    elif not monotone:
        verdict = "REFUTED"
        reason = "net dKL/step is not monotone in sigma; the sigma^2 forcing model is wrong"
    elif (1.0 / tol) <= measured / predicted <= tol:
        verdict = "SUPPORTED"
        reason = (f"measured sigma_crit = {measured:.5f} against a parameter-free "
                  f"prediction of {predicted:.5f} (ratio {measured / predicted:.2f}x)")
    else:
        verdict = "REFUTED"
        reason = (f"measured sigma_crit = {measured:.5f} vs predicted {predicted:.5f} "
                  f"(ratio {measured / predicted:.2f}x, outside {tol}x)")

    metrics = {
        "name": cfg["name"], "config_hash": config_hash, "quick_mode": bool(args.quick),
        "seeds": seeds, "sigmas": sigmas,
        "null": {"rows": null_rows, "net_dkl_per_step": null_net, "ok": null_ok},
        "calibration": {"sigma": cal_sigma, "k": k, "repair_cap": repair_cap,
                        "predicted_sigma_crit": predicted, "rows": cal_rows},
        "sweep_rows": rows,
        "net_by_sigma": {str(s): v for s, v in by_sigma.items()},
        "per_mu_mode": per_mu,
        "measured_sigma_crit": measured,
        "monotone_in_sigma": monotone,
        "grading": {"verdict": verdict, "reason": reason},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Rate-induced escape — run summary", "",
        f"Run at: {stamp}", f"Verdict: **{verdict}** — {reason}", "",
        "## Rate balance by drift rate", "",
        "| sigma | drift dKL/step | repair dKL/step | net dKL/step |", "|---|---|---|---|",
    ]
    for sg in sigmas:
        rs = [r for r in rows if r["sigma"] == sg]
        lines.append(f"| {sg} | {mean([r['dkl_drift'] for r in rs]):+.6f} "
                     f"| {mean([r['dkl_repair'] for r in rs]):+.6f} "
                     f"| {mean([r['dkl_net'] for r in rs]):+.6f} |")
    lines += [
        "", "## Parameter-free prediction", "",
        f"- calibration at sigma = {cal_sigma}: k = {k:.3f}, repair_cap = {repair_cap:.6f}",
        f"- predicted sigma_crit = sqrt(repair_cap / k) = **{predicted:.5f}**",
        f"- measured sigma_crit (zero crossing) = **{measured if measured is None else f'{measured:.5f}'}**",
        f"- null arm net dKL/step = {null_net:+.6f} (must be <= 0)",
        "", "Generated by run.py; do not edit.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE", "sim": cfg["name"],
        "claim": ("basin escape is rate-induced: a trust-region-capped repair against "
                  "sigma^2 forcing gives sigma_crit = sqrt(repair_cap/k)"),
        "refute_if": cfg["refute_if"], "verdict": verdict, "reason": reason,
        "predicted_sigma_crit": predicted, "measured_sigma_crit": measured,
        "metrics_hash": metrics_hash, "config_hash": config_hash,
        "seeds": len(seeds), "null_model": cfg["null_model"],
        "free_parameters": cfg["prediction"]["free_parameters"],
        "exploratory": bool(args.quick), "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    for mm in mu_modes:
        print(f"[rate] Q2 mu_mode={mm}: sigma_crit = {per_mu[mm]['sigma_crit']}, "
              f"repair cap = {per_mu[mm]['mean_repair_cap']:.6f}")
    print(f"\n[rate] measured sigma_crit = {measured}  predicted = {predicted:.5f}")
    print(f"[rate] VERDICT: {verdict} — {reason}")
    print(f"[rate] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
