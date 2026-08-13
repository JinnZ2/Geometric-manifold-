#!/usr/bin/env python3
"""Does the Rosetta shape cast a faithful shadow, or can faults hide in the projection?

notes 14 claims drill-down localizes a fault from the deformed shape, and measures vertex
localization at 6x, face aggregation at 2.8x and low-mode dominance at 85%. Those are all
measurements of faults that DO cast shadows. This asks the complementary question: what
does the same instrument miss?

"Shadow shape" has no prior definition in this repo (grep -ri shadow returns nothing), so
REFUTE.md proposes one: the observable is a reduced statistic of the deformed configuration,
and a shadow fault is a residual pattern that deforms the shape while leaving that statistic
unchanged.

The octahedron is isostatic in 3D -- 3*6 - 6 = 12 = its edge count -- so edge lengths
determine the shape exactly and the full displacement field should be full rank. But the
drill-down observable is per-vertex displacement *magnitudes*, which discards direction:
6 numbers cannot resolve 12 dimensions. If the rank drops as Maxwell counting predicts,
half of fault space is invisible to the instrument notes 14 actually uses.

Tier 1 (numpy). Usage: python3 run.py [--config config.json] [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Octahedron: 6 vertices at +/-1 on each axis; edges are all pairs except antipodal.
V0 = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
               [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
EDGES = [(i, j) for i in range(6) for j in range(i + 1, 6)
         if not np.allclose(V0[i], -V0[j])]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def relax(residuals, cfg):
    """Relax the spring network whose rest lengths are perturbed by the residuals.

    Edge e gets rest length d_ref * (1 + r_e); the configuration settles by gradient
    descent on the sum of squared length errors. This is the force-density picture from
    notes 14 section 2.
    """
    rc = cfg["relaxation"]
    d_ref = np.array([np.linalg.norm(V0[i] - V0[j]) for i, j in EDGES])
    target = d_ref * (1.0 + np.asarray(residuals, dtype=float))
    V = V0.copy()
    for _ in range(rc["steps"]):
        grad = np.zeros_like(V)
        for e, (i, j) in enumerate(EDGES):
            d = V[i] - V[j]
            L = np.linalg.norm(d)
            if L < 1e-12:
                continue
            g = 2.0 * (L - target[e]) * (d / L)
            grad[i] += g
            grad[j] -= g
        V = V - rc["lr"] * grad
    return V


def procrustes_align(V):
    """Remove translation and rotation, keeping scale (scale carries fault magnitude)."""
    A = V0 - V0.mean(axis=0)
    B = V - V.mean(axis=0)
    U, _, Wt = np.linalg.svd(B.T @ A)
    R = U @ Wt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Wt
    return B @ R


def observables(residuals, cfg):
    """The three nested shadows: full field, per-vertex magnitudes, Procrustes scalar."""
    disp = procrustes_align(relax(residuals, cfg)) - (V0 - V0.mean(axis=0))
    full = disp.reshape(-1)
    magnitudes = np.linalg.norm(disp, axis=1)
    scalar = np.array([np.linalg.norm(disp)])
    return {"full": full, "magnitudes": magnitudes, "scalar": scalar}


def jacobian(cfg, level, eps):
    """Numerical Jacobian of one observation level with respect to the 12 residuals."""
    base = observables(np.zeros(len(EDGES)), cfg)[level]
    J = np.zeros((base.size, len(EDGES)))
    for e in range(len(EDGES)):
        r = np.zeros(len(EDGES))
        r[e] = eps
        plus = observables(r, cfg)[level]
        r[e] = -eps
        minus = observables(r, cfg)[level]
        J[:, e] = (plus - minus) / (2 * eps)
    return J


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))
    seeds = list(cfg["seeds"])
    mags = list(cfg["sweeps"]["residual_magnitude"])
    if args.quick:
        cfg["relaxation"] = {**cfg["relaxation"], "steps": 800}
        seeds, mags = seeds[:3], mags[:2]

    pred = cfg["prediction"]
    eps = cfg["relaxation"]["jacobian_epsilon"]
    print(f"[shadow] octahedron: {len(V0)} vertices, {len(EDGES)} edges "
          f"(Maxwell 3*6-6 = 12, isostatic)")

    # --- H1/H2: rank of each observation level -------------------------------
    spectra = {}
    for level in ("full", "magnitudes", "scalar"):
        J = jacobian(cfg, level, eps)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int((sv > pred["rank_tolerance"] * max(sv[0], 1e-300)).sum())
        spectra[level] = {"singular_values": [float(v) for v in sv],
                          "rank": rank, "shape": list(J.shape)}
        print(f"  {level:<11} J{J.shape}  rank={rank}  "
              f"top sv={sv[0]:.4f}  smallest sv={sv[-1]:.2e}")

    h1 = spectra["full"]["rank"] == pred["rank_full"]
    h2 = spectra["magnitudes"]["rank"] <= pred["rank_magnitudes_max"]

    # --- H3: does the least-visible direction stay quiet at finite amplitude? -
    Jm = jacobian(cfg, "magnitudes", eps)
    _, _, Vt = np.linalg.svd(Jm)
    blind_dir = Vt[-1]                      # right-singular vector, smallest response
    blind_dir = blind_dir / np.linalg.norm(blind_dir)

    rows = []
    for mag in mags:
        blind_shadow = float(np.linalg.norm(observables(mag * blind_dir, cfg)["magnitudes"]
                                            - observables(np.zeros(len(EDGES)),
                                                          cfg)["magnitudes"]))
        rand_shadows = []
        for s in seeds:
            rng = np.random.default_rng(s)
            v = rng.normal(size=len(EDGES))
            v = v / np.linalg.norm(v)
            rand_shadows.append(float(np.linalg.norm(
                observables(mag * v, cfg)["magnitudes"]
                - observables(np.zeros(len(EDGES)), cfg)["magnitudes"])))
        typical = float(np.median(rand_shadows))
        spread = float(max(rand_shadows) / max(min(rand_shadows), 1e-300))
        ratio = blind_shadow / typical if typical > 0 else float("inf")
        rows.append({"residual_magnitude": mag, "blind_shadow": blind_shadow,
                     "typical_shadow": typical, "null_spread": spread, "ratio": ratio})
        print(f"  |r|={mag:<5} blind shadow={blind_shadow:.3e}  "
              f"typical={typical:.3e}  ratio={ratio:.4f}  null spread={spread:.1f}x")

    # --- H4: collisions at finite amplitude, the test H3 should have been ----
    # 12 residual dimensions map to 6 magnitudes, so the map cannot be injective and
    # collisions are guaranteed by dimension count. They are found by linearizing about a
    # NONZERO base fault, where |disp| is differentiable -- unlike at r = 0, where the
    # norm has no derivative and the resulting "kernel" is an artifact.
    base_mag = mags[len(mags) // 2]
    rng0 = np.random.default_rng(12345)
    r_base = rng0.normal(size=len(EDGES))
    r_base = base_mag * r_base / np.linalg.norm(r_base)
    base_obs = observables(r_base, cfg)["magnitudes"]

    Jb = np.zeros((6, len(EDGES)))
    for e in range(len(EDGES)):
        rp, rm = r_base.copy(), r_base.copy()
        rp[e] += eps
        rm[e] -= eps
        Jb[:, e] = (observables(rp, cfg)["magnitudes"]
                    - observables(rm, cfg)["magnitudes"]) / (2 * eps)
    _, _, Vtb = np.linalg.svd(Jb)
    null_dirs = Vtb[6:]                       # 12 - 6 = 6 dimensional null space

    collisions = []
    for step in (0.25 * base_mag, 0.5 * base_mag, base_mag):
        d = null_dirs[0] / np.linalg.norm(null_dirs[0])
        r_alt = r_base + step * d
        alt_obs = observables(r_alt, cfg)["magnitudes"]
        shadow_change = float(np.linalg.norm(alt_obs - base_obs))
        residual_change = float(np.linalg.norm(r_alt - r_base))
        # what a random perturbation of the same size does, for scale
        rand_changes = []
        for sd in seeds:
            rr = np.random.default_rng(900 + sd).normal(size=len(EDGES))
            rr = step * rr / np.linalg.norm(rr)
            rand_changes.append(float(np.linalg.norm(
                observables(r_base + rr, cfg)["magnitudes"] - base_obs)))
        typ = float(np.median(rand_changes))
        collisions.append({
            "residual_change": residual_change,
            "shadow_change_along_null": shadow_change,
            "shadow_change_typical": typ,
            "concealment_ratio": shadow_change / typ if typ > 0 else float("inf"),
        })
        print(f"  collision: |dr|={residual_change:.3f} -> shadow moves "
              f"{shadow_change:.3e} vs typical {typ:.3e}  "
              f"({shadow_change / typ:.3f}x)")

    h4 = all(c["concealment_ratio"] <= 0.1 for c in collisions)

    null_usable = all(r["null_spread"] <= 10.0 for r in rows)
    h3 = all(r["ratio"] <= pred["blind_ratio_max"] for r in rows)
    typical_nonzero = all(r["typical_shadow"] > 0 for r in rows)

    if not typical_nonzero:
        verdict = "VOID"
        reason = "the instrument shows no response to random faults at all"
    elif not null_usable:
        verdict = "INCONCLUSIVE"
        reason = ("the random ensemble's own shadows vary by more than 10x, so a blind "
                  "direction 10x below typical is not distinguishable from an ordinary "
                  "quiet direction")
    elif h1 and h2 and h3:
        verdict = "SUPPORTED"
        blind_dims = len(EDGES) - spectra["magnitudes"]["rank"]
        reason = (f"full field is full rank ({spectra['full']['rank']}/12) but the "
                  f"drill-down observable has rank {spectra['magnitudes']['rank']}, "
                  f"leaving {blind_dims} blind fault dimensions; the least-visible "
                  f"direction casts a shadow {max(r['ratio'] for r in rows):.4f}x typical")
    else:
        verdict = "REFUTED"
        fails = [n for n, ok in (("H1 full rank", h1), ("H2 magnitude rank", h2),
                                 ("H3 blind at finite amplitude", h3)) if not ok]
        reason = "failed: " + ", ".join(fails)
        if h4:
            reason += ("; but H4 holds -- collisions exist at finite amplitude, so the "
                       "shadow phenomenon is real and H3 was simply the wrong test")

    metrics = {
        "name": cfg["name"], "config_hash": config_hash, "quick_mode": bool(args.quick),
        "edges": [list(e) for e in EDGES], "seeds": seeds,
        "spectra": spectra,
        "blind_direction": [float(x) for x in blind_dir],
        "amplitude_rows": rows,
        "hypotheses": {"H1_full_rank": h1, "H2_magnitude_rank": h2,
                       "H3_blind_at_amplitude": h3, "H4_collisions": h4,
                       "null_usable": null_usable},
        "collisions": collisions,
        "grading": {"verdict": verdict, "reason": reason},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Shape shadow — run summary", "",
        f"Run at: {stamp}", f"Verdict: **{verdict}** — {reason}", "",
        "## Rank by observation level", "",
        "| level | Jacobian | rank | largest sv | smallest sv |", "|---|---|---|---|---|",
    ]
    for level in ("full", "magnitudes", "scalar"):
        sp = spectra[level]
        lines.append(f"| {level} | {tuple(sp['shape'])} | **{sp['rank']}** "
                     f"| {sp['singular_values'][0]:.4f} "
                     f"| {sp['singular_values'][-1]:.2e} |")
    lines += ["", "## Blind direction vs random faults of equal norm", "",
              "| \\|r\\| | blind shadow | typical shadow | ratio | null spread |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['residual_magnitude']} | {r['blind_shadow']:.3e} "
                     f"| {r['typical_shadow']:.3e} | **{r['ratio']:.4f}** "
                     f"| {r['null_spread']:.1f}x |")
    lines += ["", "## H4: collisions at finite amplitude", "",
              "| \\|dr\\| | shadow moves along null | typical | concealment |",
              "|---|---|---|---|"]
    for c in collisions:
        lines.append(f"| {c['residual_change']:.3f} "
                     f"| {c['shadow_change_along_null']:.3e} "
                     f"| {c['shadow_change_typical']:.3e} "
                     f"| **{c['concealment_ratio']:.3f}x** |")
    lines += ["", "Generated by run.py; do not edit."]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE", "sim": cfg["name"],
        "claim": ("the drill-down observable (per-vertex displacement magnitudes) is a "
                  "projection with a kernel, so some faults deform the shape without "
                  "changing the observed statistic"),
        "refute_if": cfg["refute_if"], "verdict": verdict, "reason": reason,
        "rank_full": spectra["full"]["rank"],
        "rank_magnitudes": spectra["magnitudes"]["rank"],
        "metrics_hash": metrics_hash, "config_hash": config_hash,
        "seeds": len(seeds), "null_model": cfg["null_model"],
        "exploratory": bool(args.quick), "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print(f"\n[shadow] VERDICT: {verdict} — {reason}")
    print(f"[shadow] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
