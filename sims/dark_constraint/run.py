#!/usr/bin/env python3
"""Dark matter as a shadow shape: when can an unmodelled constraint be detected?

`sims/shape_shadow/` asked whether a fault inside the model's own space could hide in a
projection. This asks the sharper question the dark-matter analogy poses: the component is
**outside the model entirely**, and is inferred only from the residual between what the
visible model predicts and what is observed -- which is what weak lensing does, reading
unseen mass off the distorted shapes of background galaxies.

The octahedron is isostatic: 12 shape degrees of freedom, 12 visible edges. Under ONE load
case the observer has 12 observations and 12 free parameters, so the visible-only model can
fit any deformation exactly -- including one partly caused by a constraint it does not
model. The dark component is absorbed into biased visible residuals and leaves no trace.
Detection needs more independent observations than parameters, so K >= 2 load cases make
the system overdetermined and the unmodelable part must surface.

Same structure as the astrophysical case: one probe admits degenerate explanations
(modified dynamics vs unseen mass, the mass-sheet and disk-halo degeneracies), and the
case is closed by combining independent probes.

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

V0 = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
               [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
EDGES = [(i, j) for i in range(6) for j in range(i + 1, 6)
         if not np.allclose(V0[i], -V0[j])]
DARK_EDGE = (0, 1)          # an antipodal pair: the only kind of non-edge available


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def relax(residuals, cfg, load=None, dark=0.0):
    """Settle the framework under edge residuals, an external load, and an optional
    hidden brace on the antipodal pair the observer's model does not include."""
    rc = cfg["relaxation"]
    d_ref = np.array([np.linalg.norm(V0[i] - V0[j]) for i, j in EDGES])
    target = d_ref * (1.0 + np.asarray(residuals, dtype=float))
    dark_ref = np.linalg.norm(V0[DARK_EDGE[0]] - V0[DARK_EDGE[1]])
    dark_target = dark_ref * (1.0 + dark)
    V = V0.copy()
    for _ in range(rc["steps"]):
        grad = np.zeros_like(V)
        for e, (i, j) in enumerate(EDGES):
            d = V[i] - V[j]
            L = np.linalg.norm(d)
            if L > 1e-12:
                g = 2.0 * (L - target[e]) * (d / L)
                grad[i] += g
                grad[j] -= g
        if dark != 0.0:
            i, j = DARK_EDGE
            d = V[i] - V[j]
            L = np.linalg.norm(d)
            if L > 1e-12:
                g = 2.0 * (L - dark_target) * (d / L)
                grad[i] += g
                grad[j] -= g
        if load is not None:
            grad = grad - load
        V = V - rc["lr"] * grad
    return V


def align(V):
    A = V0 - V0.mean(axis=0)
    B = V - V.mean(axis=0)
    U, _, Wt = np.linalg.svd(B.T @ A)
    R = U @ Wt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Wt
    return B @ R


def observe(residuals, cfg, load, dark=0.0):
    """Procrustes-aligned displacement field under one load case."""
    return (align(relax(residuals, cfg, load, dark)) - (V0 - V0.mean(axis=0))).reshape(-1)


def load_cases(k, cfg, seed=0):
    """K independent external load patterns, zero-net-force so they do not translate."""
    rng = np.random.default_rng(1000 + seed)
    out = []
    for _ in range(k):
        f = rng.normal(size=(6, 3))
        f = f - f.mean(axis=0)
        out.append(cfg["load"]["magnitude"] * f / np.linalg.norm(f))
    return out


def visible_jacobian(cfg, load, eps):
    """d(observation)/d(visible residuals) for one load case -- the observer's model."""
    base = observe(np.zeros(len(EDGES)), cfg, load)
    J = np.zeros((base.size, len(EDGES)))
    for e in range(len(EDGES)):
        r = np.zeros(len(EDGES))
        r[e] = eps
        plus = observe(r, cfg, load)
        r[e] = -eps
        minus = observe(r, cfg, load)
        J[:, e] = (plus - minus) / (2 * eps)
    return base, J


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_hash = sha256_str(canonical(cfg))
    seeds = list(cfg["seeds"])
    ks = list(cfg["sweeps"]["n_load_cases"])
    darks = list(cfg["sweeps"]["dark_strength"])
    if args.quick:
        cfg["relaxation"] = {**cfg["relaxation"], "steps": 800}
        seeds, ks, darks = seeds[:3], ks[:2], darks[:2]

    eps = cfg["relaxation"]["jacobian_epsilon"]
    pred = cfg["prediction"]
    print(f"[dark] octahedron: {len(EDGES)} visible edges + 1 hidden brace on {DARK_EDGE}")
    print("[dark] isostatic: 12 shape DOF vs 12 visible parameters")

    # The observer's model (Jacobian + unloaded baseline) depends only on the load case,
    # not on the dark strength, so it is built once per (k, seed) and reused.
    model_cache = {}
    for k in ks:
        for sd in seeds:
            loads = load_cases(k, cfg, sd)
            entry = []
            for ld in loads:
                base, J = visible_jacobian(cfg, ld, eps)
                entry.append((ld, base, J))
            model_cache[(k, sd)] = entry
        print(f"  built observer model for K={k} ({len(seeds)} seeds)")

    rows = []
    for k in ks:
        for dark in darks:
            resids, biases = [], []
            for sd in seeds:
                Js, ys = [], []
                for ld, base, J in model_cache[(k, sd)]:
                    truth = observe(np.zeros(len(EDGES)), cfg, ld, dark=dark)
                    Js.append(J)
                    ys.append(truth - base)
                A = np.vstack(Js)
                y = np.concatenate(ys)
                r_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
                unexplained = float(np.linalg.norm(A @ r_hat - y))
                resids.append(unexplained)
                # bias: the visible residuals the observer infers, when the truth is zero
                biases.append(float(np.linalg.norm(r_hat)))
            rows.append({"k": k, "dark": dark,
                         "unexplained": float(np.median(resids)),
                         "inferred_visible_norm": float(np.median(biases))})
            print(f"  K={k} dark={dark:<5} unexplained={rows[-1]['unexplained']:.3e}  "
                  f"inferred visible |r_hat|={rows[-1]['inferred_visible_norm']:.4f}")

    def get(k, dark):
        return next(r for r in rows if r["k"] == k and r["dark"] == dark)

    nz_darks = [d for d in darks if d > 0]
    h1 = all(get(1, d)["unexplained"] <= pred["k1_absorbed_tol"] for d in darks) \
        if 1 in ks else None
    multi = [k for k in ks if k >= 2]
    h2 = all(
        all(get(k, a)["unexplained"] <= get(k, b)["unexplained"]
            for a, b in zip(nz_darks, nz_darks[1:]))
        for k in multi
    ) if multi and len(nz_darks) > 1 else None
    h3 = all(
        get(k, max(nz_darks))["unexplained"]
        >= pred["separation_factor"] * max(get(k, 0.0)["unexplained"], 1e-300)
        for k in multi
    ) if multi and nz_darks and 0.0 in darks else None

    checks = [c for c in (h1, h2, h3) if c is not None]
    if all(checks) and checks:
        verdict = "SUPPORTED"
        reason = ("the dark constraint is absorbed without trace at K=1 and becomes "
                  "detectable at K>=2, growing with its strength and separable from null")
    else:
        verdict = "REFUTED"
        fails = [n for n, ok in (("H1 absorbed at K=1", h1), ("H2 grows with dark", h2),
                                 ("H3 separable from null", h3)) if ok is False]
        reason = "failed: " + ", ".join(fails)

    metrics = {
        "name": cfg["name"], "config_hash": config_hash, "quick_mode": bool(args.quick),
        "seeds": seeds, "k_values": ks, "dark_strengths": darks,
        "dark_edge": list(DARK_EDGE), "rows": rows,
        "hypotheses": {"H1_absorbed_at_k1": h1, "H2_grows_with_dark": h2,
                       "H3_separable_from_null": h3},
        "grading": {"verdict": verdict, "reason": reason},
    }
    metrics_hash = sha256_str(canonical(metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    outdir = HERE / "results" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = ["# Dark constraint — run summary", "", f"Run at: {stamp}",
             f"Verdict: **{verdict}** — {reason}", "",
             "## Unexplained residual (the detection signal)", "",
             "| K load cases | " + " | ".join(f"dark={d}" for d in darks) + " |",
             "|---" * (len(darks) + 1) + "|"]
    for k in ks:
        lines.append(f"| {k} | " + " | ".join(
            f"{get(k, d)['unexplained']:.2e}" for d in darks) + " |")
    lines += ["", "## Inferred visible residual norm (the bias)", "",
              "| K load cases | " + " | ".join(f"dark={d}" for d in darks) + " |",
              "|---" * (len(darks) + 1) + "|"]
    for k in ks:
        lines.append(f"| {k} | " + " | ".join(
            f"{get(k, d)['inferred_visible_norm']:.4f}" for d in darks) + " |")
    lines += ["", "Generated by run.py; do not edit."]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ledger = {
        "type": "MEASURE", "sim": cfg["name"],
        "claim": ("an unmodelled constraint is exactly degenerate with visible residuals "
                  "under a single load case, and detectable only with more independent "
                  "observations than model parameters"),
        "refute_if": cfg["refute_if"], "verdict": verdict, "reason": reason,
        "metrics_hash": metrics_hash, "config_hash": config_hash,
        "seeds": len(seeds), "null_model": cfg["null_model"],
        "exploratory": bool(args.quick), "recorded_at": stamp,
    }
    (outdir / "ledger_entry.jsonl").write_text(canonical(ledger) + "\n", encoding="utf-8")

    print(f"\n[dark] VERDICT: {verdict} — {reason}")
    print(f"[dark] results -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
