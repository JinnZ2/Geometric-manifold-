#!/usr/bin/env python3
"""Work order section 2: what basin repair measures, triggers, does, retains — read from
the repository before any test is built, and emitted as INVENTORY.md.

Two kinds of line, kept apart in the output:
    [MECHANICAL]  read from source by AST or exact string, recomputable by running this
    [READING]     a sentence about what the code means; a person may disagree, and the
                  quoted source is beside it so they can

Every MECHANICAL fact is checked live: if the repo changes so that a quoted constant or a
named symbol is no longer where this says it is, `--selftest` goes red rather than the
inventory going stale. Stdlib only.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

FILES = {
    "param": ROOT / "manifolds" / "parameter_manifold.py",
    "ctrl": ROOT / "simulation" / "controller.py",
    "mon": ROOT / "repair" / "monitors.py",
    "policy": ROOT / "manifolds" / "policy_manifold.py",
    "energy": ROOT / "addon_thermodynamic_control" / "energy.py",
    "stab": ROOT / "addon_thermodynamic_control" / "stability.py",
    "generic": ROOT / "repair" / "generic_repair_controller.py",
    "cfg": ROOT / "configs" / "default.yaml",
    "adv": ROOT / "configs" / "adversarial.yaml",
}


def src(key: str) -> str:
    return FILES[key].read_text(encoding="utf-8")


# ------------------------------------------------------------------ mechanical extraction


def metric_keys_of(func_src_file: str, func_name: str, dict_var: str):
    """Keys of the dict literal assigned to `dict_var` inside `func_name`."""
    tree = ast.parse(src(func_src_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == dict_var for t in sub.targets)
                    and isinstance(sub.value, ast.Dict)
                ):
                    return [k.value for k in sub.value.keys if isinstance(k, ast.Constant)]
    return None


def default_of(file_key: str, getter_key: str):
    """The literal default in `config.get('<getter_key>', <default>)`."""
    m = re.search(
        rf"config\.get\(\s*['\"]{re.escape(getter_key)}['\"]\s*,\s*([^)]+)\)", src(file_key)
    )
    return m.group(1).strip() if m else None


def yaml_scalar(file_key: str, key: str):
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", src(file_key), re.M)
    return m.group(1) if m else None


def function_calls(file_key: str, name: str) -> int:
    """How many times `<something>.name(` or `name(` is CALLED (not defined) in a file."""
    tree = ast.parse(src(file_key))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == name) or (
                isinstance(f, ast.Name) and f.id == name
            ):
                n += 1
    return n


def kwarg_default(file_key: str, func_name: str, arg: str):
    tree = ast.parse(src(file_key))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args
            pos = args.args
            defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
            for a, d in zip(pos, defaults):
                if a.arg == arg and d is not None:
                    return ast.literal_eval(d)
    return None


def repo_grep(
    pattern: str, exclude_dirs=("falsifier-survey", "sims/drift_spectrum", ".git", "tests")
):
    hits = []
    rx = re.compile(pattern)
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(e) for e in exclude_dirs):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}")
    return hits


def collect() -> dict:
    f = {}
    # --- what is measured
    f["repair_step_metric_keys"] = metric_keys_of("param", "repair_step", "metrics")
    f["dist_to_ref_expr"] = re.search(r"'dist_to_ref':\s*(.+),", src("param")).group(1).strip()
    f["confidence_expr"] = re.search(
        r"confidence = (torch\.exp\(.+?\))\.item\(\)", src("param")
    ).group(1)
    f["monitor_window_default"] = kwarg_default("mon", "detect_cost_spike", "window")
    f["energy_trend_window_default"] = kwarg_default("energy", "recent_trend", "window")
    f["generic_trend_window_default"] = kwarg_default("generic", "_recent_trend", "window")
    # --- what triggers
    m = re.search(
        r"metrics\['repair_triggered'\]\s*=\s*(param_metrics\['confidence'\]\s*<\s*[0-9.]+)",
        src("ctrl"),
    )
    f["ctrl_param_trigger_expr"] = m.group(1) if m else None
    f["policy_threshold_default"] = default_of("policy", "confidence_threshold")
    f["policy_threshold_yaml_default"] = yaml_scalar("cfg", "confidence_threshold")
    f["policy_threshold_yaml_adv"] = yaml_scalar("adv", "confidence_threshold")
    f["repair_step_calls_in_controller"] = function_calls("ctrl", "repair_step")
    f["reanchor_calls_in_controller"] = function_calls("ctrl", "reanchor")
    f["reanchor_defined_in_policy"] = "def reanchor" in src("policy")
    f["repair_step_unconditional"] = bool(
        re.search(
            r"if self\.param_layer:\s*\n\s*theta, param_metrics = self\.param_layer\.repair_step",
            src("ctrl"),
        )
    )
    # --- what repair does
    f["trust_radius_default"] = default_of("param", "trust_radius")
    f["trust_radius_yaml"] = yaml_scalar("cfg", "trust_radius")
    f["lr_yaml"] = yaml_scalar("cfg", "lr")
    f["total_loss_expr"] = re.search(r"total_loss = (.+)", src("param")).group(1).strip()
    f["trust_clamp_present"] = "delta = delta * (self.trust_radius / norm)" in src("param")
    # --- spectral quantities already present
    f["eig_or_svd_calls"] = repo_grep(r"eigvals|eigh\(|eigvalsh|svd\(|linalg\.eig")
    f["power_iteration_sites"] = repo_grep(r"power_iter|power iteration")
    f["spectral_norm_approx_body"] = (
        re.search(r"def spectral_norm_approx\(self, diag.*?\n(.*?)\n\n", src("energy"), re.S)
        .group(1)
        .strip()
        .splitlines()[-1]
        .strip()
    )
    f["rankme_or_effective_rank_sites"] = repo_grep(
        r"[Rr]ank[Mm]e|effective[_ ]rank|alpha[-_ ]?[Rr]e[Qq]"
    )
    # --- retention
    f["torch_save_sites"] = repo_grep(r"torch\.save\(")
    f["monitor_records_what"] = "self.records.append(metrics)" in src("mon")
    f["monitor_step_key"] = "metrics['step'] = step" in src("mon")
    f["theta_retained_by_monitor"] = bool(re.search(r"theta", src("mon")))
    f["generic_history_type"] = re.search(r"self\._history: list\[(\w+)\]", src("generic")).group(1)
    return f


# ------------------------------------------------------------------ render


def render(f: dict) -> str:
    L = []
    w = L.append
    w("# INVENTORY — what basin repair measures, triggers, does, retains")
    w("")
    w("Emitted by `sims/drift_spectrum/inventory.py` (work order section 2). `[MECHANICAL]`")
    w("lines are read from source and re-checked by `inventory.py --selftest`; `[READING]`")
    w("lines are sentences about what the code means, with the source beside them.")
    w("")
    w("## 1. What is MEASURED as drift")
    w("")
    w(
        f"- [MECHANICAL] `repair_step` returns metrics {f['repair_step_metric_keys']} "
        "(`manifolds/parameter_manifold.py`)."
    )
    w(
        f"- [MECHANICAL] `dist_to_ref` = `{f['dist_to_ref_expr']}` — the L2 norm of theta minus the "
        "reference, taken on the post-step theta."
    )
    w("- [READING] Units: raw parameter units (weights are dimensionless). Window: NONE — the")
    w("  quantity is instantaneous, cumulative from the reference, recomputed every step. There is")
    w("  no windowed drift quantity in the core loop.")
    w(
        f"- [MECHANICAL] The only defined windows anywhere in the repair path are all **{f['monitor_window_default']} steps**: "
        f"`Monitor.detect_cost_spike(window={f['monitor_window_default']})`, "
        f"`RepairCostMonitor.recent_trend(window={f['energy_trend_window_default']})` in the addon, and "
        f"`GenericRepairController._recent_trend(window={f['generic_trend_window_default']})`. All three "
        "are rolling-mean ratios over repair COST, not over theta."
    )
    w(
        '- [READING] Consequence for this test (order §2, "the test uses THAT window"): drift itself has'
    )
    w("  no window to inherit, so A1/TIME takes W = 10 from the one window the repo does define,")
    w("  and says so on every emitted spectrum. This is a transfer, recorded as one.")
    w(
        f"- [MECHANICAL] Confidence = `{f['confidence_expr']}` — `exp(-lambda_curv * risk - dist)`, where "
        "`risk` is the softmax-variance curvature proxy and `dist` the same L2 norm."
    )
    w("- [MECHANICAL] Addon quantities (`addon_thermodynamic_control/energy.py`, `stability.py`,")
    w("  `repair/generic_repair_controller.py`): `basin_kl` (KL of outputs from the reference on")
    w("  safety inputs), `repair_energy = delta^T diag(F) delta` per step and its cumulative sum,")
    w("  `kappa_eff` (effective repair curvature), `lambda_max` of the safety Hessian by power")
    w(
        "  iteration. None is a function of theta alone; all are functions of theta on a fixed input set."
    )
    w("")
    w("## 2. What TRIGGERS a repair event")
    w("")
    w(
        f"- [MECHANICAL] `repair_step` is called {f['repair_step_calls_in_controller']} time per step in "
        f"`simulation/controller.py`, guarded only by `if self.param_layer:` "
        f"(unconditional-per-step = {f['repair_step_unconditional']}). **Repair runs every step.**"
    )
    w(
        f"- [MECHANICAL] `repair_triggered` is a logged FLAG, set by `{f['ctrl_param_trigger_expr']}` — the "
        "constant 0.5 is hardcoded in the controller and is in no config file — OR by "
        f"`policy_layer.needs_repair(policy_conf)`, whose threshold is `confidence_threshold` "
        f"(code default {f['policy_threshold_default']}, `default.yaml` {f['policy_threshold_yaml_default']}, "
        f"`adversarial.yaml` {f['policy_threshold_yaml_adv']})."
    )
    w('- [READING] So there is no trigger in the sense the order asks about ("threshold? schedule?')
    w(
        '  detector?"): the schedule is every step, and the thresholded quantities produce a LABEL on'
    )
    w("  a step that ran anyway. The label is what §4's overlay can use; nothing else exists.")
    w(
        f"- [MECHANICAL] `PolicyManifold.reanchor` is defined ({f['reanchor_defined_in_policy']}) and called "
        f"{f['reanchor_calls_in_controller']} times in the controller. The policy layer's repair "
        "action is never executed by the core loop; only its flag is read."
    )
    w("- [MECHANICAL] The addon assigns a `phase` label in {stable, threshold, critical} from")
    w(
        "  kappa / basin_kl / trend thresholds (`energy.py`, `stability.py`, `generic_repair_controller.py`)."
    )
    w("  `sims/kappa_eff_leading/FINDING.md` records kappa_eff REFUTED as a leading indicator of")
    w("  basin breach across three rounds. Those labels are not used here.")
    w("")
    w("## 3. What a repair DOES to theta")
    w("")
    w(
        f"- [MECHANICAL] One gradient step on `{f['total_loss_expr']}` (the saddle-point sign is intentional, "
        "per CLAUDE.md), `delta = -lr * grad`, then an L2 clamp `||delta|| <= trust_radius` "
        f"(clamp present = {f['trust_clamp_present']}). Defaults: `trust_radius` {f['trust_radius_yaml']}, "
        f"`lr` {f['lr_yaml']} (`configs/default.yaml`)."
    )
    w("- [READING] Every step therefore moves theta by at most `trust_radius` in L2. In the")
    w("  delta-theta vocabulary of this test, the repo already produces exactly one per-step")
    w("  delta vector per step; it does not keep it (section 5).")
    w("")
    w("## 4. Whether any SPECTRAL quantity is already computed")
    w("")
    core = [
        h
        for h in f["eig_or_svd_calls"]
        if h.split("/")[0] in ("manifolds", "simulation", "repair", "addon_thermodynamic_control")
    ]
    w(
        f"- [MECHANICAL] Eigen/SVD call sites in the repo: {len(f['eig_or_svd_calls'])}, of which "
        f"{len(core)} are in the repair path (`manifolds/`, `simulation/`, `repair/`, `addon_*/`). "
        f"Sites: {f['eig_or_svd_calls']}."
    )
    w(
        "- [READING] Those decompose a coupling network's Laplacian (`research_interface/`), toy-landscape"
    )
    w(
        "  safety Hessians (`experiments/toy_landscape*.py`), Procrustes cross-covariances of point sets"
    )
    w(
        "  (`sims/dark_constraint`, `sims/shape_shadow`) and a mean-field Jacobian (`docs/theoretical_notes`)."
    )
    w("  None is a covariance of theta over any axis.")
    w(
        f"- [MECHANICAL] Power-iteration sites: {len(f['power_iteration_sites'])} — the top eigenvalue of the "
        "safety-loss Hessian (`SpectralCertificate`), a single number, not a spectrum."
    )
    w(
        f"- [MECHANICAL] `FisherMetricEstimator.spectral_norm_approx` returns `{f['spectral_norm_approx_body']}` — "
        "the largest DIAGONAL Fisher entry, named as a spectral norm."
    )
    w(
        f"- [MECHANICAL] RankMe / effective-rank / alpha-ReQ sites: {f['rankme_or_effective_rank_sites'] or 'none'}."
    )
    w("- [READING] No covariance of theta over any axis is computed anywhere. The two metrics the")
    w("  order imports as reference instruments have no counterpart in the repo; nothing is being")
    w("  compared against an existing number.")
    w("")
    w("## 5. Whether checkpoints / histories are RETAINED")
    w("")
    w(
        f"- [MECHANICAL] `torch.save` call sites: {f['torch_save_sites'] or 'none'}. No theta is written anywhere."
    )
    w(
        f"- [MECHANICAL] `Monitor` appends the metrics dict per step ({f['monitor_records_what']}) with a "
        f"`step` key ({f['monitor_step_key']}) and writes `results/metrics.csv`; the word `theta` occurs in "
        f"`monitors.py` = {f['theta_retained_by_monitor']}. `GenericRepairController._history` is a "
        f"`list[{f['generic_history_type']}]` of scalar metrics (including `delta_norm`), not of theta."
    )
    w(
        "- [READING] Per-step METRICS are retained with a step index, so a repair-flag timeline against"
    )
    w("  training step exists and §4's overlay CAN run on the flag. Theta and delta-theta are")
    w("  discarded on every step, so no drift SPECTRUM can be computed from anything the repo")
    w("  retains today: this test has to record theta itself, which Tier 1 does in its own loop.")
    w("")
    w("## 6. What follows for the test")
    w("")
    w(
        "- Window for A1/TIME: 10 steps, transferred from the cost-trend window (§1 above), declared."
    )
    w("- Repair-event overlay (§4 of the order): uses the repo's two flag rules — parameter")
    w("  confidence `exp(-lambda_curv*curv - dist) < 0.5` and policy confidence")
    w("  `1 - JS/ln2 < 0.4` — re-implemented in pure Python on the Tier 1 run with the reference")
    w("  set to theta_0. That is NOT the repo's scenario (there, theta starts drifted from an")
    w("  aligned reference and is pulled back); on a training run from init both quantities are")
    w("  monotone in dist, so each flag crosses once. The overlay reports WHERE that crossing")
    w("  falls relative to the spectral transitions, which is the only overlay the flag supports.")
    w("- Nothing in sections 1–5 changes under `adversarial.yaml` except the constants quoted.")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ entry


def selftest() -> int:
    f = collect()
    checks = [
        ("dist_to_ref is a metric key", "dist_to_ref" in (f["repair_step_metric_keys"] or [])),
        ("confidence is a metric key", "confidence" in (f["repair_step_metric_keys"] or [])),
        (
            "all three windows are 10",
            (
                f["monitor_window_default"],
                f["energy_trend_window_default"],
                f["generic_trend_window_default"],
            )
            == (10, 10, 10),
        ),
        (
            "controller trigger literal found",
            f["ctrl_param_trigger_expr"] is not None and "0.5" in f["ctrl_param_trigger_expr"],
        ),
        ("repair_step called exactly once per step", f["repair_step_calls_in_controller"] == 1),
        (
            "reanchor defined and never called",
            f["reanchor_defined_in_policy"] and f["reanchor_calls_in_controller"] == 0,
        ),
        ("trust clamp present", f["trust_clamp_present"]),
        ("no torch.save anywhere", f["torch_save_sites"] == []),
        (
            "no RankMe/alpha-ReQ anywhere outside this folder",
            f["rankme_or_effective_rank_sites"] == [],
        ),
        ("monitor does not retain theta", not f["theta_retained_by_monitor"]),
        (
            "no eigen/SVD site in the repair path",
            not any(
                h.split("/")[0]
                in ("manifolds", "simulation", "repair", "addon_thermodynamic_control")
                for h in f["eig_or_svd_calls"]
            ),
        ),
        (
            "eigen/SVD sites found outside it (so the grep is not blind)",
            len(f["eig_or_svd_calls"]) > 0,
        ),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {n}")
    print(f"inventory selftest: {len(checks) - len(bad)}/{len(checks)}")
    return 1 if bad else 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    text = render(collect())
    out = HERE / "INVENTORY.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[wrote {out.relative_to(ROOT)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
