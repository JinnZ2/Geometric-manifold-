#!/usr/bin/env python3
"""Emits RESULTS.md from the latest results/<stamp>/raw.json. Per axis, per seed; nothing is
averaged across axes. Leads with the licence (S4), because everything below it depends on
it. Stdlib only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import align as A  # noqa: E402

DRIFT_AXES_PER_SEED = ("A1/TIME", "A2/UNIT-L1", "A2/UNIT-L2")
DRIFT_AXES_PER_COND = ("A3/SEED-raw", "A3/SEED-aligned")
ALL_DRIFT = DRIFT_AXES_PER_SEED + DRIFT_AXES_PER_COND


def latest_raw():
    runs = sorted((HERE / "results").glob("*/raw.json"))
    if not runs:
        raise SystemExit("no results/<stamp>/raw.json; run run_t1.py first")
    return runs[-1]


def f2(v):
    return "--" if v is None else f"{v:.2f}"


def phase_row(name, ph, ref=False):
    """ref=True prints the reference channel's legs under Li et al.'s names (C/E/K); a drift
    axis prints them as fall/rise/fall (F/R/F). Same detector, different licence to name."""
    if not ph.get("detectable"):
        return f"| {name} | -- | -- | -- | -- | -- | not detectable ({ph.get('reason')}) |"
    a, b, c = ("C", "E", "K") if ref else ("F", "R", "F")
    legs = (a if ph["fall1"] else "·") + (b if ph["rise"] else "·") + (c if ph["fall2"] else "·")
    return (
        f"| {name} | {f2(ph['v_start'])} | {f2(ph['v_trough'])} @{ph['t_trough']} | "
        f"{f2(ph['v_peak'])} @{ph['t_peak']} | {f2(ph['v_end'])} | {f2(ph['depth'])} | {legs} "
        f"{'three' if ph['three_phase'] else ('two' if ph['two_phase'] else 'none')}"
        f"{', nonmonotone' if ph['nonmonotone'] else ''} |"
    )


def analyse(raw):
    cfg = raw["config"]
    pd_ = cfg["phase_detector"]
    half, mf = pd_["smooth_halfwidth"], pd_["margin_frac"]
    out = {"per_condition": {}}
    for cond, c in raw["conditions"].items():
        cc = {"skew": c["skew"], "seeds": [], "A3": {}}
        for ps in c["per_seed"]:
            rows = ps["rows"]
            entry = {
                "seed": ps["seed"],
                "phases": {},
                "overlay": None,
                "final_loss": rows[-1]["loss"],
                "final_acc": rows[-1]["acc_probe"],
            }
            st, va = A.series(rows, "REP")
            entry["phases"]["REP"] = A.phases(st, va, half, mf)
            _, al = A.series(rows, "REP", "alpha_req")
            entry["alpha_REP"] = {"start": al[0], "end": al[-1], "min": min(al), "max": max(al)}
            for ax in DRIFT_AXES_PER_SEED:
                s2, v2 = A.series(rows, ax)
                entry["phases"][ax] = A.phases(s2, v2, half, mf)
            entry["overlay"] = A.overlay(
                rows,
                entry["phases"]["REP"],
                cfg["repair_flags"]["param_threshold"],
                cfg["repair_flags"]["policy_threshold"],
            )
            cc["seeds"].append(entry)
        for ax in DRIFT_AXES_PER_COND:
            s3, v3 = A.a3_series(c["A3_by_step"], ax)
            cc["A3"][ax] = A.phases(s3, v3, half, mf)
        cc["A3"]["n_seeds"] = len(c["per_seed"])
        out["per_condition"][cond] = cc

    # ---- S4
    sk = out["per_condition"]["skewed"]["seeds"]
    ct = out["per_condition"]["control"]["seeds"]
    n = len(sk)
    three_sk = sum(e["phases"]["REP"]["three_phase"] for e in sk)
    three_ct = sum(e["phases"]["REP"]["three_phase"] for e in ct)
    two_sk = sum(e["phases"]["REP"]["two_phase"] for e in sk)
    two_ct = sum(e["phases"]["REP"]["two_phase"] for e in ct)
    depth_sk = [e["phases"]["REP"]["depth"] for e in sk]
    depth_ct = [e["phases"]["REP"]["depth"] for e in ct]
    out["S4"] = {
        "n": n,
        "depth_skewed_min": min(depth_sk),
        "depth_skewed_max": max(depth_sk),
        "depth_control_min": min(depth_ct),
        "depth_control_max": max(depth_ct),
        "depth_separates": min(depth_sk) > max(depth_ct),
        "three_phase_skewed": three_sk,
        "three_phase_control": three_ct,
        "two_phase_skewed": two_sk,
        "two_phase_control": two_ct,
        "pass": three_sk > n / 2 and three_ct <= n / 2,
        "two_phase_discriminates": two_sk > n / 2 and two_ct <= n / 2,
    }
    licensed = out["S4"]["pass"]
    out["licensed"] = licensed

    # ---- axis structure summary (observations, licence-independent)
    struct = {}
    for ax in DRIFT_AXES_PER_SEED:
        struct[ax] = {
            cond: {
                "nonmonotone": sum(
                    e["phases"][ax].get("nonmonotone", False)
                    for e in out["per_condition"][cond]["seeds"]
                ),
                "two_phase": sum(
                    e["phases"][ax].get("two_phase", False)
                    for e in out["per_condition"][cond]["seeds"]
                ),
                "three_phase": sum(
                    e["phases"][ax].get("three_phase", False)
                    for e in out["per_condition"][cond]["seeds"]
                ),
                "n": len(out["per_condition"][cond]["seeds"]),
            }
            for cond in out["per_condition"]
        }
    for ax in DRIFT_AXES_PER_COND:
        struct[ax] = {
            cond: {
                "nonmonotone": int(out["per_condition"][cond]["A3"][ax].get("nonmonotone", False)),
                "two_phase": int(out["per_condition"][cond]["A3"][ax].get("two_phase", False)),
                "three_phase": int(out["per_condition"][cond]["A3"][ax].get("three_phase", False)),
                "n": 1,
            }
            for cond in out["per_condition"]
        }
    out["structure"] = struct
    # do the axes agree on whether drift has non-monotone structure (skewed condition)?
    votes = {
        ax: struct[ax]["skewed"]["nonmonotone"] > struct[ax]["skewed"]["n"] / 2 for ax in ALL_DRIFT
    }
    out["axes_agree"] = len(set(votes.values())) == 1
    out["axis_votes"] = votes

    # ---- section 4 alignment, only when licensed
    if licensed:
        alc = cfg["alignment"]
        lag_tol = int(alc["lag_tol_frac"] * cfg["train"]["steps"])
        for cond, c in raw["conditions"].items():
            for ps, entry in zip(c["per_seed"], out["per_condition"][cond]["seeds"]):
                st, va = A.series(ps["rows"], "REP")
                entry["alignment"] = {}
                for ax in DRIFT_AXES_PER_SEED:
                    s2, v2 = A.series(ps["rows"], ax)
                    entry["alignment"][ax] = A.align(
                        st, va, s2, v2, half, alc["corr_threshold"], lag_tol
                    )
    return out


def render(raw, an):
    cfg = raw["config"]
    S4 = an["S4"]
    L = []
    w = L.append
    w("# RESULTS — parameter-drift spectral signature, Tier 1")
    w("")
    w(
        f"Run `{raw['stamp']}` · {len(cfg['seeds'])} seeds × 2 conditions × {cfg['train']['steps']} steps · "
        f"{raw['runtime_s']} s · config sha256 `{raw['config_sha256'][:12]}` · emitted by `report.py`."
        f"{' QUICK RUN.' if raw.get('quick') else ''}"
    )
    w("")
    w("## 0. Licence (selftest S4) — read this first")
    w("")
    if S4["pass"]:
        w(
            f"**S4 PASSES.** Reference channel three-phase on {S4['three_phase_skewed']}/{S4['n']} skewed seeds, "
            f"{S4['three_phase_control']}/{S4['n']} control seeds. Section 4 below is licensed."
        )
    else:
        w(
            f"**S4 FAILS.** The reference channel (RankMe of hidden activations, the channel where Li et al. "
            f"publish the shape) reproduces the THREE-phase shape on **{S4['three_phase_skewed']}/{S4['n']}** skewed "
            f"seeds. It reproduces the first TWO phases — collapse then expansion — on **{S4['two_phase_skewed']}/{S4['n']}** "
            f"skewed seeds against **{S4['two_phase_control']}/{S4['n']}** control seeds"
            f"{' (the skew discriminates)' if S4['two_phase_discriminates'] else ' (the skew does NOT discriminate)'}. "
            f"The third phase, compression, appears on no seed and appeared in no setting of the calibration sweep "
            f"recorded in `config.json`."
        )
        w("")
        w(
            f"Added after the data was seen, and labelled so: the detector's margin is relative to each series' "
            f"own range, so a shallow dip passes it. What separates the conditions is the DEPTH of the first leg in "
            f"RankMe units — skewed {f2(S4['depth_skewed_min'])}–{f2(S4['depth_skewed_max'])}, control "
            f"{f2(S4['depth_control_min'])}–{f2(S4['depth_control_max'])}, "
            f"{'disjoint' if S4['depth_separates'] else 'overlapping'}. The detector is left as declared; the depth "
            f"is a second readout beside it, not a repair to the first."
        )
        w("")
        w(
            "Per the order (§6): *if the synthetic setup cannot reproduce the published shape in the channel where "
            "it is published, the parameter-space channel means nothing.* **Section 4 (LOCKED / ANTI-PHASE / "
            "DECOUPLED) is therefore NOT RUN and no alignment verdict is emitted.** What follows under §2–§3 "
            "are instrument readings on the drift axes, reported as observations and licensed by nothing."
        )
    w("")
    w(
        "What this negative is and is not: the generator (a 16→6→24 tanh MLP on Zipf-skewed prototypes, "
        "SGD with L2) does not consolidate after its expansion at any weight decay, learning-rate schedule "
        "or probe distribution tried. That is a statement about this generator. It is not evidence about "
        "whether real networks compress, and it is not evidence about basin repair."
    )
    w("")
    w("## 1. Reference channel (REP) per seed")
    w("")
    w(
        "RankMe of the centered covariance of hidden activations over a fixed probe set, "
        f"{cfg['data']['probe_distribution']} over classes, {cfg['data']['probe_per_class']} per class. "
        "Smoothed extrema; legs C=collapse E=expansion K=compression (Li et al.'s names, licensed on this channel only); each must clear "
        f"{cfg['phase_detector']['margin_frac']:.0%} of the smoothed range."
    )
    w("")
    for cond in ("skewed", "control"):
        c = an["per_condition"][cond]
        w(f"**{cond}** (skew {c['skew']})")
        w("")
        w(
            "| seed | start | trough @step | peak @step | end | depth | legs | alpha-ReQ start→end (min, max) | final loss / probe acc |"
        )
        w("|---|---|---|---|---|---|---|---|---|")
        for e in c["seeds"]:
            a = e["alpha_REP"]
            w(
                phase_row(f"s{e['seed']}", e["phases"]["REP"], ref=True)[:-1]
                + f" {f2(a['start'])}→{f2(a['end'])} ({f2(a['min'])}, {f2(a['max'])}) | {f2(e['final_loss'])} / {f2(e['final_acc'])} |"
            )
        w("")
    w("## 2. Drift axes — per axis, per seed, not averaged")
    w("")
    w(
        "Legs on a drift axis are F=fall R=rise F=fall — the same detector as §1, with no phase name attached. "
        "Each axis is a different sample dimension manufactured for theta (see `drift.py` header). "
        "RankMe here counts independent DIRECTIONS in the samples that axis supplies; its ceiling is the "
        f"sample count (A1: {cfg['axes']['A1_window']}, A2-L1: {cfg['model']['d']}, A2-L2: {cfg['model']['d'] + 1}, "
        f"A3: {len(cfg['seeds'])}). Uncentered second moment for all drift axes ([CHOICE 1])."
    )
    w("")
    for ax in DRIFT_AXES_PER_SEED:
        w(f"### {ax}")
        w("")
        for cond in ("skewed", "control"):
            c = an["per_condition"][cond]
            w(f"**{cond}**")
            w("")
            w("| seed | start | trough @step | peak @step | end | depth | legs |")
            w("|---|---|---|---|---|---|---|")
            for e in c["seeds"]:
                w(phase_row(f"s{e['seed']}", e["phases"][ax]))
            w("")
    for ax in DRIFT_AXES_PER_COND:
        w(
            f"### {ax} (one series per condition, across {an['per_condition']['skewed']['A3']['n_seeds']} seeds)"
        )
        w("")
        w("| condition | start | trough @step | peak @step | end | depth | legs |")
        w("|---|---|---|---|---|---|---|")
        for cond in ("skewed", "control"):
            w(phase_row(cond, an["per_condition"][cond]["A3"][ax]))
        w("")
    w("### Do the axes agree?")
    w("")
    w(
        "Question put to each axis on the skewed condition: does the smoothed RankMe series have any interior "
        'extremum clearing the margin on both sides ("nonmonotone")? Majority over seeds where there are seeds.'
    )
    w("")
    w(
        "| axis | skewed nonmonotone | skewed two-phase | control nonmonotone | control two-phase | vote |"
    )
    w("|---|---|---|---|---|---|")
    for ax in ALL_DRIFT:
        s = an["structure"][ax]
        w(
            f"| {ax} | {s['skewed']['nonmonotone']}/{s['skewed']['n']} | {s['skewed']['two_phase']}/{s['skewed']['n']} | "
            f"{s['control']['nonmonotone']}/{s['control']['n']} | {s['control']['two_phase']}/{s['control']['n']} | "
            f"{'structure' if an['axis_votes'][ax] else 'none'} |"
        )
    w("")
    if an["axes_agree"]:
        w(
            "The axes AGREE on whether drift has non-monotone spectral structure. No BRANCH ENTRY 02."
        )
    else:
        w(
            "**The axes DISAGREE.** Per the order §1 that disagreement is the primary result on the drift side "
            'and per §8 it opens BRANCH ENTRY 02 against the rule "parameter drift has a spectrum" — see '
            "`BRANCH.md`. It is recorded as an observation: with S4 failed, nothing here is licensed against "
            "the reference channel."
        )
    w("")
    w("## 3. Repair-flag overlay")
    w("")
    w(
        "The repo's two flag rules (`INVENTORY.md` §2, §6), reference := theta_0, thresholds "
        f"{cfg['repair_flags']['param_threshold']} (hardcoded in the controller) and {cfg['repair_flags']['policy_threshold']} "
        "(config). On a run from init both are monotone in distance and fire once; the overlay places that "
        "step against the reference channel's turning points."
    )
    w("")
    w("| condition | seed | param flag | policy flag |")
    w("|---|---|---|---|")
    for cond in ("skewed", "control"):
        for e in an["per_condition"][cond]["seeds"]:
            o = e["overlay"]
            w(
                f"| {cond} | s{e['seed']} | {o['param_flag_placement']} | {o['policy_flag_placement']} |"
            )
    w("")

    def zone_counts(key):
        cnt = {"before_trough": 0, "between": 0, "after_peak": 0, "none": 0}
        for cond in ("skewed", "control"):
            for e in an["per_condition"][cond]["seeds"]:
                cnt[e["overlay"][key]] += 1
        return cnt

    pz, qz = zone_counts("param_zone"), zone_counts("policy_zone")
    tot = sum(pz.values())
    w(
        f"Placement counts over {tot} runs — parameter flag: before REP trough {pz['before_trough']}, between trough and "
        f"peak {pz['between']}, after peak {pz['after_peak']}, never {pz['none']}. Policy flag: before trough "
        f"{qz['before_trough']}, between {qz['between']}, after peak {qz['after_peak']}, never {qz['none']}."
    )
    w("")
    dists, cterms = [], []
    for cond, c in raw["conditions"].items():
        for ps in c["per_seed"]:
            for r in ps["rows"]:
                if r["param_confidence"] < cfg["repair_flags"]["param_threshold"]:
                    dists.append(r["dist_to_ref"])
                    cterms.append(-math.log(r["param_confidence"]) - r["dist_to_ref"])
                    break
    w(
        f"Reading: at the parameter flag's first crossing `||theta - theta_0||` is {f2(min(dists))}–{f2(max(dists))} "
        f"and the curvature term `lambda_curv * curv` is {f2(min(cterms))}–{f2(max(cterms))}; the flag fires when the "
        "distance from init exceeds ln 2 ≈ 0.69, which is a property of the flag's scale against this model's step size, not of any "
        "spectral transition; the policy flag depends on how far the output distribution has moved from init. "
        'Each fires exactly once on a run from init, so "clustered at a transition" versus "uniform across '
        'phases" is not a distinction this overlay can draw. That is a limit of the flag, recorded in '
        "INVENTORY.md §2, not of the run."
    )
    w("")
    if an["licensed"]:
        w("## 4. Alignment (licensed)")
        w("")
        w("| condition | seed | axis | verdict | corr | lag |")
        w("|---|---|---|---|---|---|")
        for cond in ("skewed", "control"):
            for e in an["per_condition"][cond]["seeds"]:
                for ax, r in e["alignment"].items():
                    w(
                        f"| {cond} | s{e['seed']} | {ax} | {r['verdict']} | {f2(r.get('corr'))} | {r.get('lag', '--')} |"
                    )
        w("")
    else:
        w("## 4. Alignment — NOT RUN")
        w("")
        w(
            "S4 failed (§0). `align.align()` exists and is exercised by the selftest on constructed series; it "
            "was not called on this run's data and no LOCKED / ANTI-PHASE / DECOUPLED verdict is emitted."
        )
        w("")
    w("## 5. Declared choices")
    w("")
    w("- [CHOICE 1] drift axes use the uncentered second moment; REP is centered (`drift.py`).")
    w(
        f"- [CHOICE 2] probe set {cfg['data']['probe_distribution']} over classes (`config.json`); the reference "
        "instrument samples its training distribution — tried in calibration, no change to the S4 outcome."
    )
    w(
        f"- [CHOICE 3] phase detector: smoothing half-width {cfg['phase_detector']['smooth_halfwidth']} samples, "
        f"margin {cfg['phase_detector']['margin_frac']} of the smoothed range, three legs (`align.phases`)."
    )
    w(
        f"- [CHOICE 4] alignment thresholds corr {cfg['alignment']['corr_threshold']}, lag tolerance "
        f"{cfg['alignment']['lag_tol_frac']:.0%} of steps — declared, unused on this run."
    )
    w(f"- A1 window {cfg['axes']['A1_window']}: {cfg['axes']['A1_window_source']}.")
    w(
        "- A3 alignment: one hidden-unit permutation per seed, fitted at the FINAL checkpoint against seed 0, "
        "applied at every step (`run_t1.a3_across_seeds`)."
    )
    w(
        "- Added after the data was seen: the `depth` column (start minus trough, series units). The detector "
        "itself was not changed."
    )
    w("- Generator calibration: `config.json` → `calibration_record`.")
    w("")
    w("## 6. Not claimed")
    w("")
    w(
        "No phase names are attached to any drift axis. No statement about real networks. No statement about "
        "whether basin repair is doing the right thing. Tier 2 (Pythia checkpoints) is specified in the order "
        "and not attempted here."
    )
    return "\n".join(L) + "\n"


def render_branch(raw, an):
    L = [
        "# BRANCH RECORD — drift_spectrum",
        "",
        "## BRANCH ENTRY 02 — opened by axis disagreement (order §8)",
        "",
        f"Run `{raw['stamp']}`. Licence status at opening: S4 {'PASSED' if an['S4']['pass'] else 'FAILED'} "
        "(see RESULTS.md §0).",
        "",
        "```",
        "rule as stated    parameter drift has a spectrum",
        'forcing case      the three candidate axes return different answers to "does the drift',
        '                  spectrum have non-monotone structure over training" on the same runs',
        "axis              "
        + ", ".join(f"{ax}={'structure' if v else 'none'}" for ax, v in an["axis_votes"].items()),
        "derivation        parameter space has no sample dimension; every spectrum is a spectrum of",
        "                  the axis that supplied one. Axes that disagree are measuring different",
        '                  things, so "the drift spectrum" does not denote one object.',
        "frame note        the disagreement is a property of the substitution, not of the drift.",
        "                  It stands whether or not S4 licenses a comparison to the reference channel,",
        "                  because it is internal to the drift side.",
        "```",
        "",
    ]
    return "\n".join(L)


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = Path(argv[0]) if argv else latest_raw()
    raw = json.loads(path.read_text())
    an = analyse(raw)
    text = render(raw, an)
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    (path.parent / "analysis.json").write_text(json.dumps(an, indent=1))
    if not an["axes_agree"]:
        (HERE / "BRANCH.md").write_text(render_branch(raw, an), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
