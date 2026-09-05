#!/usr/bin/env python3
"""Work order section 6. S1-S3 and S5 are known-answer checks on the instruments; S4 is the
LICENCE and runs the generator. Exit code is nonzero if any of S1-S5 fails, and S4 failing is
the recorded state of this folder (RESULTS.md section 0): this script going red on S4 is the
finding, not a broken test. `tests/test_drift_spectrum.py` pins that state so a change to it
turns the repo suite red rather than passing silently.

Usage: python3 selftest.py [--skip-s4]
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import align as A  # noqa: E402
import run_t1 as R  # noqa: E402
import spectrum as S  # noqa: E402
from spectrum import Spectrum  # noqa: E402


def _report(results):
    bad = [n for n, ok, _ in results if not ok]
    for n, ok, msg in results:
        print(f"  [{'ok' if ok else 'FAIL'}] {n}: {msg}")
    print(f"drift_spectrum selftest: {len(results) - len(bad)}/{len(results)} checks pass")
    return bad


# ------------------------------------------------------------------ S1


def s1_jacobi():
    """Seeded symmetric matrix with KNOWN eigenvalues: A = Q diag(lam) Q^T, Q from Gram-Schmidt."""
    rng = random.Random(20260905)
    n = 9
    lam = sorted((rng.uniform(-5, 5) for _ in range(n)), reverse=True)
    m = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    q = []
    for v in m:
        for u in q:
            d = sum(a * b for a, b in zip(v, u))
            v = [a - d * b for a, b in zip(v, u)]
        nv = math.sqrt(sum(a * a for a in v))
        q.append([a / nv for a in v])
    a = [[sum(q[k][i] * lam[k] * q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    got = S.jacobi_eigenvalues(a)
    err = max(abs(x - y) for x, y in zip(got, lam))
    # and the asymmetry guard fires
    try:
        S.jacobi_eigenvalues([[1.0, 2.0], [0.0, 1.0]])
        guard = False
    except ValueError:
        guard = True
    return [
        ("S1 Jacobi recovers seeded eigenvalues to 1e-9", err < 1e-9, f"max err {err:.2e}"),
        ("S1 Jacobi refuses an asymmetric matrix", guard, "ValueError raised"),
    ]


# ------------------------------------------------------------------ S2


def s2_rankme():
    d = 11
    iso = S.rankme([2.5] * d)
    r1 = S.rankme([3.0] + [0.0] * (d - 1))
    rng = random.Random(7)
    rows = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(300)]
    sp = S.spectrum_of_rows("REP", rows, centered=True)
    return [
        ("S2 RankMe(isotropic, d=11) == 11", abs(iso - d) < 1e-9, f"{iso:.12f}"),
        ("S2 RankMe(rank-1) == 1", abs(r1 - 1.0) < 1e-9, f"{r1:.12f}"),
        (
            "S2 RankMe of 300 gaussian samples in R^6 is near 6",
            5.7 < sp.rankme <= 6.0,
            f"{sp.rankme:.3f}",
        ),
    ]


# ------------------------------------------------------------------ S3


def s3_alpha():
    out = []
    for a in (0.7, 1.3, 2.4):
        ev = [(i + 1) ** (-a) for i in range(40)]
        got = S.alpha_req(ev)
        out.append(
            (
                f"S3 alpha-ReQ recovers exponent {a} within 5%",
                abs(got - a) / a < 0.05,
                f"got {got:.4f}",
            )
        )
    # the metric is not constant: two different exponents give two different readings
    g1, g2 = (
        S.alpha_req([(i + 1) ** -0.7 for i in range(40)]),
        S.alpha_req([(i + 1) ** -2.4 for i in range(40)]),
    )
    out.append(
        (
            "S3 alpha-ReQ separates exponents 0.7 and 2.4",
            abs(g1 - g2) > 1.0,
            f"{g1:.2f} vs {g2:.2f}",
        )
    )
    out.append(
        (
            "S3 alpha-ReQ is None on one eigenvalue (no slope from one point)",
            S.alpha_req([1.0]) is None,
            "None",
        )
    )
    return out


# ------------------------------------------------------------------ S4 (the licence)


def s4_reference_channel(cfg_path=HERE / "config.json", seeds=(0, 1)):
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["seeds"] = list(seeds)
    cfg["train"]["steps"] = min(cfg["train"]["steps"], 800)
    steps = R.measure_steps(cfg)
    pd_ = cfg["phase_detector"]
    verdict = {}
    for label, skew in (("skewed", cfg["data"]["skew"]), ("control", cfg["data"]["control_skew"])):
        three, two, depth = 0, 0, []
        for seed in cfg["seeds"]:
            rows = R.run_condition(cfg, skew, seed, steps)["rows"]
            st, va = A.series(rows, "REP")
            ph = A.phases(st, va, pd_["smooth_halfwidth"], pd_["margin_frac"])
            three += ph["three_phase"]
            two += ph["two_phase"]
            depth.append(ph["depth"])
        verdict[label] = {"three": three, "two": two, "depth": depth, "n": len(cfg["seeds"])}
    n = len(seeds)
    sk, ct = verdict["skewed"], verdict["control"]
    ok = sk["three"] > n / 2 and ct["three"] <= n / 2
    msg = (
        f"three-phase skewed {sk['three']}/{n}, control {ct['three']}/{n}; two-phase skewed {sk['two']}/{n}, "
        f"control {ct['two']}/{n}; first-leg depth skewed {min(sk['depth']):.2f}-{max(sk['depth']):.2f} "
        f"vs control {min(ct['depth']):.2f}-{max(ct['depth']):.2f}"
    )
    return [
        ("S4 reference channel: three-phase on skewed+bottleneck AND not on control", ok, msg)
    ], verdict


# ------------------------------------------------------------------ S5


def s5_axis_label():
    out = []
    for bad in ("", "   ", None, "BOGUS"):
        try:
            Spectrum(axis=bad, eigenvalues=[1.0])
            out.append((f"S5 Spectrum(axis={bad!r}) is refused", False, "constructed"))
        except (ValueError, TypeError):
            out.append((f"S5 Spectrum(axis={bad!r}) is refused", True, "raised"))
    # every spectrum in the latest results file carries a declared axis
    runs = sorted((HERE / "results").glob("*/raw.json"))
    if runs:
        raw = json.loads(runs[-1].read_text())
        n, badn = 0, 0
        for c in raw["conditions"].values():
            for ps in c["per_seed"]:
                for r in ps["rows"]:
                    for k, v in r.items():
                        if isinstance(v, dict) and "eigenvalues" in v:
                            n += 1
                            badn += (v.get("axis") not in S.AXES) or (v.get("axis") != k)
            for t, d in c["A3_by_step"].items():
                for k, v in d.items():
                    n += 1
                    badn += (v.get("axis") not in S.AXES) or (v.get("axis") != k)
        out.append(
            (
                f"S5 every emitted spectrum in {runs[-1].parent.name} carries its axis ({n} spectra)",
                badn == 0,
                f"{badn} unlabelled",
            )
        )
    return out


# ------------------------------------------------------------------ align.py known answers (not in the order's list; the tool has to read)


def align_known_answers():
    steps = list(range(0, 200, 2))
    three = [
        math.cos(3 * math.pi * t / 200) for t in steps
    ]  # +1 -> -1 -> +1 -> -1: fall, rise, fall
    mono = [t / 200 for t in steps]
    ph3 = A.phases(steps, three, 3, 0.1)
    phm = A.phases(steps, mono, 3, 0.1)
    inv = [-v for v in three]
    lock = A.align(steps, three, steps, three, 3, 0.5, 20)
    anti = A.align(steps, three, steps, inv, 3, 0.5, 20)
    rng = random.Random(3)
    noise = [rng.gauss(0, 1) for _ in steps]
    dec = A.align(steps, three, steps, noise, 3, 0.5, 20)
    return [
        (
            "align.phases finds three legs on a fall-rise-fall series",
            ph3["three_phase"],
            "three_phase True",
        ),
        (
            "align.phases finds no legs on a monotone series",
            not phm["nonmonotone"] and not phm["two_phase"],
            "monotone",
        ),
        (
            "align.align: identical series -> LOCKED",
            lock["verdict"] == "LOCKED",
            f"corr {lock['corr']:.3f}",
        ),
        (
            "align.align: negated series -> ANTI-PHASE",
            anti["verdict"] == "ANTI-PHASE",
            f"corr {anti['corr']:.3f}",
        ),
        (
            "align.align: white noise -> DECOUPLED",
            dec["verdict"] == "DECOUPLED",
            f"corr {dec['corr']:.3f}",
        ),
    ]


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    results = []
    results += s1_jacobi()
    results += s2_rankme()
    results += s3_alpha()
    results += align_known_answers()
    results += s5_axis_label()
    if "--skip-s4" not in argv:
        r4, _ = s4_reference_channel()
        results += r4
    bad = _report(results)
    if any(n.startswith("S4") for n in bad):
        print(
            "S4 FAILED: the reference channel does not reproduce the published three-phase shape here."
        )
        print(
            "This is the recorded state (RESULTS.md section 0). Section 4 of the order is not licensed."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
