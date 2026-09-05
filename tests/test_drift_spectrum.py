"""Smoke and known-answer tests for sims/drift_spectrum (stdlib-only folder).

S1, S2, S3, S5 and the align.py known answers must pass. S4 is the LICENCE test of the
work order and is recorded FAILED in sims/drift_spectrum/RESULTS.md section 0; the test
below PINS that state so a change in either direction turns the suite red and forces the
report to be re-emitted, rather than passing silently.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "sims" / "drift_spectrum"
sys.path.insert(0, str(HERE))

import selftest as T  # noqa: E402


def _all_ok(results):
    bad = [(n, msg) for n, ok, msg in results if not ok]
    assert not bad, bad


def test_s1_jacobi():
    _all_ok(T.s1_jacobi())


def test_s2_rankme():
    _all_ok(T.s2_rankme())


def test_s3_alpha_req():
    _all_ok(T.s3_alpha())


def test_s5_axis_label_is_hard_failure():
    _all_ok(T.s5_axis_label())


def test_align_known_answers():
    _all_ok(T.align_known_answers())


def test_inventory_facts_still_hold():
    import inventory as I

    assert I.selftest() == 0


def test_s4_pinned_state():
    """PINNED: S4 fails on the third phase, two phases reproduce on both conditions, and the
    first-leg depth separates skewed from control. If the generator or detector changes so
    that any of these moves, re-emit RESULTS.md and update this pin -- do not loosen it."""
    results, v = T.s4_reference_channel(seeds=(0, 1))
    assert results[0][1] is False, "S4 now PASSES: re-run report.py, RESULTS.md section 0 is stale"
    n = v["skewed"]["n"]
    assert v["skewed"]["three"] == 0 and v["control"]["three"] == 0
    assert v["skewed"]["two"] == n and v["control"]["two"] == n
    assert min(v["skewed"]["depth"]) > max(v["control"]["depth"])
