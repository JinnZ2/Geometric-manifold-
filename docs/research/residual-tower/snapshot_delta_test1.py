#!/usr/bin/env python3
"""Snapshot-delta experiment, Test 1 (see SNAPSHOT_DELTA_EXPERIMENT.md).

Test 1 as the doc states it: "A large linear computation, done two ways:
incremental from the diff, and full recompute. Pass = bit-identical results.
Record the cost of each."

The doc leaves open which computation, its size, the snapshots, how cost is
measured and how many runs. Those are fixed here as [CHOICE] constants; each is
the builder's, not the order's.

Also built, because the order for this script names them as planted cases with
known answers (rules from the doc's "checks" entries):
  - coarsest level moved: arithmetic difference authoritative, XOR an upper bound
  - Gray cross-check for unit steps, index read from the LOWER value,
    flat binary only
  - 2D chunk parity, row-major: 1 error located; 2 detected, not located;
    3 corners miscorrect; 4 corners undetected; a lone flag = bad parity bit,
    data not certified

Scope: flat binary snapshots (lists of unsigned 64-bit integers), exact integer
arithmetic. Not built: tower codes, edge / tie flips, nonlinear bounds,
non-dyadic values, line operations. Their slots stay VOID.

Usage:
    python3 snapshot_delta_test1.py --self-test
    python3 snapshot_delta_test1.py --run --out snapshot_delta_test1_results.json

Exit codes: 0 pass; 1 a self-test case or the Test 1 pass criterion failed;
2 usage.
Stdlib only. Co-authored (JinnZ2 + Claude).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time

# --- [CHOICE] constants: not stated in the doc ------------------------------
SEED = 20261120
WORD_BITS = 64                 # one snapshot entry = one unsigned 64-bit word
N_ENTRIES = 200_000            # snapshot length for sum and dot
CHANGED = [1, 100, 10_000, 100_000, 200_000]   # entries changed between snapshots
MAT_ROWS, MAT_COLS = 8, 20_000                 # matmul: matrix (8 x 20000) times vector
MAT_CHANGED = [1, 100, 20_000]
TIMING_REPEATS = 5             # wall time = median of this many repeats
N_RIPPLE = 10_000              # random (old, new) pairs for the level comparison
RIPPLE_MAX_STEP = 1 << 16
N_GRAY_CLEAN = 10_000
N_GRAY_CORRUPT = 1_000
N_PARITY = 1_000               # random chunks per planted error pattern
SIDE = 8                       # 64-bit chunk laid out 8 x 8, row-major:
                               # bit index i = row * SIDE + col, bit 0 = least significant
MASK = (1 << WORD_BITS) - 1


# --- level moved: difference vs XOR -----------------------------------------

def xor_level(old, new):
    """Highest changed bit. Upper bound only (carry ripple overstates)."""
    x = old ^ new
    return x.bit_length() - 1 if x else None


def diff_level(old, new):
    """Highest set bit of |new - old|. Authoritative."""
    d = abs(new - old)
    return d.bit_length() - 1 if d else None


# --- Gray cross-check (unit steps, flat binary only) ------------------------

def gray(b):
    return b ^ (b >> 1)


def trailing_ones(b):
    k = 0
    while b & 1:
        k += 1
        b >>= 1
    return k


def gray_unit_check(old, new):
    """Check a claimed +/-1 step without using subtraction.

    Expected: exactly one differing Gray bit, at index = trailing-ones count of
    the LOWER of the two values. Returns AGREE or DISAGREE. The caller claims
    the step is a unit step; this function does not test that by arithmetic.
    """
    lower = old if old < new else new
    k = trailing_ones(lower)
    g = gray(old) ^ gray(new)
    return "AGREE" if g == (1 << k) else "DISAGREE"


def gray_scope(old, new):
    """Steps other than +/-1 are out of the check's scope."""
    return "IN_SCOPE" if abs(new - old) == 1 else "OUT_OF_SCOPE"


# --- 2D chunk parity, row-major ---------------------------------------------

def bit(r, c):
    return 1 << (r * SIDE + c)


def parities(chunk):
    rows = 0
    cols = 0
    for r in range(SIDE):
        row = (chunk >> (r * SIDE)) & ((1 << SIDE) - 1)
        if bin(row).count("1") & 1:
            rows |= 1 << r
    for c in range(SIDE):
        n = 0
        for r in range(SIDE):
            n += (chunk >> (r * SIDE + c)) & 1
        if n & 1:
            cols |= 1 << c
    return rows, cols


def parity_classify(chunk, stored_rows, stored_cols):
    """Classify a chunk against its stored parities.

    Returns a dict: status, failing rows, failing cols, and for LOCATED the
    (row, col) the parity points at. `certified` is always False: parity alone
    never certifies the data (4-corner blind spot, lone-flag case).
    """
    rows, cols = parities(chunk)
    fr = [r for r in range(SIDE) if ((rows ^ stored_rows) >> r) & 1]
    fc = [c for c in range(SIDE) if ((cols ^ stored_cols) >> c) & 1]
    out = {"rows": fr, "cols": fc, "certified": False}
    if not fr and not fc:
        out["status"] = "NO_FLAG"
    elif len(fr) == 1 and len(fc) == 1:
        out["status"] = "LOCATED"
        out["at"] = [fr[0], fc[0]]
    elif len(fr) + len(fc) == 1:
        out["status"] = "PARITY_BIT_BAD"      # lone flag: leave the data unchanged
    else:
        out["status"] = "DETECTED_NOT_LOCATED"
    return out


def parity_correct(chunk, stored_rows, stored_cols, reference):
    """Apply the doc's rule: correct only with a second independent check after
    correcting. The independent check here is the arithmetic difference against
    `reference` (in a test, the known-good chunk).

    Returns (chunk_out, outcome). Outcomes:
      CORRECTED               parity located one bit; flip confirmed by the second check
      MISCORRECTION_CAUGHT    parity located a bit; the second check rejects the flip;
                              the data is returned unflipped
      UNCHANGED_<status>      nothing was flipped
    """
    c = parity_classify(chunk, stored_rows, stored_cols)
    if c["status"] != "LOCATED":
        return chunk, "UNCHANGED_" + c["status"]
    r, col = c["at"]
    flipped = chunk ^ bit(r, col)
    if flipped - reference == 0:
        return flipped, "CORRECTED"
    return chunk, "MISCORRECTION_CAUGHT"


# --- linear computations, two ways ------------------------------------------

def diff_entries(old, new):
    """Scan both snapshots; return [(index, new - old)] for changed entries."""
    return [(i, b - a) for i, (a, b) in enumerate(zip(old, new)) if a != b]


def full_sum(v):
    return sum(v)


def inc_sum(f_old, delta):
    return f_old + sum(d for _, d in delta)


def full_dot(w, v):
    return sum(a * b for a, b in zip(w, v))


def inc_dot(f_old, w, delta):
    return f_old + sum(w[i] * d for i, d in delta)


def full_matvec(M, v):
    return [sum(a * b for a, b in zip(row, v)) for row in M]


def inc_matvec(f_old, M, delta):
    return [f + sum(row[i] * d for i, d in delta) for f, row in zip(f_old, M)]


def as_bytes(x):
    """Canonical bytes of an exact result (int or list of ints)."""
    xs = x if isinstance(x, list) else [x]
    return b"|".join(str(v).encode() for v in xs)


def bit_identical(a, b):
    return as_bytes(a) == as_bytes(b)


def make_snapshots(rng, n, k):
    old = [rng.getrandbits(WORD_BITS) for _ in range(n)]
    new = list(old)
    for i in rng.sample(range(n), k):
        v = rng.getrandbits(WORD_BITS)
        while v == old[i]:
            v = rng.getrandbits(WORD_BITS)
        new[i] = v
    return old, new


def median_time(fn, repeats):
    ts = []
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return out, ts[len(ts) // 2]


# --- planted cases with known answers ---------------------------------------

def _chunk_with(errors, base=0x0123456789ABCDEF):
    c = base
    for r, col in errors:
        c ^= bit(r, col)
    return c


def planted_cases():
    """Each case: (name, callable returning the observed value, known answer)."""
    base = 0x0123456789ABCDEF
    pr, pc = parities(base)
    three = _chunk_with([(1, 2), (1, 5), (4, 2)])
    four = _chunk_with([(1, 2), (1, 5), (4, 2), (4, 5)])
    cases = [
        # difference authoritative over XOR
        ("level 0111->1000: xor level", lambda: xor_level(0b0111, 0b1000), 3),
        ("level 0111->1000: diff level", lambda: diff_level(0b0111, 0b1000), 0),
        ("level 1000->0111: diff level", lambda: diff_level(0b1000, 0b0111), 0),
        ("level 0100->0110: xor = diff", lambda: (xor_level(4, 6), diff_level(4, 6)), (1, 1)),
        ("level no change", lambda: (xor_level(5, 5), diff_level(5, 5)), (None, None)),
        # Gray, index from the lower value
        ("gray 7->8 flips bit 3", lambda: gray(7) ^ gray(8), 1 << 3),
        ("gray 8->7 flips bit 3 (lower value 7)", lambda: gray(8) ^ gray(7), 1 << 3),
        ("gray 7->8 check", lambda: gray_unit_check(7, 8), "AGREE"),
        ("gray 8->7 check", lambda: gray_unit_check(8, 7), "AGREE"),
        ("gray 4->5 check", lambda: gray_unit_check(4, 5), "AGREE"),
        ("gray 7->8 with bit 4 of new corrupted", lambda: gray_unit_check(7, 8 ^ 16), "DISAGREE"),
        ("gray 7->8 with bit 3 of new corrupted", lambda: gray_unit_check(7, 8 ^ 8), "DISAGREE"),
        ("gray 5->6 with bit 1 of new corrupted lands on 4, the other neighbour: not flagged",
         lambda: gray_unit_check(5, 6 ^ 2), "AGREE"),
        ("gray step 1->3 scope", lambda: gray_scope(1, 3), "OUT_OF_SCOPE"),
        ("gray step >1: top Gray bit = top XOR bit",
         lambda: (gray(1) ^ gray(6)).bit_length() == (1 ^ 6).bit_length(), True),
        # 2D parity, row-major
        ("parity clean chunk", lambda: parity_classify(base, pr, pc)["status"], "NO_FLAG"),
        ("parity 1 error located at (3,6)",
         lambda: parity_classify(_chunk_with([(3, 6)]), pr, pc).get("at"), [3, 6]),
        ("parity 1 error corrected and confirmed",
         lambda: parity_correct(_chunk_with([(3, 6)]), pr, pc, base), (base, "CORRECTED")),
        ("parity 2 errors, same row",
         lambda: parity_classify(_chunk_with([(2, 1), (2, 6)]), pr, pc)["status"],
         "DETECTED_NOT_LOCATED"),
        ("parity 2 errors, same column",
         lambda: parity_classify(_chunk_with([(0, 4), (7, 4)]), pr, pc)["status"],
         "DETECTED_NOT_LOCATED"),
        ("parity 2 errors, different row and column",
         lambda: parity_classify(_chunk_with([(0, 0), (5, 3)]), pr, pc)["status"],
         "DETECTED_NOT_LOCATED"),
        ("parity 3 corners point at the 4th corner (4,5)",
         lambda: parity_classify(three, pr, pc).get("at"), [4, 5]),
        ("parity 3 corners: flipping the 4th corner passes parity",
         lambda: parity_classify(three ^ bit(4, 5), pr, pc)["status"], "NO_FLAG"),
        ("parity 3 corners: second check catches the miscorrection",
         lambda: parity_correct(three, pr, pc, base), (three, "MISCORRECTION_CAUGHT")),
        ("parity 4 corners undetected",
         lambda: (parity_classify(four, pr, pc)["status"], four != base), ("NO_FLAG", True)),
        ("parity lone row flag = bad parity bit",
         lambda: parity_classify(base, pr ^ (1 << 2), pc)["status"], "PARITY_BIT_BAD"),
        ("parity lone column flag = bad parity bit",
         lambda: parity_classify(base, pr, pc ^ (1 << 6))["status"], "PARITY_BIT_BAD"),
        ("parity lone flag leaves the data unchanged",
         lambda: parity_correct(base, pr ^ 1, pc, base), (base, "UNCHANGED_PARITY_BIT_BAD")),
        ("parity lone flag over 4 bad data bits: not certified, data is bad",
         lambda: (parity_classify(four, pr ^ 1, pc)["status"],
                  parity_classify(four, pr ^ 1, pc)["certified"], four != base),
         ("PARITY_BIT_BAD", False, True)),
        ("parity never certifies",
         lambda: parity_classify(base, pr, pc)["certified"], False),
        # linear, two ways
        ("sum incremental = full",
         lambda: inc_sum(full_sum([1, 2, 3]), diff_entries([1, 2, 3], [1, 9, 3])),
         full_sum([1, 9, 3])),
        ("dot incremental = full",
         lambda: inc_dot(full_dot([2, 3, 5], [1, 2, 3]), [2, 3, 5],
                         diff_entries([1, 2, 3], [0, 2, 7])), 2 * 0 + 3 * 2 + 5 * 7),
        ("matvec incremental = full",
         lambda: inc_matvec(full_matvec([[1, 2], [3, 4]], [5, 6]), [[1, 2], [3, 4]],
                            diff_entries([5, 6], [5, 1])), [1 * 5 + 2 * 1, 3 * 5 + 4 * 1]),
        ("negative delta handled", lambda: inc_sum(10, diff_entries([7, 3], [2, 3])), 5),
    ]
    return cases


def planted_wrong_cases():
    """Same computations as above, each paired with a WRONG answer. The checker
    must report every one of these as failing."""
    base = 0x0123456789ABCDEF
    pr, pc = parities(base)
    three = _chunk_with([(1, 2), (1, 5), (4, 2)])
    return [
        ("WRONG: diff level of 0111->1000 is 3", lambda: diff_level(0b0111, 0b1000), 3),
        ("WRONG: gray 8->7 flips bit 0", lambda: gray(8) ^ gray(7), 1),
        ("WRONG: corrupted unit step agrees", lambda: gray_unit_check(7, 8 ^ 16), "AGREE"),
        ("WRONG: 2 errors are located",
         lambda: parity_classify(_chunk_with([(0, 0), (5, 3)]), pr, pc)["status"], "LOCATED"),
        ("WRONG: 3 corners are corrected",
         lambda: parity_correct(three, pr, pc, base)[1], "CORRECTED"),
        ("WRONG: 4 corners are detected",
         lambda: parity_classify(_chunk_with([(1, 2), (1, 5), (4, 2), (4, 5)]), pr, pc)["status"],
         "DETECTED_NOT_LOCATED"),
        ("WRONG: a lone flag certifies the data",
         lambda: parity_classify(base, pr ^ 1, pc)["certified"], True),
        ("WRONG: incremental that drops one delta entry is identical",
         lambda: bit_identical(inc_sum(6, diff_entries([1, 2, 3], [4, 9, 3])[:-1]),
                               full_sum([4, 9, 3])), True),
    ]


def check_cases(cases):
    """Return the list of failing cases as (name, observed, expected)."""
    fails = []
    for name, fn, expected in cases:
        try:
            got = fn()
        except Exception as e:                     # a crash is a failure, not a pass
            got = "EXCEPTION: %r" % (e,)
        if got != expected:
            fails.append((name, got, expected))
    return fails


def self_test(verbose=True):
    good = planted_cases()
    wrong = planted_wrong_cases()
    fails = check_cases(good)
    wrong_fails = check_cases(wrong)
    not_caught = [n for n, _, _ in wrong if n not in {w[0] for w in wrong_fails}]
    ok = (not fails) and (not not_caught)
    report = {
        "self_test": "PASS" if ok else "FAIL",
        "planted_cases": len(good),
        "planted_cases_failed": [{"case": n, "observed": repr(g), "expected": repr(e)}
                                 for n, g, e in fails],
        "planted_wrong_cases": len(wrong),
        "planted_wrong_reported_failing": len(wrong_fails),
        "planted_wrong_not_caught": not_caught,
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return ok, report


# --- the run ------------------------------------------------------------------

def run_linear(rng):
    out = {"sum": [], "dot": [], "matvec": []}
    mismatches = 0
    comparisons = 0

    w = [rng.getrandbits(WORD_BITS) for _ in range(N_ENTRIES)]
    for k in CHANGED:
        old, new = make_snapshots(rng, N_ENTRIES, k)
        f_old_sum, f_old_dot = full_sum(old), full_dot(w, old)
        delta, t_scan = median_time(lambda: diff_entries(old, new), TIMING_REPEATS)
        assert len(delta) == k

        full_s, t_full_s = median_time(lambda: full_sum(new), TIMING_REPEATS)
        inc_s, t_inc_s = median_time(lambda: inc_sum(f_old_sum, delta), TIMING_REPEATS)
        same = bit_identical(full_s, inc_s)
        comparisons += 1
        mismatches += 0 if same else 1
        out["sum"].append(_row(k, N_ENTRIES, N_ENTRIES, k, same, t_full_s, t_inc_s, t_scan,
                               unit="additions"))

        full_d, t_full_d = median_time(lambda: full_dot(w, new), TIMING_REPEATS)
        inc_d, t_inc_d = median_time(lambda: inc_dot(f_old_dot, w, delta), TIMING_REPEATS)
        same = bit_identical(full_d, inc_d)
        comparisons += 1
        mismatches += 0 if same else 1
        out["dot"].append(_row(k, N_ENTRIES, N_ENTRIES, k, same, t_full_d, t_inc_d, t_scan,
                               unit="multiply-adds"))

    M = [[rng.getrandbits(WORD_BITS) for _ in range(MAT_COLS)] for _ in range(MAT_ROWS)]
    for k in MAT_CHANGED:
        old, new = make_snapshots(rng, MAT_COLS, k)
        f_old = full_matvec(M, old)
        delta, t_scan = median_time(lambda: diff_entries(old, new), TIMING_REPEATS)
        full_m, t_full = median_time(lambda: full_matvec(M, new), TIMING_REPEATS)
        inc_m, t_inc = median_time(lambda: inc_matvec(f_old, M, delta), TIMING_REPEATS)
        same = bit_identical(full_m, inc_m)
        comparisons += 1
        mismatches += 0 if same else 1
        out["matvec"].append(_row(k, MAT_COLS, MAT_ROWS * MAT_COLS, MAT_ROWS * k, same,
                                  t_full, t_inc, t_scan, unit="multiply-adds"))
    return out, comparisons, mismatches


def _row(k, n, ops_full, ops_inc, same, t_full, t_inc, t_scan, unit):
    return {
        "entries": n,
        "changed": k,
        "bit_identical": same,
        "ops_unit": unit,
        "ops_full": ops_full,
        "ops_incremental": ops_inc,
        "ops_diff_scan_compares": n,
        "speedup_ops_excluding_scan": round(ops_full / ops_inc, 4),
        "speedup_ops_including_scan": round(ops_full / (ops_inc + n), 4),
        "wall_s_full": round(t_full, 6),
        "wall_s_incremental": round(t_inc, 6),
        "wall_s_diff_scan": round(t_scan, 6),
        "speedup_wall_excluding_scan": round(t_full / t_inc, 2) if t_inc else None,
        "speedup_wall_including_scan": round(t_full / (t_inc + t_scan), 4),
    }


def run_levels(rng):
    disagree = 0
    bound_violations = 0
    for _ in range(N_RIPPLE):
        old = rng.getrandbits(WORD_BITS)
        step = rng.randint(1, RIPPLE_MAX_STEP) * rng.choice((1, -1))
        new = old + step
        if not 0 <= new <= MASK:
            new = old - step
        xl, dl = xor_level(old, new), diff_level(old, new)
        if xl != dl:
            disagree += 1
        if dl > xl:
            bound_violations += 1
    return {"pairs": N_RIPPLE, "max_step": RIPPLE_MAX_STEP,
            "carry_ripple_count": disagree,
            "xor_below_difference_count": bound_violations}


def run_gray(rng):
    agree = disagree = 0
    for _ in range(N_GRAY_CLEAN):
        b = rng.randrange(1, MASK)
        new = b + rng.choice((1, -1))
        if gray_unit_check(b, new) == "AGREE":
            agree += 1
        else:
            disagree += 1
    c_agree = c_disagree = c_agree_other_neighbour = 0
    for _ in range(N_GRAY_CORRUPT):
        b = rng.randrange(1, MASK)
        new = (b + rng.choice((1, -1))) ^ (1 << rng.randrange(WORD_BITS))
        if gray_unit_check(b, new) == "AGREE":
            c_agree += 1
            if abs(new - b) == 1:      # the flip turned b+1 into b-1 or the reverse
                c_agree_other_neighbour += 1
        else:
            c_disagree += 1
    return {"clean_unit_steps": N_GRAY_CLEAN,
            "gray_unit_agree_count": agree,
            "gray_unit_disagree_count": disagree,
            "corrupted_steps": N_GRAY_CORRUPT,
            "corrupted_flagged": c_disagree,
            "corrupted_not_flagged": c_agree,
            "corrupted_not_flagged_other_unit_neighbour": c_agree_other_neighbour}


def run_parity(rng):
    res = {"chunks_per_pattern": N_PARITY, "side": SIDE}
    located = corrected = 0
    two_detected = two_located = two_missed = 0
    three_points_at_good = three_parity_clean_after = three_caught = 0
    four_missed = 0
    lone_flag = lone_changed = 0
    for _ in range(N_PARITY):
        base = rng.getrandbits(WORD_BITS)
        pr, pc = parities(base)
        # 1 error
        r, c = rng.randrange(SIDE), rng.randrange(SIDE)
        cl = parity_classify(base ^ bit(r, c), pr, pc)
        if cl["status"] == "LOCATED" and cl["at"] == [r, c]:
            located += 1
        if parity_correct(base ^ bit(r, c), pr, pc, base) == (base, "CORRECTED"):
            corrected += 1
        # 2 errors
        p1, p2 = rng.sample(range(SIDE * SIDE), 2)
        st = parity_classify(base ^ (1 << p1) ^ (1 << p2), pr, pc)["status"]
        if st == "DETECTED_NOT_LOCATED":
            two_detected += 1
        elif st == "LOCATED":
            two_located += 1
        else:
            two_missed += 1
        # rectangle
        r1, r2 = rng.sample(range(SIDE), 2)
        c1, c2 = rng.sample(range(SIDE), 2)
        three = base ^ bit(r1, c1) ^ bit(r1, c2) ^ bit(r2, c1)
        cl = parity_classify(three, pr, pc)
        if cl["status"] == "LOCATED" and cl["at"] == [r2, c2]:
            three_points_at_good += 1
            if parity_classify(three ^ bit(r2, c2), pr, pc)["status"] == "NO_FLAG":
                three_parity_clean_after += 1
        if parity_correct(three, pr, pc, base) == (three, "MISCORRECTION_CAUGHT"):
            three_caught += 1
        four = three ^ bit(r2, c2)
        if parity_classify(four, pr, pc)["status"] == "NO_FLAG" and four != base:
            four_missed += 1
        # lone flag: one stored parity bit bad
        if rng.random() < 0.5:
            bpr, bpc = pr ^ (1 << rng.randrange(SIDE)), pc
        else:
            bpr, bpc = pr, pc ^ (1 << rng.randrange(SIDE))
        if parity_classify(base, bpr, bpc)["status"] == "PARITY_BIT_BAD":
            lone_flag += 1
        if parity_correct(base, bpr, bpc, base)[0] != base:
            lone_changed += 1
    res.update({
        "parity_located_count": located,
        "one_error_corrected_and_confirmed": corrected,
        "parity_detected_unlocated_count": two_detected,
        "two_errors_wrongly_located": two_located,
        "two_errors_not_detected": two_missed,
        "miscorrection_count": three_points_at_good,
        "three_corner_flip_passes_parity_after": three_parity_clean_after,
        "three_corner_caught_by_second_check": three_caught,
        "rectangle_miss_count": four_missed,
        "lone_flag_classified_parity_bit_bad": lone_flag,
        "lone_flag_data_changed": lone_changed,
    })
    return res


def run(out_path):
    ok, st = self_test(verbose=False)
    if not ok:
        print(json.dumps(st, indent=2))
        print("self-test failed; the run is refused", file=sys.stderr)
        return 1
    rng = random.Random(SEED)
    linear, comparisons, mismatches = run_linear(rng)
    levels = run_levels(rng)
    gray_res = run_gray(rng)
    parity = run_parity(rng)

    def no_wall(rows):
        return [{k: v for k, v in r.items() if "wall" not in k} for r in rows]

    deterministic = {
        "linear": {name: no_wall(rows) for name, rows in linear.items()},
        "linear_comparisons": comparisons,
        "mismatch_count": mismatches,
        "levels": levels,
        "gray": gray_res,
        "parity": parity,
    }
    digest = hashlib.sha256(
        json.dumps(deterministic, sort_keys=True).encode()).hexdigest()
    result = {
        "step": "snapshot_delta_test1",
        "status": "ran",
        "test1_pass": mismatches == 0,
        "python": sys.version.split()[0],
        "seed": SEED,
        "choices": {
            "note": "not stated in SNAPSHOT_DELTA_EXPERIMENT.md; fixed by the builder",
            "snapshots": "lists of unsigned 64-bit words, exact integer arithmetic",
            "computations": ["sum", "dot with a fixed 64-bit weight vector",
                             "matrix (8 x 20000) times vector"],
            "entries": N_ENTRIES, "changed": CHANGED,
            "matvec_shape": [MAT_ROWS, MAT_COLS], "matvec_changed": MAT_CHANGED,
            "cost": "operation counts (deterministic) and wall time, median of "
                    "%d repeats (this machine, this run)" % TIMING_REPEATS,
            "bit_identical": "equal canonical bytes of the exact integer results",
            "parity_layout": "8 x 8 row-major, bit index = row * 8 + col, bit 0 least significant",
            "second_check": "arithmetic difference against the known-good chunk",
        },
        "self_test": st,
        "deterministic": deterministic,
        "deterministic_sha256": digest,
        "timing_this_run": linear,
        "slots": {
            "speedup": "table, not one number: deterministic.linear[*].speedup_ops_* and "
                       "timing_this_run[*].speedup_wall_*; depends on changed / entries",
            "mismatch count": mismatches,
            "carry_ripple_count": levels["carry_ripple_count"],
            "gray_unit_agree_count": gray_res["gray_unit_agree_count"],
            "gray_unit_disagree_count": gray_res["gray_unit_disagree_count"],
            "parity_located_count": parity["parity_located_count"],
            "parity_detected_unlocated_count": parity["parity_detected_unlocated_count"],
            "rectangle_miss_count": parity["rectangle_miss_count"],
            "miscorrection_count": parity["miscorrection_count"],
            "edge-flip count": "VOID",
            "tie_flip_count": "VOID",
            "nonlinear bound width": "VOID",
            "nondyadic_nonzero_residual_count": "VOID",
            "line_op_speedup": "VOID",
        },
        "not_built": ["tower codes (encode_tower) as snapshots", "edge and tie flips",
                      "nonlinear bounds", "non-dyadic values", "line operations"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("self-test         PASS (%d planted, %d planted-wrong all reported failing)"
          % (st["planted_cases"], st["planted_wrong_cases"]))
    print("linear comparisons %d   mismatch count %d   test1_pass %s"
          % (comparisons, mismatches, result["test1_pass"]))
    for name, rows in linear.items():
        for r in rows:
            print("  %-6s changed %7d / %7d  ops x%-10s (with scan x%-7s)  "
                  "wall x%-9s (with scan x%s)"
                  % (name, r["changed"], r["entries"], r["speedup_ops_excluding_scan"],
                     r["speedup_ops_including_scan"], r["speedup_wall_excluding_scan"],
                     r["speedup_wall_including_scan"]))
    print("levels   %s" % json.dumps(levels))
    print("gray     %s" % json.dumps(gray_res))
    print("parity   %s" % json.dumps(parity))
    print("deterministic sha256 %s" % digest)
    print("-> %s" % out_path)
    return 0 if mismatches == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default="snapshot_delta_test1_results.json")
    args = ap.parse_args(argv)
    if args.self_test:
        ok, _ = self_test()
        return 0 if ok else 1
    if args.run:
        return run(args.out)
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
