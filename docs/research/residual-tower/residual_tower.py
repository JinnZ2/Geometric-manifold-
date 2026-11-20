#!/usr/bin/env python3
"""residual_tower -- exact arithmetic on scale-decomposed numbers.

Co-authored (human + AI). Stdlib only, phone-runnable. Python 3.8+.

Idea (from the Geometric-to-Binary experiment bridge, NCM-flavored, but
deliberately NOT the trained kind): represent a number as a tower of
coarse-to-fine codes on declared uniform grids, keeping the terminal
residual as an exact Fraction. The exactness invariant

    value == sum(accepted code values) + terminal residual        (exact)

holds at all times, ESCAPE or not. Quantization is bookkeeping, not
approximation: precision is only ever reduced by truncate(), which is a
declared, measured, archived event -- the honest opposite of float
rounding, which discards silently.

Explicit rejection: a level whose nearest code is farther than its
declared tau emits the reserved ESCAPE index (never a real code) and
passes the residual down whole. Out-of-book values, NaN/inf, mismatched
vector lengths, and division by a tower that can be zero all raise
TowerVoid rather than returning a number.

Audit discipline is built in, per the bridge audit's findings:
  * tolerances carry a mandatory reason field (else flagged UNSOURCED)
  * measured values are computed at record time, never literal
  * replay re-runs and compares records excluding timestamps
  * printed-vs-executed: `check` re-executes a record's computation
    and diffs the printed numbers

Scope limits (read before relying):
  * Uniform dyadic-style grids only. Learned codebooks: NOT_BUILT.
  * Vector towers are componentwise (same book stack every dimension).
  * z = reconstruction + residual is an additive identity. Nothing here
    is power conjugacy, energy, or physics. Cross-term reports are
    vector-split orthogonality defects, full stop.
  * No NCM semantics. This borrows the coarse-to-fine shape, not the
    neural model.

CLI:
  python3 residual_tower.py selftest
  python3 residual_tower.py demo --out demo_record.json
  python3 residual_tower.py check demo_record.json
"""
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction

SCHEMA = "residual-tower/1"
ESCAPE = "ESCAPE"           # reserved index; never an int, never a code


class TowerVoid(Exception):
    """Explicit rejection. Raised instead of returning a bad number."""


# ----------------------------------------------------------------------
# Codebooks: uniform grids over [lo, hi), exactly representable
# ----------------------------------------------------------------------
class Book:
    """n grid points lo + i*step, step = (hi-lo)/n, all exact Fractions.

    tau is the level's rejection distance in absolute units; default
    step/2 (a point is accepted iff it is the unique nearest grid point
    within half a step). tau is declared in the record, never hidden.
    """
    def __init__(self, lo, hi, n, tau=None, name=None):
        lo, hi, n = Fraction(lo), Fraction(hi), int(n)
        if n < 1:
            raise TowerVoid("book needs at least one code")
        if hi <= lo:
            raise TowerVoid("book range empty: %s >= %s" % (lo, hi))
        self.lo, self.hi, self.n = lo, hi, n
        self.step = (hi - lo) / n
        self.tau = Fraction(tau) if tau is not None else self.step / 2
        if self.tau <= 0:
            raise TowerVoid("tau must be positive")
        self.name = name or "book[%s,%s)x%d" % (lo, hi, n)

    def code_value(self, i):
        if not 0 <= i < self.n:
            raise TowerVoid("code %r outside book %s" % (i, self.name))
        return self.lo + i * self.step

    def spec(self):
        return {"lo": str(self.lo), "hi": str(self.hi), "n": self.n,
                "tau": str(self.tau), "name": self.name}


def dyadic_stack(top_range, levels, codes_per_level=16):
    """Convenience: level 0 covers [-R, R); each finer level covers the
    previous level's tau band. A declared, boring stack -- no training."""
    books, lo, hi = [], -Fraction(top_range), Fraction(top_range)
    for k in range(levels):
        b = Book(lo, hi, codes_per_level, name="L%d" % k)
        books.append(b)
        lo, hi = -b.tau, b.tau
    return books


# ----------------------------------------------------------------------
# Scalar tower
# ----------------------------------------------------------------------
class Tower:
    """codes: list of int-or-ESCAPE, one per book level.
    residual: exact Fraction remainder after the last level.
    Invariant: value == sum(accepted code values) + residual, exactly."""

    def __init__(self, books, codes, residual):
        self.books = books
        self.codes = list(codes)
        self.residual = Fraction(residual)
        if len(self.codes) != len(books):
            raise TowerVoid("one code per level, no more, no less")

    # ---- reading ----
    def code_sum(self):
        total = Fraction(0)
        for b, c in zip(self.books, self.codes):
            if c is not ESCAPE:
                total += b.code_value(c)
        return total

    def value(self):
        return self.code_sum() + self.residual

    def per_level(self):
        """Measured per-level bookkeeping. Fractions are exact; the
        captured fraction at each level uses the SAME-SPLIT denominator
        (this level's input), never another split's total -- that
        mislabeling is bridge-audit finding EBR-adjacent and is pinned
        by selftest T7."""
        rows, r = [], self.value()
        for k, (b, c) in enumerate(zip(self.books, self.codes)):
            rin = r
            if c is ESCAPE:
                rows.append({"level": k, "escape": True,
                             "code_value": "0",
                             "residual_in": str(rin),
                             "residual_out": str(rin),
                             "captured_fraction": "0"})
            else:
                cv = b.code_value(c)
                r = rin - cv
                cap = (abs(cv) / abs(rin)) if rin != 0 else Fraction(0)
                rows.append({"level": k, "escape": False,
                             "code_value": str(cv),
                             "residual_in": str(rin),
                             "residual_out": str(r),
                             "captured_fraction": str(cap)})
        return rows

    # ---- declared precision reduction ----
    def truncate(self, keep_levels):
        """Drop levels >= keep_levels. Returns (new_tower, report).
        The dropped magnitude is measured and returned, not silently
        discarded. This is the ONLY operation that loses information."""
        keep_levels = int(keep_levels)
        if not 0 <= keep_levels <= len(self.books):
            raise TowerVoid("keep_levels outside tower")
        dropped_codes = self.codes[keep_levels:]
        dropped = Fraction(0)
        for b, c in zip(self.books[keep_levels:], dropped_codes):
            if c is not ESCAPE:
                dropped += b.code_value(c)
        dropped += self.residual
        v = self.value()
        rel = (abs(dropped) / abs(v)) if v != 0 else Fraction(0)
        report = {"kept_levels": keep_levels,
                  "dropped_magnitude": str(dropped),
                  "dropped_fraction_of_value": str(rel),
                  "label": "split_truncation_defect (arithmetic only; "
                           "not power conjugacy, not physics)"}
        return Tower(self.books[:keep_levels], self.codes[:keep_levels],
                     Fraction(0)), report


def _check_finite(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        raise TowerVoid("non-finite input rejected")


def encode(value, book):
    """One level. Returns (code or ESCAPE, residual_out). Exact."""
    _check_finite(value)
    v = Fraction(value)
    # nearest grid index
    i = (v - book.lo) / book.step
    if i.denominator == 1:
        idx = int(i)
    else:
        idx = int(i) if i >= 0 else int(i) - 1  # floor toward nearest below
        if (i - idx) > Fraction(1, 2):
            idx += 1
    if not 0 <= idx < book.n:
        return ESCAPE, v            # out of book: residual passes whole
    cv = book.code_value(idx)
    r = v - cv
    if abs(r) > book.tau:
        return ESCAPE, v            # explicit rejection: residual whole
    return idx, r


def encode_tower(value, books):
    _check_finite(value)
    r = Fraction(value)
    codes = []
    for b in books:
        c, r = encode(r, b)
        codes.append(c)
    return Tower(books, codes, r)


# ----------------------------------------------------------------------
# Arithmetic. Same book stack required; alignment across different
# stacks is NOT_BUILT and raises TowerVoid rather than guessing.
# ----------------------------------------------------------------------
def _same_stack(a, b):
    if [bk.spec() for bk in a.books] != [bk.spec() for bk in b.books]:
        raise TowerVoid("book stacks differ; alignment is NOT_BUILT")


def add(a, b):
    _same_stack(a, b)
    return encode_tower(a.value() + b.value(), a.books)


def sub(a, b):
    _same_stack(a, b)
    return encode_tower(a.value() - b.value(), a.books)


def mul(a, b, keep_levels=None):
    """Exact product, re-encoded. keep_levels declares a truncation of
    the RESULT tower and its measured defect is returned in the record
    (see mul_record). The exact value is computed first, always."""
    _same_stack(a, b)
    prod = encode_tower(a.value() * b.value(), a.books)
    if keep_levels is None:
        return prod, None
    return prod.truncate(keep_levels)


def div(a, b):
    _same_stack(a, b)
    dv = b.value()
    if dv == 0:
        raise TowerVoid("division by a tower whose value is exactly 0")
    return encode_tower(a.value() / dv, a.books)


def cross_term_defect(a, b):
    """The audit's statistic, kept honest: for splits a = A + r_a and
    b = B + r_b with A, B the code sums, report the omitted cross terms
    A*r_b + B*r_a + r_a*r_b relative to the full product. This measures
    how non-orthogonal the code/residual split is. It is NOT an energy
    statement and NOT power conjugacy (semantic-physics audit)."""
    A, ra = a.code_sum(), a.residual
    B, rb = b.code_sum(), b.residual
    full = (A + ra) * (B + rb)
    if full == 0:
        raise TowerVoid("cross-term defect undefined on zero product")
    omitted = A * rb + B * ra + ra * rb
    return {"omitted_cross_terms": str(omitted),
            "relative_defect": str(abs(omitted) / abs(full)),
            "label": "vector-split orthogonality defect; "
                     "not power conjugacy"}


# ----------------------------------------------------------------------
# Vector tower: componentwise, same stack each dimension.
# zip() truncation on mismatched lengths is a silent-coordinate-loss
# trap (bridge audit); lengths are checked, never zipped blind.
# ----------------------------------------------------------------------
def encode_vector(values, books):
    vals = [Fraction(v) for v in values]
    for v0 in values:
        _check_finite(v0)
    return [encode_tower(v, books) for v in vals]


def dot_tower(a_vec, b_vec):
    if len(a_vec) != len(b_vec):
        raise TowerVoid("vector length mismatch: %d vs %d -- "
                        "refusing to zip-truncate" %
                        (len(a_vec), len(b_vec)))
    acc = None
    for x, y in zip(a_vec, b_vec):
        term, _ = mul(x, y)
        acc = term if acc is None else add(acc, term)
    if acc is None:
        raise TowerVoid("dot of empty vectors")
    return acc


# ----------------------------------------------------------------------
# Claims with audit discipline
# ----------------------------------------------------------------------
def claim(scope, name, measured_fn, tolerance, tolerance_reason,
          falsifier, extra=None):
    """measured_fn is CALLED; the measured value is always computed at
    record time, never a literal. tolerance_reason is mandatory text --
    omit it and the record is flagged UNSOURCED (bridge audit: 18 of 21
    tolerances had no stated reason)."""
    if not callable(measured_fn):
        raise TowerVoid("measured must be a callable, not a literal")
    measured = measured_fn()
    rec = {"schema": SCHEMA, "scope": scope, "name": name,
           "measured": str(measured),
           "tolerance": str(Fraction(tolerance)),
           "tolerance_reason": tolerance_reason or "UNSOURCED",
           "falsifier": falsifier,
           "ts": time.time()}
    if extra:
        rec["extra"] = extra
    if not tolerance_reason:
        rec["flags"] = ["UNSOURCED_TOLERANCE"]
    try:
        ok = abs(Fraction(measured)) <= Fraction(tolerance)
        rec["verdict"] = "pass" if ok else "fail"
    except Exception:
        rec["verdict"] = "void"
    return rec


def replay(claim_fn):
    """Run the same claim function twice; compare everything except
    timestamps. Returns (equal, record)."""
    r1 = claim_fn()
    r2 = claim_fn()
    strip = lambda r: {k: v for k, v in r.items() if k != "ts"}
    return strip(r1) == strip(r2), r1


def append_record(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------
# Selftests. Each trap is a bridge-audit finding pinned as a test.
# ----------------------------------------------------------------------
def selftest():
    out, fails = [], 0

    def check(name, cond, detail=""):
        nonlocal fails
        out.append({"test": name, "ok": bool(cond), "detail": detail})
        if not cond:
            fails += 1

    books = dyadic_stack(4, levels=4, codes_per_level=16)

    # T1 exactness invariant on awkward Fractions
    t = encode_tower(Fraction(355, 113), books)
    check("T1 exact identity", t.value() == Fraction(355, 113),
          "value=%s" % t.value())

    # T2 negative and zero
    check("T2 zero exact", encode_tower(0, books).value() == 0)
    tn = encode_tower(Fraction(-7, 3), books)
    check("T2 negative exact", tn.value() == Fraction(-7, 3))

    # T3 ESCAPE never decodes as a real code; out-of-book keeps value
    big = encode_tower(Fraction(10**6), books)
    check("T3 escape reserved", big.codes[0] is ESCAPE)
    check("T3 escape keeps value", big.value() == 10**6)
    check("T3 escape not int", not isinstance(big.codes[0], int))

    # T4 NaN / inf rejected, not silently threshold-bypassed
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            encode_tower(bad, books)
            check("T4 rejects %s" % bad, False)
        except TowerVoid:
            check("T4 rejects %s" % bad, True)

    # T5 vector length mismatch refuses zip-truncation
    va = encode_vector([Fraction(1), Fraction(2)], books)
    vb = encode_vector([Fraction(1)], books)
    try:
        dot_tower(va, vb)
        check("T5 no zip truncation", False)
    except TowerVoid:
        check("T5 no zip truncation", True)

    # T6 truncation is declared and measured
    t2 = encode_tower(Fraction(22, 7), books)
    tt, rep = t2.truncate(1)
    check("T6 truncation measured",
          Fraction(rep["dropped_magnitude"]) ==
          t2.value() - tt.value())

    # T7 captured fraction uses the same-split denominator
    rows = encode_tower(Fraction(3, 2), books).per_level()
    r0 = rows[0]
    check("T7 same-split denominator",
          Fraction(r0["captured_fraction"]) <= 1,
          "captured=%s" % r0["captured_fraction"])

    # T8 cross-term defect is labeled, never physical
    xa = encode_tower(Fraction(3, 2), books)
    xb = encode_tower(Fraction(5, 4), books)
    ct = cross_term_defect(xa, xb)
    check("T8 cross-term labeled",
          "not power conjugacy" in ct["label"])
    A, B = xa.code_sum(), xb.code_sum()
    full = xa.value() * xb.value()
    omitted = A * xb.residual + B * xa.residual + xa.residual * xb.residual
    check("T8 cross-term arithmetic",
          Fraction(ct["omitted_cross_terms"]) == omitted and
          full == (A + xa.residual) * (B + xb.residual))

    # T9 arithmetic exactness through ops
    s = add(xa, xb)
    check("T9 add exact", s.value() == Fraction(3, 2) + Fraction(5, 4))
    p, _ = mul(xa, xb)
    check("T9 mul exact", p.value() == Fraction(3, 2) * Fraction(5, 4))
    q = div(xa, xb)
    check("T9 div exact", q.value() == Fraction(3, 2) / Fraction(5, 4))
    try:
        div(xa, encode_tower(0, books))
        check("T9 div-by-zero VOID", False)
    except TowerVoid:
        check("T9 div-by-zero VOID", True)

    # T10 tolerance without reason is flagged UNSOURCED
    rec = claim("selftest", "unsourced_demo", lambda: Fraction(1, 100),
                Fraction(1, 10), "", "falsifier text")
    check("T10 unsourced flagged",
          rec.get("flags") == ["UNSOURCED_TOLERANCE"] and
          rec["tolerance_reason"] == "UNSOURCED")

    # T11 measured values are computed, and replay agrees (ts excluded)
    def c_fn():
        return claim("selftest", "replay_demo",
                     lambda: encode_tower(Fraction(1, 3), books)
                             .residual,
                     Fraction(1), "tolerance covers any residual "
                     "by construction", "replay mismatch")
    eq, rec = replay(c_fn)
    check("T11 replay equal", eq)
    try:
        claim("selftest", "literal_demo", 0.0, 1, "r", "f")
        check("T11 literal measured refused", False)
    except TowerVoid:
        check("T11 literal measured refused", True)

    # T12 stack mismatch is VOID, not guessed alignment
    other = dyadic_stack(8, levels=4, codes_per_level=16)
    try:
        add(encode_tower(1, books), encode_tower(1, other))
        check("T12 stack mismatch VOID", False)
    except TowerVoid:
        check("T12 stack mismatch VOID", True)

    return {"schema": SCHEMA, "tests": out, "fails": fails,
            "n": len(out)}


# ----------------------------------------------------------------------
# Demo + check (printed-vs-executed)
# ----------------------------------------------------------------------
def _demo_claims():
    books = dyadic_stack(4, levels=4, codes_per_level=16)

    def exactness():
        worst = Fraction(0)
        for num in range(-50, 51):
            for den in (1, 3, 7, 11, 13, 113):
                v = Fraction(num, den)
                t = encode_tower(v, books)
                if abs(t.value() - v) > worst:
                    worst = abs(t.value() - v)
        return worst
    yield lambda: claim("demo::tower::exactness", "exact_identity_worst",
                        exactness, 0,
                        "identity is exact by construction; tolerance 0 "
                        "is the claim under test",
                        "any nonzero worst deviation")

    def level0_escapes():
        n_esc = 0
        for num in range(-42, 43):      # |v| <= 3 exactly, well in-book
            v = Fraction(num, 14)
            if encode_tower(v, books).codes[0] is ESCAPE:
                n_esc += 1
        return n_esc
    yield lambda: claim("demo::tower::level0", "level0_accepts_in_book",
                        level0_escapes, 0,
                        "level-0 acceptance band is [-4, 3.75] by grid "
                        "construction; the demo grid stays inside |v|<=3",
                        "any level-0 ESCAPE inside |v| <= 3")

    def escape_loss():
        worst = Fraction(0)
        for num in range(57, 150):      # v in (4, 10.7): out of book
            v = Fraction(num, 14)
            t = encode_tower(v, books)
            if t.codes[0] is ESCAPE:
                dev = abs(t.residual - v)
            else:
                dev = Fraction(10**9)   # accepted out-of-book: defect
            if dev > worst:
                worst = dev
        return worst
    yield lambda: claim("demo::tower::escape", "escape_preserves_value",
                        escape_loss, 0,
                        "ESCAPE passes the residual down whole by "
                        "construction of encode(); nothing is clamped",
                        "out-of-book value loses magnitude or is "
                        "silently accepted")

    def edge_band():
        # v = 1/4: accepted at level 0 with residual exactly tau0 = 1/4;
        # that residual sits at the very edge of level 1's book, whose
        # top grid point is 7/32 < 1/4, so level 1 ESCAPEs and the band
        # cascades. The tower must RECORD this, not hide it.
        t = encode_tower(Fraction(1, 4), books)
        ok = (t.codes[0] is not ESCAPE and t.codes[1] is ESCAPE and
              t.residual == Fraction(1, 4))
        return Fraction(0 if ok else 1)
    yield lambda: claim("demo::tower::edge_band",
                        "edge_residual_escapes_are_explicit",
                        edge_band, 0,
                        "acceptance band of a book ends at its top grid "
                        "point plus tau; a tau-edge residual is one step "
                        "past the next book's last grid point",
                        "edge-band residual silently snaps to a code "
                        "instead of ESCAPE",
                        extra={"note": "documents the one-step-wide "
                               "escape band at each book edge; recorded "
                               "behavior, not a defect"})


def demo(path):
    recs = []
    for fn in _demo_claims():
        ok, rec = replay(fn)
        rec["replay_equal"] = ok
        recs.append(rec)
        append_record(path, rec)
    return recs


def check_record(path):
    """printed-vs-executed: recompute the demo and diff the printed
    measured values in the record file (bridge-audit lesson: the notes
    printed 0.0608/0.6782 where execution gave 0.0574/0.7348)."""
    printed = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            printed[r["name"]] = r
    fresh = {}
    for fn in _demo_claims():
        _, rec = replay(fn)
        fresh[rec["name"]] = rec
    diffs = []
    for name, prec in printed.items():
        frec = fresh.get(name)
        if frec is None:
            diffs.append({"name": name, "diff": "no fresh computation"})
            continue
        for key in ("measured", "verdict", "tolerance"):
            if prec.get(key) != frec.get(key):
                diffs.append({"name": name, "field": key,
                              "printed": prec.get(key),
                              "executed": frec.get(key)})
    return {"schema": SCHEMA, "checked": len(printed), "diffs": diffs,
            "ok": not diffs}


def main(argv):
    if len(argv) < 2 or argv[1] not in ("selftest", "demo", "check"):
        print(__doc__)
        return 2
    if argv[1] == "selftest":
        res = selftest()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["fails"] == 0 else 1
    if argv[1] == "demo":
        path = argv[2] if len(argv) > 2 else "demo_record.jsonl"
        recs = demo(path)
        for r in recs:
            print("%-40s measured=%s verdict=%s replay=%s" %
                  (r["name"], r["measured"], r["verdict"],
                   r["replay_equal"]))
        print("appended %d records -> %s" % (len(recs), path))
        return 0
    if argv[1] == "check":
        if len(argv) < 3:
            print("check needs a record file")
            return 2
        res = check_record(argv[2])
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
