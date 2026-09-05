#!/usr/bin/env python3
"""Spectrum instruments: symmetric Jacobi eigensolver, RankMe, alpha-ReQ, and the
Spectrum record that refuses to exist without an AXIS label.

Reference instrument (definitions only, not results): Li et al., "Tracing the
Representation Geometry of Language Models from Pretraining to Post-training",
arXiv 2509.23024. RankMe = exp(entropy of the normalised singular values);
alpha-ReQ = power-law decay exponent of the covariance eigenspectrum.

Every spectrum emitted anywhere in this folder is built through `Spectrum(...)`,
whose constructor raises on a missing or empty axis. That is selftest S5: a
spectrum whose sample axis was silently substituted is the failure mode this
whole test exists to avoid, so it is a hard error and not a warning.

Stdlib only. Pure Python lists, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ------------------------------------------------------------------ linear algebra


def transpose(m):
    return [list(r) for r in zip(*m)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(ra, cb)) for cb in bt] for ra in a]


def gram(rows):
    """rows: list of k vectors (same length). Returns k x k matrix of dot products."""
    return [[sum(x * y for x, y in zip(ri, rj)) for rj in rows] for ri in rows]


def second_moment(rows, centered: bool):
    """(1/k) * X^T X over k sample rows, optionally after subtracting the sample mean.

    Returns the p x p matrix where p is the vector length. Use this when p is small
    (representation channel, d-dimensional). For p large and k small, use
    `gram_spectrum` instead: the nonzero eigenvalues of X^T X and X X^T coincide.
    """
    k = len(rows)
    if k == 0:
        raise ValueError("second_moment of zero samples")
    p = len(rows[0])
    if centered:
        mu = [sum(r[j] for r in rows) / k for j in range(p)]
        rows = [[r[j] - mu[j] for j in range(p)] for r in rows]
    m = [[0.0] * p for _ in range(p)]
    for r in rows:
        for i in range(p):
            ri = r[i]
            if ri == 0.0:
                continue
            mi = m[i]
            for j in range(p):
                mi[j] += ri * r[j]
    return [[v / k for v in row] for row in m]


def jacobi_eigenvalues(a, tol: float = 1e-12, max_sweeps: int = 100):
    """Eigenvalues of a real symmetric matrix by cyclic Jacobi rotation.

    Returns eigenvalues sorted descending. Raises on a non-square or asymmetric input
    (asymmetry beyond 1e-9 relative), because Jacobi silently returns garbage on one.
    """
    n = len(a)
    if any(len(r) != n for r in a):
        raise ValueError("jacobi: matrix is not square")
    scale = max(1.0, max(abs(v) for r in a for v in r))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-9 * scale:
                raise ValueError("jacobi: matrix is not symmetric")
    a = [list(r) for r in a]
    for _ in range(max_sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < tol * tol * scale * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                app, aqq = a[p][p], a[q][q]
                theta = (aqq - app) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted((a[i][i] for i in range(n)), reverse=True)


# ------------------------------------------------------------------ the two metrics


def rankme(eigenvalues, eps: float = 1e-12) -> float:
    """exp(entropy of normalised singular values). sigma_i = sqrt(max(lambda_i, 0)).

    Isotropic covariance in d dimensions -> d. Rank-1 covariance -> 1.
    """
    sig = [math.sqrt(max(v, 0.0)) for v in eigenvalues]
    z = sum(sig)
    if z <= eps:
        return 0.0
    h = 0.0
    for s in sig:
        p = s / z
        if p > eps:
            h -= p * math.log(p)
    return math.exp(h)


def alpha_req(eigenvalues, rel_floor: float = 1e-8):
    """Power-law decay exponent: fit log(lambda_i) = -alpha * log(i) + c over the
    eigenvalues above rel_floor * lambda_max, rank i starting at 1. Returns alpha, or
    None when fewer than two eigenvalues survive the floor (no slope is a slope).
    """
    ev = sorted((v for v in eigenvalues if v > 0.0), reverse=True)
    if not ev:
        return None
    floor = rel_floor * ev[0]
    ev = [v for v in ev if v > floor]
    if len(ev) < 2:
        return None
    xs = [math.log(i + 1) for i in range(len(ev))]
    ys = [math.log(v) for v in ev]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return -(sxy / sxx)


# ------------------------------------------------------------------ the record

AXES = ("REP", "A1/TIME", "A2/UNIT-L1", "A2/UNIT-L2", "A3/SEED-raw", "A3/SEED-aligned")


@dataclass
class Spectrum:
    """A spectrum with the axis it was taken over. The axis is not optional."""

    axis: str
    eigenvalues: list = field(default_factory=list)
    n_samples: int = 0
    centered: bool = False
    note: str = ""

    def __post_init__(self):
        if not isinstance(self.axis, str) or not self.axis.strip():
            raise ValueError(
                "Spectrum without a declared AXIS. Refused. Parameter space has no sample "
                "dimension; the axis that supplied one must be named on every spectrum."
            )
        if self.axis not in AXES:
            raise ValueError(f"Spectrum axis {self.axis!r} is not one of the declared axes {AXES}")

    @property
    def rankme(self) -> float:
        return rankme(self.eigenvalues)

    @property
    def alpha(self):
        return alpha_req(self.eigenvalues)

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "n_samples": self.n_samples,
            "centered": self.centered,
            "rankme": self.rankme,
            "alpha_req": self.alpha,
            "eigenvalues": self.eigenvalues,
            "note": self.note,
        }


def spectrum_of_rows(axis: str, rows, centered: bool, note: str = "") -> Spectrum:
    """Spectrum over a set of sample rows. Chooses the cheaper of X^T X / X X^T.

    The nonzero eigenvalues of (1/k) X^T X and (1/k) X X^T are identical, so when the
    vectors are long and the samples few (every drift axis) the k x k Gram is solved.
    Centering across samples is applied before either route when requested.
    """
    k = len(rows)
    if k == 0:
        raise ValueError("spectrum_of_rows: no samples")
    p = len(rows[0])
    if centered:
        mu = [sum(r[j] for r in rows) / k for j in range(p)]
        rows = [[r[j] - mu[j] for j in range(p)] for r in rows]
    if p <= k:
        m = second_moment(rows, centered=False)
        ev = jacobi_eigenvalues(m)
    else:
        g = gram(rows)
        ev = [v / k for v in jacobi_eigenvalues(g)]
    ev = [v if v > 0.0 else 0.0 for v in ev]
    return Spectrum(axis=axis, eigenvalues=ev, n_samples=k, centered=centered, note=note)
