"""
Coupling from Physics: the interior optimum, derived rather than asserted

A common form for a coupling-quality term f(C) is a Gaussian bump around a chosen
optimum C*: "too weak = fragmented, too strong = rigid." That is an assertion
dressed as a measurement — move C* and the optimum moves with it.

Synchronization theory derives the same shape from the dynamics, with no free
aesthetic parameter. For N units with identical dynamics coupled through a
network, the Master Stability Function (Pecora & Carroll 1998, Phys. Rev. Lett.
80:2109) reduces stability of the synchronous state to a single scalar condition.
Writing the network's Laplacian eigenvalues as 0 = lambda_1 <= lambda_2 <= ... <=
lambda_N and the coupling strength as sigma, the synchronous state is stable
exactly when every scaled eigenvalue lands inside the MSF's negative region:

    nu_1 < sigma * lambda_i < nu_2        for i = 2 .. N

Huang, Chen, Lai & Pecora (2009, Phys. Rev. E 80:036204) show that only three
classes of behaviour are possible for any pair of node dynamics and coupling
function:

    Class I   - the MSF never goes negative. No coupling strength synchronizes.
                There is no optimum because there is no stable region at all.
    Class II  - one crossing, at nu_1. Stable for every sigma above a threshold.
                Physics gives a threshold, not an optimum: more coupling is
                never worse.
    Class III - two crossings, nu_1 and nu_2. Stable only inside a bounded
                window. Too little coupling and the slowest mode (lambda_2)
                never locks; too much and the fastest mode (lambda_N) is driven
                unstable.

**Class III is where an interior optimum actually comes from, and it is a
property of the node dynamics — not a universal law.** Asserting an interior
optimum for a Class II system would be inventing a penalty physics does not
impose. This module reports the class rather than assuming one.

TWO CONSEQUENCES WORTH THE SPACE
--------------------------------
1. **Fragmentation is structural, not a tuning error.** A disconnected network
   has lambda_2 = 0, so no sigma satisfies the lower bound. "Too weak =
   fragmented" is not a statement about turning coupling down far enough; it is
   a statement that the network has no path.

2. **Some networks cannot be fixed by tuning at all.** Both bounds are
   satisfiable simultaneously only when

       lambda_N / lambda_2  <  nu_2 / nu_1

   the eigenratio criterion (Barahona & Pecora 2002, Phys. Rev. Lett. 89:054101).
   The left side is pure topology; the right side is pure dynamics. If the
   network's eigenratio exceeds the dynamics' window, *no* coupling strength
   works and the only remedy is changing the network. Lower eigenratio = more
   synchronizable.

MEASUREMENT, NOT CONTROL
------------------------
Reporting that a network sits outside its synchronization window is not an
instruction to retune it. Plenty of real systems should not be synchronized at
all, and a high eigenratio is sometimes exactly what keeps a failure local.

SCOPE
-----
This is Tier 1 (numpy) per the ecosystem's dependency discipline. It is an
instrument, not part of the repair loop: nothing here writes to manifolds/,
simulation/, or repair/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isinf, log, sqrt
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class MSFWindow:
    """The negative region of a Master Stability Function.

    This is a property of the *node dynamics and coupling function*, not of the
    network. It has to be measured for the system in question — computed as the
    maximum transverse Lyapunov exponent versus the scaled coupling parameter —
    and is supplied here the way the calibration adapters take measured data.

    Args:
        nu_lower: Lower zero crossing. The MSF is positive below it. Ignored
                  when never_negative is True.
        nu_upper: Upper zero crossing, or math.inf for Class II dynamics where
                  the MSF stays negative once it crosses.
        never_negative: True for Class I dynamics, whose MSF has no negative
                        region at all. Construct these with MSFWindow.class_i().
        source: Where the window was measured.
        system: What node dynamics it describes.
    """

    nu_lower: float = inf
    nu_upper: float = inf
    never_negative: bool = False
    source: str = ""
    system: str = ""

    def __post_init__(self) -> None:
        if self.never_negative:
            return
        if self.nu_lower <= 0 or isinf(self.nu_lower):
            raise ValueError(
                "nu_lower must be positive and finite; a non-positive lower "
                "crossing would make the scaled coupling condition vacuous"
            )
        if self.nu_upper <= self.nu_lower:
            raise ValueError("nu_upper must exceed nu_lower")

    @classmethod
    def class_i(cls, source: str = "", system: str = "") -> "MSFWindow":
        """Class I dynamics: the MSF never goes negative, so nothing synchronizes.

        Represented explicitly rather than left unconstructible, so that "these
        dynamics admit no stable coupling" is a reading the caller can receive
        instead of an input they cannot express.
        """
        return cls(never_negative=True, source=source, system=system)

    @property
    def msf_class(self) -> str:
        """Class I (no negative region), II (threshold only), or III (window)."""
        if self.never_negative:
            return "I"
        return "II" if isinf(self.nu_upper) else "III"

    @property
    def width_ratio(self) -> float:
        """nu_2 / nu_1 — the dynamics' tolerance for spectral spread.

        This is the quantity the network's eigenratio must come in under.
        Infinite for Class II: those dynamics tolerate any spread. Zero for
        Class I: no spread is tolerable because nothing is stable.
        """
        if self.never_negative:
            return 0.0
        return inf if isinf(self.nu_upper) else self.nu_upper / self.nu_lower

    @property
    def has_interior_optimum(self) -> bool:
        """True only for Class III. Class II has a threshold, not a peak."""
        return self.msf_class == "III"


@dataclass
class SpectrumReading:
    """Laplacian spectrum of a coupling network."""

    eigenvalues: List[float]
    lambda_2: float                  # algebraic connectivity (Fiedler value)
    lambda_n: float                  # largest Laplacian eigenvalue
    eigenratio: float                # lambda_N / lambda_2, inf if disconnected
    connected: bool = True
    n_components: int = 1
    zero_tolerance: float = 0.0      # threshold used to count structural zeros
    warnings: List[str] = field(default_factory=list)


@dataclass
class CouplingReading:
    """Where a coupling strength sits relative to the stable window."""

    coherence: float                      # f(C) in [0, 1]
    synchronizable: bool                  # does any sigma work at all?
    sigma_min: Optional[float] = None     # nu_1 / lambda_2
    sigma_max: Optional[float] = None     # nu_2 / lambda_N
    sigma_optimal: Optional[float] = None  # geometric centre in log-coupling
    margin: Optional[float] = None        # sqrt(width_ratio / eigenratio)
    msf_class: str = ""
    regime: str = ""                      # FRAGMENTED | STABLE | RIGID | ...
    source: str = ""
    notes: List[str] = field(default_factory=list)


def laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Combinatorial graph Laplacian L = D - A.

    Args:
        adjacency: Symmetric, non-negative weighted adjacency matrix. Diagonal
                   entries are ignored (no self-loops).

    Returns:
        The Laplacian matrix.
    """
    A = np.array(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if A.shape[0] < 2:
        raise ValueError("a coupling network needs at least two units")
    if not np.allclose(A, A.T):
        raise ValueError(
            "adjacency must be symmetric; the eigenratio criterion is derived "
            "for undirected coupling, and a directed network needs the "
            "complex-plane form of the stability condition instead"
        )
    if np.any(A < 0):
        raise ValueError("adjacency weights must be non-negative")
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def zero_tolerance(n: int, lambda_max: float) -> float:
    """Numerical-rank tolerance for counting a Laplacian's structural zeros.

    Every Laplacian has at least one exact zero eigenvalue; floating point puts
    it near, not at, zero. A fixed absolute cutoff is wrong for large sparse
    networks, whose genuine algebraic connectivity can be legitimately tiny — a
    path graph on n nodes has lambda_2 ~ pi^2 / n^2, which falls below 1e-9
    around n = 1e5 while the graph is perfectly connected. Scaling the tolerance
    by n * eps * lambda_max (the convention numpy uses for matrix rank) keeps
    the test numerical rather than physical.
    """
    return max(n * np.finfo(float).eps * max(lambda_max, 1.0), 1e-300)


def spectrum(adjacency: np.ndarray, tolerance: Optional[float] = None) -> SpectrumReading:
    """Laplacian spectrum and eigenratio of a coupling network.

    The eigenratio lambda_N / lambda_2 is the topology half of the
    synchronizability condition: the smaller it is, the wider the range of
    coupling strengths that keep every mode stable.

    Args:
        adjacency: Symmetric non-negative adjacency matrix.
        tolerance: Override for the structural-zero threshold. Defaults to a
                   scale-aware numerical-rank tolerance; see zero_tolerance().

    Returns:
        SpectrumReading. A disconnected network has lambda_2 = 0 and an infinite
        eigenratio — no coupling strength synchronizes it.
    """
    L = laplacian(adjacency)
    n = L.shape[0]
    values = sorted(float(v) for v in np.linalg.eigvalsh(L))
    tol = zero_tolerance(n, values[-1]) if tolerance is None else tolerance
    values = [0.0 if abs(v) < tol else v for v in values]

    n_zero = sum(1 for v in values if v == 0.0)
    lambda_2 = values[1]
    lambda_n = values[-1]
    connected = n_zero == 1
    warnings: List[str] = []

    if not connected:
        ratio = inf
        warnings.append(
            f"network splits into {n_zero} components (lambda_2 = 0). No "
            "coupling strength synchronizes a disconnected network — this is "
            "fragmentation as a structural fact, not a tuning problem."
        )
    else:
        ratio = lambda_n / lambda_2

    return SpectrumReading(
        eigenvalues=values,
        lambda_2=lambda_2,
        lambda_n=lambda_n,
        eigenratio=ratio,
        connected=connected,
        n_components=n_zero,
        zero_tolerance=tol,
        warnings=warnings,
    )


def synchronizable(spec: SpectrumReading, window: MSFWindow) -> bool:
    """Does *any* coupling strength stabilize this network?

    True exactly when the topology's eigenratio comes in under the dynamics'
    window ratio (Barahona & Pecora 2002). Note the two sides are independent:
    topology on the left, dynamics on the right. Always False for Class I
    dynamics, whose window ratio is zero.
    """
    if window.never_negative or not spec.connected:
        return False
    return spec.eigenratio < window.width_ratio


def optimal_coupling(spec: SpectrumReading, window: MSFWindow) -> Optional[float]:
    """Coupling strength furthest from both stability boundaries.

    Derived, not cited: maximizing the smaller of the two logarithmic margins,
    ln(sigma*lambda_2/nu_1) and ln(nu_2/(sigma*lambda_N)), equalizes them, which
    gives

        sigma* = sqrt(nu_1 * nu_2 / (lambda_2 * lambda_N))

    the geometric centre of the window in coupling. Log-margins are the natural
    measure here because both boundaries are multiplicative in sigma. Returns
    None for Class II dynamics, which have no upper boundary to be far from, and
    for Class I, which has no stable region at all.
    """
    if not synchronizable(spec, window) or not window.has_interior_optimum:
        return None
    return sqrt(window.nu_lower * window.nu_upper / (spec.lambda_2 * spec.lambda_n))


def coupling_coherence(
    sigma: float,
    adjacency: np.ndarray,
    window: MSFWindow,
) -> CouplingReading:
    """f(C) for a coupling strength on a network, from stability theory.

    For Class III dynamics this has a genuine interior maximum: it is 1 at the
    geometric centre of the stable window and falls linearly in log coupling to
    0 at either boundary. Below the window the slowest mode never locks
    (FRAGMENTED); above it the fastest mode goes unstable (RIGID).

    For Class II dynamics it is binary. Physics gives a threshold and no
    gradient above it, and inventing a smooth ramp there would be manufacturing
    a penalty the dynamics do not impose.

    Args:
        sigma: Coupling strength.
        adjacency: Symmetric non-negative adjacency matrix.
        window: Measured MSF negative region for the node dynamics.

    Returns:
        CouplingReading with f(C), the stable window, and the regime.
    """
    if sigma < 0:
        raise ValueError("coupling strength cannot be negative")

    source = (
        "Pecora & Carroll 1998, Phys. Rev. Lett. 80:2109; "
        "Barahona & Pecora 2002, Phys. Rev. Lett. 89:054101; "
        "Huang et al. 2009, Phys. Rev. E 80:036204"
    )

    if window.never_negative:
        return CouplingReading(
            coherence=0.0, synchronizable=False, msf_class="I",
            regime="NO_STABLE_REGION_CLASS_I", source=source,
            notes=["these node dynamics have no stable coupling region at any "
                   "strength on any network; the question of an optimum does "
                   "not arise"],
        )

    spec = spectrum(adjacency)
    notes = list(spec.warnings)

    if not spec.connected:
        notes.append(
            "f(C) = 0 because the network has no path between all units, not "
            "because the coupling strength is mistuned"
        )
        notes.append(
            "a partition is not necessarily an ending: surviving components can "
            "still be individually viable, and whether reconnecting them is "
            "worth doing is not a question this measurement answers"
        )
        return CouplingReading(
            coherence=0.0, synchronizable=False, msf_class=window.msf_class,
            regime="FRAGMENTED_STRUCTURALLY", source=source, notes=notes,
        )

    sigma_min = window.nu_lower / spec.lambda_2
    sigma_max = inf if isinf(window.nu_upper) else window.nu_upper / spec.lambda_n

    if not synchronizable(spec, window):
        notes.append(
            f"eigenratio {spec.eigenratio:.3f} exceeds the dynamics' window "
            f"ratio {window.width_ratio:.3f} — no coupling strength stabilizes "
            "this network. The remedy, if one is wanted, is a different "
            "network, not a different coupling strength."
        )
        return CouplingReading(
            coherence=0.0, synchronizable=False,
            sigma_min=sigma_min, sigma_max=sigma_max,
            msf_class=window.msf_class, regime="NO_STABLE_WINDOW",
            source=source, notes=notes,
        )

    margin = inf if isinf(window.width_ratio) else sqrt(window.width_ratio / spec.eigenratio)
    sigma_opt = optimal_coupling(spec, window)

    if sigma <= sigma_min:
        notes.append(
            f"sigma * lambda_2 = {sigma * spec.lambda_2:.4f} is at or below "
            f"nu_1 = {window.nu_lower:.4f} — the slowest mode never locks. "
            "This is the 'too weak = fragmented' branch."
        )
        regime, coherence = "FRAGMENTED", 0.0
    elif sigma >= sigma_max:
        notes.append(
            f"sigma * lambda_N = {sigma * spec.lambda_n:.4f} is at or above "
            f"nu_2 = {window.nu_upper:.4f} — the fastest mode is driven "
            "unstable. This is the 'too strong = rigid' branch, and it exists "
            "only because these dynamics are Class III."
        )
        regime, coherence = "RIGID", 0.0
    else:
        regime = "STABLE"
        if window.has_interior_optimum:
            lower_margin = log(sigma * spec.lambda_2 / window.nu_lower)
            upper_margin = log(window.nu_upper / (sigma * spec.lambda_n))
            peak = log(margin)
            coherence = min(1.0, max(0.0, min(lower_margin, upper_margin) / peak))
            notes.append(
                f"stable window sigma in ({sigma_min:.4f}, {sigma_max:.4f}), "
                f"widest margin at sigma* = {sigma_opt:.4f}"
            )
        else:
            coherence = 1.0
            notes.append(
                f"Class II dynamics: stable for every sigma above {sigma_min:.4f}. "
                "Physics gives a threshold here, not an optimum, so f(C) is "
                "reported as binary rather than as an invented gradient."
            )

    if window.has_interior_optimum:
        notes.append(
            f"safety margin sqrt(window ratio / eigenratio) = {margin:.3f}; "
            "values near 1 mean the network barely fits inside the window at "
            "any coupling strength"
        )

    return CouplingReading(
        coherence=coherence,
        synchronizable=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_optimal=sigma_opt,
        margin=margin,
        msf_class=window.msf_class,
        regime=regime,
        source=source,
        notes=notes,
    )


def format_coupling(r: CouplingReading) -> str:
    """Human-readable rendering of a CouplingReading."""

    def num(v: Optional[float], fmt: str = "{:.4f}") -> str:
        if v is None:
            return "n/a"
        return "inf" if isinf(v) else fmt.format(v)

    lines = [
        "=" * 70,
        f"COUPLING: {r.regime}    f(C) = {r.coherence:.4f}    "
        f"MSF class {r.msf_class or '?'}",
        "=" * 70,
        f"  stable window   = ({num(r.sigma_min)}, {num(r.sigma_max)})",
        f"  optimal sigma   = {num(r.sigma_optimal)}",
        f"  safety margin   = {num(r.margin, '{:.3f}')}",
    ]
    if r.notes:
        lines.append("")
        lines.append("NOTES:")
        for n in r.notes:
            lines.append(f"  - {n}")
    if r.source:
        lines.append("")
        lines.append(f"  source: {r.source}")
    lines.extend([
        "",
        "Outside the window is a reading, not a fault. Some systems should not "
        "synchronize.",
        "=" * 70,
    ])
    return "\n".join(lines)


# Demo
if __name__ == "__main__":
    # A Class III window: bounded, so there is a real interior optimum.
    class_iii = MSFWindow(
        nu_lower=0.2, nu_upper=4.0,
        source="illustrative window; measure the MSF for your own dynamics",
        system="Class III node dynamics",
    )

    ring = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=float)

    spec = spectrum(ring)
    print(f"5-ring: lambda_2 = {spec.lambda_2:.4f}, lambda_N = "
          f"{spec.lambda_n:.4f}, eigenratio = {spec.eigenratio:.4f}")
    print(f"window ratio = {class_iii.width_ratio:.2f} -> "
          f"synchronizable: {synchronizable(spec, class_iii)}")
    print()

    sigma_opt = optimal_coupling(spec, class_iii)
    for label, sigma in [
        ("far too weak", sigma_opt / 8),
        ("just inside", sigma_opt / 2.2),
        ("optimal", sigma_opt),
        ("too strong", sigma_opt * 4),
    ]:
        reading = coupling_coherence(sigma, ring, class_iii)
        print(f"  sigma = {sigma:7.4f}  ({label:13s}) -> "
              f"f(C) = {reading.coherence:.4f}  {reading.regime}")
    print()

    # A star network: high eigenratio, hard to synchronize.
    star = np.zeros((6, 6))
    star[0, 1:] = 1.0
    star[1:, 0] = 1.0
    star_spec = spectrum(star)
    print(f"6-star: eigenratio = {star_spec.eigenratio:.4f} vs window ratio "
          f"{class_iii.width_ratio:.2f}")
    print(format_coupling(coupling_coherence(1.0, star, class_iii)))
    print()

    # A disconnected network: fragmentation as structure.
    split = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)
    print(format_coupling(coupling_coherence(1.0, split, class_iii)))
