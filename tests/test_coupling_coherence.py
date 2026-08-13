"""Tests for research_interface.coupling_coherence.

These pin the mathematical contract, not just the plumbing: the interior optimum
is a derived quantity, so it is checked against the identity it is derived from
rather than against a recorded number.
"""

from math import isclose, isinf, log, sqrt

import numpy as np
import pytest

from research_interface.coupling_coherence import (
    MSFWindow,
    coupling_coherence,
    format_coupling,
    laplacian,
    optimal_coupling,
    spectrum,
    synchronizable,
    zero_tolerance,
)

CLASS_III = MSFWindow(nu_lower=0.2, nu_upper=4.0, system="test class III")
CLASS_II = MSFWindow(nu_lower=0.2, system="test class II")
CLASS_I = MSFWindow.class_i(system="test class I")

RING5 = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
], dtype=float)

SPLIT4 = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=float)


def star(n):
    a = np.zeros((n, n))
    a[0, 1:] = 1.0
    a[1:, 0] = 1.0
    return a


# --- classification --------------------------------------------------------

def test_msf_classes():
    assert CLASS_III.msf_class == "III"
    assert CLASS_II.msf_class == "II"
    assert CLASS_I.msf_class == "I"


def test_only_class_iii_has_interior_optimum():
    """The whole point: an interior optimum is a property of the dynamics."""
    assert CLASS_III.has_interior_optimum
    assert not CLASS_II.has_interior_optimum
    assert not CLASS_I.has_interior_optimum


def test_width_ratio():
    assert CLASS_III.width_ratio == 20.0
    assert isinf(CLASS_II.width_ratio)
    assert CLASS_I.width_ratio == 0.0


def test_invalid_windows_rejected():
    with pytest.raises(ValueError):
        MSFWindow(nu_lower=0.0, nu_upper=1.0)
    with pytest.raises(ValueError):
        MSFWindow(nu_lower=-1.0, nu_upper=1.0)
    with pytest.raises(ValueError):
        MSFWindow(nu_lower=2.0, nu_upper=1.0)


# --- spectrum --------------------------------------------------------------

def test_laplacian_rows_sum_to_zero():
    L = laplacian(RING5)
    assert np.allclose(L.sum(axis=1), 0.0)


def test_laplacian_validation():
    with pytest.raises(ValueError):
        laplacian(np.array([[0.0]]))
    with pytest.raises(ValueError):
        laplacian(np.array([[0, 1], [0, 0]], dtype=float))       # asymmetric
    with pytest.raises(ValueError):
        laplacian(np.array([[0, -1], [-1, 0]], dtype=float))     # negative weight


def test_ring_spectrum_matches_closed_form():
    """5-cycle Laplacian eigenvalues are 2 - 2cos(2*pi*k/5)."""
    spec = spectrum(RING5)
    expected = sorted(2 - 2 * np.cos(2 * np.pi * k / 5) for k in range(5))
    assert np.allclose(spec.eigenvalues, expected, atol=1e-9)
    assert spec.connected and spec.n_components == 1


def test_disconnected_network_has_infinite_eigenratio():
    spec = spectrum(SPLIT4)
    assert not spec.connected
    assert spec.n_components == 2
    assert spec.lambda_2 == 0.0
    assert isinf(spec.eigenratio)


def test_star_eigenratio_equals_n():
    """An n-star has lambda_2 = 1 and lambda_N = n, so the eigenratio is n."""
    spec = spectrum(star(6))
    assert isclose(spec.eigenratio, 6.0, rel_tol=1e-9)


def test_large_sparse_path_not_misread_as_disconnected():
    """A fixed 1e-9 cutoff would call this disconnected; it is connected.

    A path graph on n nodes has lambda_2 = 2(1 - cos(pi/n)), which is genuinely
    tiny for large n. The tolerance must be numerical, not physical.
    """
    n = 400
    a = np.zeros((n, n))
    idx = np.arange(n - 1)
    a[idx, idx + 1] = 1.0
    a[idx + 1, idx] = 1.0
    spec = spectrum(a)
    assert spec.connected
    assert spec.lambda_2 > 0.0
    assert spec.zero_tolerance < spec.lambda_2


def test_zero_tolerance_scales_with_size():
    assert zero_tolerance(10_000, 4.0) > zero_tolerance(10, 4.0)


# --- the derived optimum ---------------------------------------------------

def test_optimal_coupling_equalizes_log_margins():
    """sigma* is derived by equalizing the two log-margins; check the identity."""
    spec = spectrum(RING5)
    s = optimal_coupling(spec, CLASS_III)
    lower = log(s * spec.lambda_2 / CLASS_III.nu_lower)
    upper = log(CLASS_III.nu_upper / (s * spec.lambda_n))
    assert isclose(lower, upper, rel_tol=1e-12)


def test_margin_equals_exp_of_peak_log_margin():
    """margin = sqrt(width_ratio / eigenratio) must be exp(max log-margin)."""
    spec = spectrum(RING5)
    s = optimal_coupling(spec, CLASS_III)
    peak = log(s * spec.lambda_2 / CLASS_III.nu_lower)
    r = coupling_coherence(s, RING5, CLASS_III)
    assert isclose(r.margin, np.exp(peak), rel_tol=1e-12)
    assert isclose(r.margin, sqrt(CLASS_III.width_ratio / spec.eigenratio), rel_tol=1e-12)


def test_coherence_peaks_at_one_at_sigma_star():
    spec = spectrum(RING5)
    s = optimal_coupling(spec, CLASS_III)
    assert isclose(coupling_coherence(s, RING5, CLASS_III).coherence, 1.0, rel_tol=1e-9)


def test_coherence_is_zero_at_both_boundaries_and_bounded_between():
    spec = spectrum(RING5)
    lo = CLASS_III.nu_lower / spec.lambda_2
    hi = CLASS_III.nu_upper / spec.lambda_n
    assert coupling_coherence(lo, RING5, CLASS_III).coherence == 0.0
    assert coupling_coherence(hi, RING5, CLASS_III).coherence == 0.0
    for s in np.linspace(lo * 1.001, hi * 0.999, 25):
        assert 0.0 <= coupling_coherence(s, RING5, CLASS_III).coherence <= 1.0


def test_regimes_on_either_side():
    spec = spectrum(RING5)
    s = optimal_coupling(spec, CLASS_III)
    assert coupling_coherence(s / 8, RING5, CLASS_III).regime == "FRAGMENTED"
    assert coupling_coherence(s, RING5, CLASS_III).regime == "STABLE"
    assert coupling_coherence(s * 8, RING5, CLASS_III).regime == "RIGID"


# --- class II and I --------------------------------------------------------

def test_class_ii_is_binary_with_no_optimum():
    """More coupling is never worse, so no gradient is invented above threshold."""
    spec = spectrum(RING5)
    assert optimal_coupling(spec, CLASS_II) is None
    thresh = CLASS_II.nu_lower / spec.lambda_2
    assert coupling_coherence(thresh * 0.5, RING5, CLASS_II).regime == "FRAGMENTED"
    for s in (thresh * 1.5, thresh * 100, thresh * 10_000):
        r = coupling_coherence(s, RING5, CLASS_II)
        assert r.coherence == 1.0 and r.regime == "STABLE"


def test_class_i_never_synchronizes():
    r = coupling_coherence(1.0, RING5, CLASS_I)
    assert r.coherence == 0.0
    assert not r.synchronizable
    assert r.regime == "NO_STABLE_REGION_CLASS_I"
    assert not synchronizable(spectrum(RING5), CLASS_I)


# --- the eigenratio criterion ----------------------------------------------

def test_eigenratio_criterion_decides_synchronizability():
    """Topology on the left, dynamics on the right, and they are independent."""
    narrow = MSFWindow(nu_lower=1.0, nu_upper=3.0)   # width ratio 3
    spec = spectrum(star(6))                          # eigenratio 6
    assert spec.eigenratio > narrow.width_ratio
    assert not synchronizable(spec, narrow)
    r = coupling_coherence(1.0, star(6), narrow)
    assert r.regime == "NO_STABLE_WINDOW" and r.coherence == 0.0
    # widening the dynamics' window - not touching the network - fixes it
    wide = MSFWindow(nu_lower=1.0, nu_upper=30.0)     # width ratio 30
    assert synchronizable(spec, wide)


def test_denser_network_is_more_synchronizable():
    """Complete graph has eigenratio 1, the best possible."""
    n = 6
    complete = np.ones((n, n)) - np.eye(n)
    assert isclose(spectrum(complete).eigenratio, 1.0, rel_tol=1e-9)
    assert spectrum(complete).eigenratio < spectrum(star(n)).eigenratio


# --- fragmentation ---------------------------------------------------------

def test_disconnected_is_structural_not_a_tuning_problem():
    """No sigma rescues a partition, so every strength reads the same."""
    for s in (0.0, 0.01, 1.0, 1e6):
        r = coupling_coherence(s, SPLIT4, CLASS_III)
        assert r.coherence == 0.0
        assert not r.synchronizable
        assert r.regime == "FRAGMENTED_STRUCTURALLY"
        assert any("structural fact" in n for n in r.notes)


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        coupling_coherence(-1.0, RING5, CLASS_III)


def test_format_coupling_renders_all_regimes():
    spec = spectrum(RING5)
    s = optimal_coupling(spec, CLASS_III)
    for reading in (coupling_coherence(s, RING5, CLASS_III),
                    coupling_coherence(1.0, SPLIT4, CLASS_III),
                    coupling_coherence(1.0, RING5, CLASS_I),
                    coupling_coherence(1.0, star(6), MSFWindow(nu_lower=1.0, nu_upper=3.0))):
        out = format_coupling(reading)
        assert reading.regime in out
        assert "f(C)" in out
