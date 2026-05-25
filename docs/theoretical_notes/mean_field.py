# mean_field.py — stdlib + numpy + scipy.optimize only
import numpy as np
from scipy.optimize import fsolve

def mf_rhs(state, a, c, sigma_n, D_mut):
    """
    state = (m, s2) for one active dim under symmetric ansatz.
    Returns (dm/dt, ds2/dt).
    """
    m, s2 = state
    s2 = max(s2, 1e-12)
    dm  = -m**3 + a*m - 3.0*m*s2
    ds2 = 2.0*s2*(-3.0*m**2 + a - 3.0*s2 - c) + 2.0*(sigma_n**2 + D_mut)
    return np.array([dm, ds2])

def find_fixed_points(a, c, sigma_n, D_mut):
    """Return dict of fixed points found by multi-start fsolve."""
    seeds = [
        ( 0.0, 1.0),     # diverse candidate
        ( np.sqrt(a), 1e-3),  # homogeneous +
        (-np.sqrt(a), 1e-3),  # homogeneous -
        ( 0.5*np.sqrt(a), 0.1),  # saddle-ish
    ]
    fps = []
    for s in seeds:
        try:
            sol, info, ier, _ = fsolve(
                mf_rhs, s, args=(a, c, sigma_n, D_mut),
                full_output=True, xtol=1e-10
            )
            if ier == 1 and sol[1] > 0:
                fps.append(tuple(sol))
        except Exception:
            pass
    # dedupe
    uniq = []
    for fp in fps:
        if not any(np.allclose(fp, u, atol=1e-4) for u in uniq):
            uniq.append(fp)
    return uniq

def classify_fixed_point(fp, a, c, sigma_n, D_mut, eps=1e-6):
    """Jacobian eigenvalues at fp; returns (eigs, kind)."""
    m, s2 = fp
    J = np.zeros((2, 2))
    for i, dx in enumerate([(eps, 0), (0, eps)]):
        f_plus  = mf_rhs((m + dx[0], s2 + dx[1]), a, c, sigma_n, D_mut)
        f_minus = mf_rhs((m - dx[0], s2 - dx[1]), a, c, sigma_n, D_mut)
        J[:, i] = (f_plus - f_minus) / (2*eps)
    eigs = np.linalg.eigvals(J)
    n_pos = int(np.sum(eigs.real > 1e-8))
    n_neg = int(np.sum(eigs.real < -1e-8))
    if n_pos == 0:
        kind = "stable"
    elif n_neg == 0:
        kind = "repeller"
    else:
        kind = f"saddle_idx{n_pos}"
    return eigs, kind

def mf_potential(state_path, a, c, sigma_n, D_mut, n_steps=200):
    """
    Line-integrate -RHS · d(state) along straight path between two fps
    to estimate ΔΦ. RHS is gradient of -Φ, so ΔΦ ≈ -∫ RHS · ds.
    """
    s0, s1 = np.array(state_path[0]), np.array(state_path[1])
    ts = np.linspace(0, 1, n_steps)
    ds = (s1 - s0) / (n_steps - 1)
    integ = 0.0
    for t in ts:
        s = s0 + t*(s1 - s0)
        rhs = mf_rhs(s, a, c, sigma_n, D_mut)
        integ += -np.dot(rhs, ds)
    return integ

def lambda_plus_at_saddle(saddle_fp, a, c, sigma_n, D_mut, eps=1e-6):
    eigs, _ = classify_fixed_point(saddle_fp, a, c, sigma_n, D_mut, eps)
    pos = [e.real for e in eigs if e.real > 0]
    return max(pos) if pos else np.nan
