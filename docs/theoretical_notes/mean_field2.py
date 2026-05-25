# mean_field.py
# stdlib + numpy + scipy.optimize
# Symmetric Gaussian-closure mean-field for the diversity-collapse model.
# Per active dim k under symmetry: state = (m, s2).
#   ṁ   = -m³ + a·m - 3·m·s2
#   ṡ2  = 2·s2·(-3·m² + a - 3·s2 - c) + 2·(σ_n² + D_mut)

import numpy as np
from scipy.optimize import fsolve

def mf_rhs(state, a, c, sigma_n, D_mut):
    m, s2 = state
    s2 = max(s2, 1e-12)
    dm  = -m**3 + a*m - 3.0*m*s2
    ds2 = 2.0*s2*(-3.0*m**2 + a - 3.0*s2 - c) + 2.0*(sigma_n**2 + D_mut)
    return np.array([dm, ds2])

def find_fixed_points(a, c, sigma_n, D_mut):
    seeds = [
        (0.0, 0.5),
        (0.0, 1.5),
        ( np.sqrt(max(a, 1e-6)),  1e-3),
        (-np.sqrt(max(a, 1e-6)),  1e-3),
        ( 0.5*np.sqrt(max(a, 1e-6)), 0.2),
        (-0.5*np.sqrt(max(a, 1e-6)), 0.2),
    ]
    fps = []
    for s in seeds:
        try:
            sol, info, ier, _ = fsolve(
                mf_rhs, s, args=(a, c, sigma_n, D_mut),
                full_output=True, xtol=1e-12
            )
            if ier == 1 and sol[1] > 0:
                res = np.linalg.norm(mf_rhs(sol, a, c, sigma_n, D_mut))
                if res < 1e-8:
                    fps.append(tuple(sol))
        except Exception:
            pass
    uniq = []
    for fp in fps:
        if not any(np.allclose(fp, u, atol=1e-4) for u in uniq):
            uniq.append(fp)
    return uniq

def jacobian(fp, a, c, sigma_n, D_mut, eps=1e-6):
    m, s2 = fp
    J = np.zeros((2, 2))
    for j, dx in enumerate([(eps, 0.0), (0.0, eps)]):
        f_plus  = mf_rhs((m + dx[0], s2 + dx[1]), a, c, sigma_n, D_mut)
        f_minus = mf_rhs((m - dx[0], s2 - dx[1]), a, c, sigma_n, D_mut)
        J[:, j] = (f_plus - f_minus) / (2*eps)
    return J

def classify(fp, a, c, sigma_n, D_mut):
    J = jacobian(fp, a, c, sigma_n, D_mut)
    eigs = np.linalg.eigvals(J)
    n_pos = int(np.sum(eigs.real >  1e-8))
    n_neg = int(np.sum(eigs.real < -1e-8))
    if n_pos == 0 and n_neg == 2:
        kind = "stable"
    elif n_pos == 2:
        kind = "repeller"
    elif n_pos == 1 and n_neg == 1:
        kind = "saddle_idx1"
    else:
        kind = f"degenerate(npos={n_pos},nneg={n_neg})"
    return eigs, kind

def lambda_plus(fp, a, c, sigma_n, D_mut):
    eigs, _ = classify(fp, a, c, sigma_n, D_mut)
    pos = [e.real for e in eigs if e.real > 1e-8]
    return max(pos) if pos else float("nan")

def mf_potential_line(fp_a, fp_b, a, c, sigma_n, D_mut, n_steps=400):
    # ΔΦ ≈ -∫ RHS · ds   along straight segment fp_a → fp_b
    s0 = np.array(fp_a); s1 = np.array(fp_b)
    pts = np.linspace(0.0, 1.0, n_steps)
    ds  = (s1 - s0) / (n_steps - 1)
    total = 0.0
    for t in pts:
        s = s0 + t*(s1 - s0)
        rhs = mf_rhs(s, a, c, sigma_n, D_mut)
        total += -np.dot(rhs, ds)
    return total

def label_fps(fps, a, c, sigma_n, D_mut):
    # diverse = (m≈0, s2 large), homog ±, saddle = idx1 not at 0
    labeled = {"diverse": None, "homog_plus": None,
               "homog_minus": None, "saddle": []}
    for fp in fps:
        m, s2 = fp
        eigs, kind = classify(fp, a, c, sigma_n, D_mut)
        if kind == "stable" and abs(m) < 1e-3:
            labeled["diverse"] = fp
        elif kind == "stable" and m >  1e-3:
            labeled["homog_plus"]  = fp
        elif kind == "stable" and m < -1e-3:
            labeled["homog_minus"] = fp
        elif "saddle" in kind:
            labeled["saddle"].append(fp)
    return labeled

def find_c_for_target_barrier(a, sigma_n, D_mut, target_dphi,
                              c_lo=0.05, c_hi=3.0, tol=1e-3, max_iter=60):
    # monotone search: ΔΦ increases as c increases (homog basin deepens)
    def dphi_of(c):
        fps = find_fixed_points(a, c, sigma_n, D_mut)
        lab = label_fps(fps, a, c, sigma_n, D_mut)
        if lab["diverse"] is None or not lab["saddle"]:
            return None
        saddle = min(lab["saddle"], key=lambda fp: abs(fp[0]))
        return mf_potential_line(lab["diverse"], saddle,
                                 a, c, sigma_n, D_mut)
    # bracket
    for _ in range(max_iter):
        c_mid = 0.5*(c_lo + c_hi)
        d_mid = dphi_of(c_mid)
        if d_mid is None:
            c_lo = c_mid; continue
        if abs(d_mid - target_dphi) < tol:
            return c_mid, d_mid
        if d_mid < target_dphi:
            c_lo = c_mid
        else:
            c_hi = c_mid
    return c_mid, d_mid

# ---- self-test ----
if __name__ == "__main__":
    a, sigma_n, D_mut = 1.0, 0.15, 0.02
    for c in [0.3, 0.6, 0.9, 1.2]:
        fps = find_fixed_points(a, c, sigma_n, D_mut)
        lab = label_fps(fps, a, c, sigma_n, D_mut)
        print(f"\nc = {c}")
        for k, v in lab.items():
            if k == "saddle":
                for s in v:
                    eigs, kind = classify(s, a, c, sigma_n, D_mut)
                    lp = lambda_plus(s, a, c, sigma_n, D_mut)
                    print(f"  saddle: {s}  eigs={eigs}  λ+={lp:.4f}")
            else:
                print(f"  {k}: {v}")
        if lab["diverse"] and lab["saddle"]:
            saddle = min(lab["saddle"], key=lambda fp: abs(fp[0]))
            dphi = mf_potential_line(lab["diverse"], saddle,
                                     a, c, sigma_n, D_mut)
            print(f"  ΔΦ(diverse → saddle) = {dphi:.4f}")
