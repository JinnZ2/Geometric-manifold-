"""
Toy Landscape v3: True Double-Well Barrier Geometry

What v1 and v2 showed:
  Flat repair energy in both cases.
  v1: bang-bang controller clipped at constant contour.
  v2: bowl-plus-bump topology. Drift and safety gradient reach
      equilibrium before the state enters the steep curvature region.
      No ridge crossing. No spike. System stabilized in compromise.

Diagnosis:
  Curvature-induced energy spikes require genuine barrier geometry.
  Smooth convex compromise systems won't show it.
  If you don't build a phase transition into the Hamiltonian,
  you won't see one.

Fix:
  Replace quadratic base + Gaussian bump with true double-well.

  1D intuition:
    L(x) = x^4 - x^2
    Two attractors at x = +/-1/sqrt(2), saddle at x = 0.
    Curvature at saddle: L''(0) = -2  (negative = unstable equilibrium)
    Curvature at attractors: L''(+/-1/sqrt(2)) = +4  (stable)

  n-dimensional generalization:
    Safe attractor:   theta_safe  (origin or shifted)
    Unsafe attractor: theta_unsafe = a
    Saddle:           theta_saddle = (theta_safe + theta_unsafe) / 2

    L_safety(theta) = ((theta - theta_safe)^T (theta - theta_safe) - r^2)^2
                    projected onto the axis connecting safe to unsafe attractor.

    This creates a genuine energy barrier between wells.
    As state approaches saddle from safe side, Hessian eigenvalue
    along the connecting axis goes from positive -> zero -> negative.
    The sign flip IS the phase transition.

Prediction:
  As drift pushes state toward saddle:
    - lambda_max(H_safety) climbs then spikes near saddle
    - kappa_eff spikes when state enters saddle region
    - repair energy E = ||delta||^2 grows nonlinearly with proximity to saddle
    - kappa_eff spike precedes L_safety threshold crossing (leading indicator)

Falsification condition:
  If E vs dist_to_saddle is linear: geometry is still wrong.
  If kappa_eff does not spike near saddle: Hessian structure is not
    what we think it is.
  Either way we learn something real.
"""

import torch
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# Double-well landscape
# ─────────────────────────────────────────────────────────────────────────────

class DoubleWellLandscape:
    """
    True double-well in R^n along a 1D axis connecting two attractors.

    Let u = unit vector from safe to unsafe attractor.
    Let s = (theta - theta_safe) . u  (scalar projection onto axis)

    Double-well along axis:
      V_dw(s) = (s^2 - r^2)^2
      = s^4 - 2*r^2*s^2 + r^4

    Attractors at s = +/- r.
    Saddle at s = 0.

    Transverse directions: harmonic confinement (keeps state near axis).
      V_trans = kappa_t * ||theta_perp||^2

    Full potential:
      L_safety(theta) = (s^2 - r^2)^2 + kappa_t * ||theta_perp||^2

    Safe basin: s in [-r + margin, 0)  (left well, before saddle)
    Unsafe:     s in (0, r - margin]   (right well, after saddle)
    Saddle:     s = 0

    Curvature along axis:
      d^2 V_dw / ds^2 = 12*s^2 - 4*r^2
      At s=0:  = -4*r^2   (negative - saddle is unstable along axis)
      At s=+/-r: = 8*r^2  (positive - wells are stable)

    Hessian eigenvalue along u flips sign as state crosses saddle.
    That sign flip IS the measurable phase transition.
    """

    def __init__(self, n=10, r=1.5, kappa_t=2.0, safe_offset=0.0, unsafe_offset=None):
        self.n        = n
        self.r        = r          # half-distance between attractors
        self.kappa_t  = kappa_t   # transverse confinement strength

        # Safe attractor: at s = -r on axis
        self.u = torch.zeros(n)
        self.u[0] = 1.0            # axis = first coordinate

        # Safe attractor position
        self.theta_safe   = torch.zeros(n)
        self.theta_safe[0] = -r    # s = -r

        # Unsafe attractor
        self.theta_unsafe  = torch.zeros(n)
        self.theta_unsafe[0] = r   # s = +r

        # Saddle
        self.theta_saddle  = torch.zeros(n)  # s = 0

    def _decompose(self, theta):
        """Decompose theta into axial (s) and transverse components."""
        s          = (theta * self.u).sum()           # scalar projection
        theta_perp = theta - s * self.u               # transverse component
        return s, theta_perp

    def L_safety(self, theta):
        """
        Double-well safety potential.
        Safe basin = left well (s near -r).
        """
        s, theta_perp = self._decompose(theta)
        V_dw   = (s**2 - self.r**2)**2
        V_trans = self.kappa_t * (theta_perp**2).sum()
        return V_dw + V_trans

    def L_task(self, theta):
        """Pull toward safe attractor."""
        return ((theta - self.theta_safe)**2).sum()

    def axial_position(self, theta) -> float:
        """s: position along double-well axis. 0 = saddle."""
        return (theta * self.u).sum().item()

    def dist_to_saddle(self, theta) -> float:
        """Distance from saddle (s=0). This is what we plot E against."""
        return abs(self.axial_position(theta))

    def dist_to_unsafe(self, theta) -> float:
        return ((theta - self.theta_unsafe)**2).sum().sqrt().item()

    def hessian_safety(self, theta):
        t  = theta.detach().requires_grad_(True)
        ls = self.L_safety(t)
        g  = torch.autograd.grad(ls, t, create_graph=True)[0]
        H  = torch.zeros(self.n, self.n)
        for i in range(self.n):
            H[i] = torch.autograd.grad(g[i], t, retain_graph=True)[0].detach()
        return H

    def lambda_max(self, theta) -> float:
        return torch.linalg.eigvalsh(self.hessian_safety(theta)).max().item()

    def lambda_axial(self, theta) -> float:
        """
        Eigenvalue of H_safety along the double-well axis u.
        = u^T H u
        = d^2 V_dw / ds^2 = 12*s^2 - 4*r^2

        Analytic formula - no Hessian computation needed.
        Negative near saddle, positive in wells.
        """
        s = self.axial_position(theta)
        return 12.0 * s**2 - 4.0 * self.r**2

    def kappa_eff(self, theta, theta_dot) -> float:
        denom = (theta_dot**2).sum().item()
        if denom < 1e-12: return 0.0
        H  = self.hessian_safety(theta)
        Hv = H @ theta_dot
        return (theta_dot * Hv).sum().item() / denom

    def hessian_frob_sq(self, theta):
        return (self.hessian_safety(theta)**2).sum()

    def saddle_proximity_zone(self, theta, margin=0.3) -> bool:
        """True when state is near saddle - high curvature region."""
        return abs(self.axial_position(theta)) < margin


# ─────────────────────────────────────────────────────────────────────────────
# Drift: push along axis toward saddle and beyond
# ─────────────────────────────────────────────────────────────────────────────

def axial_drift(theta, landscape, strength, step, ramp_start=20, ramp_len=100):
    """
    Push along double-well axis toward unsafe attractor.
    Ramps up gradually to allow curvature measurement across full trajectory.
    """
    if step < ramp_start:
        return torch.zeros(landscape.n)
    ramp = min(1.0, (step - ramp_start) / ramp_len)
    return strength * ramp * landscape.u   # push in +u direction toward unsafe


# ─────────────────────────────────────────────────────────────────────────────
# Soft continuous repair
# ─────────────────────────────────────────────────────────────────────────────

def soft_repair(theta, landscape, epsilon_s, gamma):
    """
    Soft repair: scales with violation magnitude.
    No hard threshold clip.
    """
    t      = theta.detach().requires_grad_(True)
    ls     = landscape.L_safety(t)
    violation = max(0.0, ls.item() - epsilon_s)
    if violation < 1e-6:
        return torch.zeros(landscape.n), 0.0
    grad  = torch.autograd.grad(ls, t)[0].detach()
    delta = -gamma * violation * grad
    return delta, (delta**2).sum().item()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n=10, r=1.5, kappa_t=2.0,
    lambda_s=0.3,           # weak safety gradient: let drift accumulate
    drift_strength=0.5,     # strong axial push
    epsilon_s=6.0,          # high threshold: permit deep excursion
    lr=0.05,
    gamma_repair=0.02,      # very weak repair: system must wander deep
    steps=500,
    curvature_reg=0.0,
    label='',
    seed=42,
    ramp_start=30,
    ramp_len=120
):
    torch.manual_seed(seed)
    land  = DoubleWellLandscape(n=n, r=r, kappa_t=kappa_t)

    # Start in left well (safe attractor)
    theta = land.theta_safe.clone() + 0.05 * torch.randn(n)

    records = []

    for step in range(steps):
        # ── Gradient of objective ────────────────────────────────────────────
        t = theta.detach().requires_grad_(True)
        l_task   = land.L_task(t)
        l_safety = land.L_safety(t)

        if curvature_reg > 0.0:
            h_reg = curvature_reg * land.hessian_frob_sq(theta.detach())
            total = l_task + lambda_s * l_safety + h_reg
        else:
            total = l_task + lambda_s * l_safety

        total.backward()
        grad_step = -lr * t.grad.detach()

        # ── Axial drift ──────────────────────────────────────────────────────
        eta       = axial_drift(theta, land, drift_strength, step,
                                ramp_start, ramp_len)
        theta_dot = grad_step + eta
        theta     = theta.detach() + theta_dot

        # ── Soft repair ──────────────────────────────────────────────────────
        repair_delta, repair_energy = soft_repair(theta, land, epsilon_s, gamma_repair)
        if repair_energy > 0:
            theta = theta + repair_delta

        # ── Measurements ─────────────────────────────────────────────────────
        s          = land.axial_position(theta)
        d_saddle   = land.dist_to_saddle(theta)
        lambda_ax  = land.lambda_axial(theta)   # analytic, cheap
        ls_val     = land.L_safety(theta).item()
        kappa      = land.kappa_eff(theta, theta_dot)
        lmax       = land.lambda_max(theta)
        in_zone    = land.saddle_proximity_zone(theta)

        records.append({
            'step':             step,
            'label':            label,
            's':                s,                  # axial position
            'dist_to_saddle':   d_saddle,
            'L_safety':         ls_val,
            'L_task':           land.L_task(theta).item(),
            'lambda_axial':     lambda_ax,          # analytic curvature on axis
            'lambda_max_H':     lmax,
            'kappa_eff':        kappa,
            'repair_energy':    repair_energy,
            'repair_active':    repair_energy > 0,
            'in_saddle_zone':   in_zone,
            'drift_norm':       eta.norm().item(),
        })

        if step % 100 == 0:
            print(f"  {step:4d} | s={s:+.3f} | d_saddle={d_saddle:.3f} | "
                  f"lam_ax={lambda_ax:+.3f} | kappa={kappa:.3f} | "
                  f"E={repair_energy:.5f} | zone={in_zone}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot(results: dict, r: float, output_dir: str):
    colors = {'baseline': 'steelblue', 'regularized': 'firebrick'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'Double-Well v3 | Saddle at s=0, Wells at s=±{r}\n'
        'Does repair energy spike as state crosses saddle?',
        fontsize=11
    )

    for label, df in results.items():
        c = colors.get(label, 'gray')

        # Axial trajectory
        axes[0,0].plot(df['step'], df['s'], color=c, alpha=0.8, label=label)
        axes[0,0].axhline(0,  color='black', ls='--', lw=0.8, label='saddle')
        axes[0,0].axhline(-r, color='green', ls=':',  lw=0.8, label='safe well')
        axes[0,0].axhline( r, color='red',   ls=':',  lw=0.8, label='unsafe well')

        # Analytic axial curvature over time
        axes[0,1].plot(df['step'], df['lambda_axial'], color=c, alpha=0.8, label=label)
        axes[0,1].axhline(0, color='black', ls='--', lw=0.8)

        # kappa_eff over time
        axes[0,2].plot(df['step'], df['kappa_eff'], color=c, alpha=0.8, label=label)

        # KEY: E_repair vs dist_to_saddle
        active = df[df['repair_active'] & (df['repair_energy'] > 1e-8)]
        if len(active):
            axes[1,0].scatter(active['dist_to_saddle'], active['repair_energy'],
                              color=c, alpha=0.6, s=20, label=label)

        # kappa_eff vs dist_to_saddle
        axes[1,1].scatter(df['dist_to_saddle'], df['kappa_eff'],
                          color=c, alpha=0.3, s=8, label=label)

        # lambda_max over time
        axes[1,2].plot(df['step'], df['lambda_max_H'], color=c, alpha=0.8, label=label)

    axes[0,0].set(title='Axial position s over time', xlabel='step', ylabel='s')
    axes[0,1].set(title='Analytic axial curvature (12s²-4r²)',
                  xlabel='step', ylabel='d²V/ds²')
    axes[0,2].set(title='kappa_eff over time', xlabel='step', ylabel='kappa_eff')
    axes[1,0].set(title='E_repair vs dist_to_saddle  [KNEE TEST]',
                  xlabel='dist to saddle', ylabel='E_repair')
    axes[1,1].set(title='kappa_eff vs dist_to_saddle',
                  xlabel='dist to saddle', ylabel='kappa_eff')
    axes[1,2].set(title='lambda_max(H) over time', xlabel='step', ylabel='lambda_max')

    for ax in axes.flat:
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{output_dir}/double_well_results.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Falsification report
# ─────────────────────────────────────────────────────────────────────────────

def falsification_report(results: dict, r: float):
    print("\n" + "="*65)
    print("FALSIFICATION REPORT v3: Double-Well")
    print(f"Saddle at s=0 | Wells at s=±{r}")
    print("Prediction: E_repair spikes as dist_to_saddle -> 0")
    print("="*65)

    for label, df in results.items():
        print(f"\n--- {label} ---")

        # Did the state actually approach the saddle?
        min_d = df['dist_to_saddle'].min()
        saddle_crossings = (df['s'] > 0).sum()  # steps past saddle
        print(f"  Min dist to saddle:     {min_d:.4f}")
        print(f"  Steps past saddle (s>0):{saddle_crossings}")

        if min_d > 0.5:
            print("  WARNING: state never got close to saddle.")
            print("  Increase drift_strength or reduce lambda_s.")
            continue

        active = df[df['repair_active'] & (df['repair_energy'] > 1e-8)]
        print(f"  Repair events:          {len(active)}")

        if len(active) < 5:
            print("  Insufficient repair events. Raise epsilon_s or reduce gamma.")
            continue

        # Nonlinearity: close vs far from saddle
        median_d = active['dist_to_saddle'].median()
        far_E    = active[active['dist_to_saddle'] >  median_d]['repair_energy'].mean()
        close_E  = active[active['dist_to_saddle'] <= median_d]['repair_energy'].mean()
        ratio    = close_E / (far_E + 1e-10)

        r_p, _ = pearsonr( active['dist_to_saddle'], active['repair_energy'])
        r_s, _ = spearmanr(active['dist_to_saddle'], active['repair_energy'])

        # Leading indicator: kappa_eff spike before saddle crossing
        kappa_90     = df['kappa_eff'].quantile(0.90)
        first_kspike = df[df['kappa_eff'] > kappa_90]['step'].min()
        first_cross  = df[df['s'] > -0.2]['step'].min()  # entering saddle approach
        lead = (int(first_cross) - int(first_kspike)
                if not (np.isnan(first_kspike) or np.isnan(first_cross)) else None)

        # Analytic check: does lambda_axial go negative near saddle?
        near_saddle   = df[df['dist_to_saddle'] < 0.3]
        lam_ax_saddle = near_saddle['lambda_axial'].mean() if len(near_saddle) else np.nan

        print(f"  Close/Far E ratio:      {ratio:.2f}x  (>2 = knee)")
        print(f"  Pearson(dist, E):       {r_p:.3f}")
        print(f"  Spearman(dist, E):      {r_s:.3f}")
        print(f"  kappa spike step:       {first_kspike}")
        print(f"  Saddle approach step:   {first_cross}")
        print(f"  Lead time:              {lead} steps")
        print(f"  Mean lambda_axial near saddle: {lam_ax_saddle:.3f}  (negative = saddle)")

        knee    = ratio > 2.0
        leading = lead is not None and lead > 0
        saddle  = not np.isnan(lam_ax_saddle) and lam_ax_saddle < 0

        print(f"\n  [KNEE]           {'SUPPORTED' if knee    else 'NOT SUPPORTED'}")
        print(f"  [LEADING IND.]   {'SUPPORTED' if leading else 'NOT SUPPORTED'}")
        print(f"  [SADDLE REACHED] {'CONFIRMED' if saddle  else 'NOT REACHED'}")

    print("\n" + "="*65)
    print("Key diagnostic: check 'Axial position s over time' plot.")
    print("If s never approaches 0: drift insufficient.")
    print("If s crosses 0 but no knee: curvature-energy link is wrong.")
    print("If knee appears only in baseline: regularization is working.")
    print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUTPUT = 'results/toy_v3'
    os.makedirs(OUTPUT, exist_ok=True)

    R = 1.5
    SHARED = dict(
        n=10, r=R, kappa_t=2.0,
        lambda_s=0.3, drift_strength=0.5,
        epsilon_s=6.0, lr=0.05,
        gamma_repair=0.02, steps=500, seed=42
    )

    results = {}
    for reg_label, curv_reg in [('baseline', 0.0), ('regularized', 0.1)]:
        print(f"\n=== {reg_label} ===")
        df = run(**SHARED, curvature_reg=curv_reg, label=reg_label)
        df.to_csv(f'{OUTPUT}/{reg_label}.csv', index=False)
        results[reg_label] = df

    plot(results, R, OUTPUT)
    falsification_report(results, R)



patches:

def run_no_repair(
    n=10, r=1.5, kappa_t=2.0,
    lambda_s=0.1,           # very weak: let drift dominate
    drift_strength=0.6,
    steps=300,
    lr=0.05,
    seed=42,
    ramp_start=20,
    ramp_len=80
):
    """
    Repair disabled entirely.
    Pure measurement of landscape dynamics under drift.

    Observe:
      - When does s cross 0 (saddle)?
      - Does trajectory diverge after crossing? (unstable manifold)
      - What does lambda_axial look like across the full crossing?

    This is the clean geometric baseline before any controller effects.
    """
    torch.manual_seed(seed)
    land  = DoubleWellLandscape(n=n, r=r, kappa_t=kappa_t)
    theta = land.theta_safe.clone() + 0.05 * torch.randn(n)

    records = []

    for step in range(steps):
        t = theta.detach().requires_grad_(True)
        total = land.L_task(t) + lambda_s * land.L_safety(t)
        total.backward()
        grad_step = -lr * t.grad.detach()

        eta       = axial_drift(theta, land, drift_strength, step, ramp_start, ramp_len)
        theta_dot = grad_step + eta
        theta     = theta.detach() + theta_dot

        s         = land.axial_position(theta)
        lam_ax    = land.lambda_axial(theta)   # 12s² - 4r²: sign is the signal
        kappa     = land.kappa_eff(theta, theta_dot)

        records.append({
            'step':           step,
            's':              s,
            'lambda_axial':   lam_ax,
            'kappa_eff':      kappa,
            'lambda_max_H':   land.lambda_max(theta),
            'L_safety':       land.L_safety(theta).item(),
            'theta_dot_norm': theta_dot.norm().item(),
            'in_unstable':    lam_ax < 0,        # past saddle, axially unstable
        })

        if step % 60 == 0:
            print(f"  {step:4d} | s={s:+.4f} | lam_ax={lam_ax:+.3f} | "
                  f"kappa={kappa:.3f} | unstable={lam_ax < 0}")

    return pd.DataFrame(records)


def run_repair_on_unstable_manifold(
    n=10, r=1.5, kappa_t=2.0,
    lambda_s=0.1,
    drift_strength=0.6,
    lr=0.05,
    gamma_repair=0.05,
    steps=300,
    seed=42,
    ramp_start=20,
    ramp_len=80,
    repair_start_s=0.5     # only trigger repair after crossing this far past saddle
):
    """
    Repair triggers ONLY after state has wandered into unstable manifold.
    Measures repair energy as function of lambda_axial at trigger time.

    This directly tests: does repair cost scale with instability magnitude?

    If E_repair vs lambda_axial_at_trigger is nonlinear: 
        repair cost is instability-dominated. Hypothesis supported.
    If linear:
        repair cost is gradient-magnitude-dominated. Hypothesis weakened.

    repair_start_s: how far past saddle before repair activates.
    Larger value = deeper into unstable manifold = steeper curvature at trigger.
    """
    torch.manual_seed(seed)
    land  = DoubleWellLandscape(n=n, r=r, kappa_t=kappa_t)
    theta = land.theta_safe.clone() + 0.05 * torch.randn(n)

    records = []

    for step in range(steps):
        t = theta.detach().requires_grad_(True)
        total = land.L_task(t) + lambda_s * land.L_safety(t)
        total.backward()
        grad_step = -lr * t.grad.detach()

        eta       = axial_drift(theta, land, drift_strength, step, ramp_start, ramp_len)
        theta_dot = grad_step + eta
        theta     = theta.detach() + theta_dot

        s       = land.axial_position(theta)
        lam_ax  = land.lambda_axial(theta)

        # Repair only after crossing repair_start_s past saddle
        repair_energy    = 0.0
        repair_triggered = False
        lam_ax_at_repair = None

        if s > repair_start_s:
            t2   = theta.detach().requires_grad_(True)
            ls   = land.L_safety(t2)
            grad = torch.autograd.grad(ls, t2)[0].detach()

            # Scale repair by lambda_axial magnitude - this is the key measurement
            # Standard repair: delta = -gamma * grad
            delta         = -gamma_repair * grad
            repair_energy = (delta**2).sum().item()
            theta         = theta.detach() + delta
            repair_triggered  = True
            lam_ax_at_repair  = lam_ax   # curvature at moment of repair

        records.append({
            'step':               step,
            's':                  s,
            'lambda_axial':       lam_ax,
            'lambda_axial_at_repair': lam_ax_at_repair,
            'kappa_eff':          land.kappa_eff(theta, theta_dot),
            'repair_energy':      repair_energy,
            'repair_triggered':   repair_triggered,
            'L_safety':           land.L_safety(theta).item(),
        })

    return pd.DataFrame(records)


def falsification_report_v3b(df_no_repair: pd.DataFrame,
                              df_unstable: pd.DataFrame,
                              r: float):
    """
    Two-part report:

    Part 1 (no repair):
      Did the trajectory cross the saddle?
      Did lambda_axial go negative?
      Did the trajectory diverge after crossing? (unstable manifold confirmed)

    Part 2 (repair on unstable manifold):
      Is E_repair nonlinearly related to lambda_axial at trigger time?
      That's the direct test of instability-vs-distance dominated repair cost.
    """
    print("\n" + "="*65)
    print("FALSIFICATION REPORT v3b")
    print("="*65)

    print("\n--- Part 1: No-repair trajectory ---")
    crossed   = df_no_repair[df_no_repair['s'] > 0]
    unstable  = df_no_repair[df_no_repair['in_unstable']]
    max_s     = df_no_repair['s'].max()
    min_lam   = df_no_repair['lambda_axial'].min()

    print(f"  Max axial position:        {max_s:.4f}  (r={r}, saddle=0)")
    print(f"  Saddle crossed (s>0):      {len(crossed)} steps")
    print(f"  Steps in unstable region:  {len(unstable)}")
    print(f"  Min lambda_axial:          {min_lam:.4f}  (negative = saddle confirmed)")

    if len(crossed) > 0:
        # Check divergence: does theta_dot_norm increase after crossing?
        pre_cross  = df_no_repair[df_no_repair['s'] <= 0]['theta_dot_norm'].mean()
        post_cross = df_no_repair[df_no_repair['s'] >  0]['theta_dot_norm'].mean()
        divergence_ratio = post_cross / (pre_cross + 1e-8)
        print(f"  Mean speed pre/post saddle:{pre_cross:.4f} / {post_cross:.4f}")
        print(f"  Divergence ratio:          {divergence_ratio:.2f}x")
        print(f"  Unstable manifold confirmed: {'YES' if divergence_ratio > 1.2 else 'NO'}")
    else:
        print("  Saddle never crossed. Increase drift_strength.")

    print("\n--- Part 2: Repair on unstable manifold ---")
    repairs = df_unstable[df_unstable['repair_triggered']].dropna(
        subset=['lambda_axial_at_repair']
    )
    print(f"  Repair events:             {len(repairs)}")

    if len(repairs) < 5:
        print("  Insufficient events. Reduce repair_start_s.")
    else:
        # Is E_repair nonlinear with lambda_axial at trigger?
        lam_vals = repairs['lambda_axial_at_repair'].values
        e_vals   = repairs['repair_energy'].values

        r_p, _ = pearsonr( lam_vals, e_vals)
        r_s, _ = spearmanr(lam_vals, e_vals)

        # Quartile comparison
        q1_mask = lam_vals <= np.percentile(lam_vals, 25)
        q4_mask = lam_vals >= np.percentile(lam_vals, 75)
        e_q1 = e_vals[q1_mask].mean()   # low curvature repairs
        e_q4 = e_vals[q4_mask].mean()   # high curvature repairs
        ratio = e_q4 / (e_q1 + 1e-10)

        print(f"  Pearson(lambda_ax, E):     {r_p:.3f}")
        print(f"  Spearman(lambda_ax, E):    {r_s:.3f}")
        print(f"  E at high vs low curvature:{ratio:.2f}x  (>2 = instability-dominated)")
        print(f"  Mean E Q1 (low curv):      {e_q1:.5f}")
        print(f"  Mean E Q4 (high curv):     {e_q4:.5f}")

        instability_dominated = ratio > 2.0
        print(f"\n  [INSTABILITY-DOMINATED COST] "
              f"{'SUPPORTED' if instability_dominated else 'NOT SUPPORTED'}")
        if not instability_dominated:
            print("  Repair cost scales with gradient magnitude, not Hessian structure.")
            print("  Curvature-cost hypothesis weakened. Back to whiteboard on cost model.")
        else:
            print("  Repair cost scales with curvature at trigger point.")
            print("  Thermodynamic framing gains credibility.")

    print("="*65 + "\n")

Add to __main__ block:

    # v3b: no-repair baseline + repair on unstable manifold
    print("\n=== NO REPAIR: pure landscape dynamics ===")
    df_no_repair = run_no_repair(n=10, r=R, kappa_t=2.0,
                                  lambda_s=0.1, drift_strength=0.6, steps=300)
    df_no_repair.to_csv(f'{OUTPUT}/no_repair.csv', index=False)

    print("\n=== REPAIR ON UNSTABLE MANIFOLD ===")
    df_unstable = run_repair_on_unstable_manifold(
        n=10, r=R, kappa_t=2.0,
        lambda_s=0.1, drift_strength=0.6,
        gamma_repair=0.05, steps=300,
        repair_start_s=0.3
    )
    df_unstable.to_csv(f'{OUTPUT}/repair_unstable.csv', index=False)

    falsification_report_v3b(df_no_repair, df_unstable, R)


The divergence ratio in Part 1 is the diagnostic that confirms the unstable manifold is real - if speed increases after saddle crossing, the system is genuinely on the unstable side and perturbations are amplifying. If it doesn’t diverge, the transverse confinement is overwhelming the axial instability and kappa_t needs to come down.
Part 2 measures E_repair against lambda_axial at the moment repair fires, not against distance. That’s GPT’s point made precise - if the ratio is above 2x, the system is paying for instability, not travel distance. That’s the distinction that separates the thermodynamic framing from a simple gradient-magnitude story.​​​​​​​​​​​​​​​​


