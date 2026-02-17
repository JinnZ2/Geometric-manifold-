"""
Toy Landscape v2: Delayed Correction Regime

What v1 showed:
  Flat repair energy per event = bang-bang controller clamping oscillations
  before curvature can steepen. System clips at constant energy contour.
  Sawtooth dynamic, not nonlinear spike.

What v1 was missing:
  Drift must outrun the controller long enough for the system to wander
  into genuinely high-curvature territory before correction fires.

Changes from v1:
  1. gamma_repair reduced (weaker correction)
  2. epsilon_s raised (later trigger)
  3. Continuous soft repair penalty instead of hard threshold
  4. Repair energy measured as function of distance to unsafe attractor
     not just distance to threshold contour

Key distinction:
  v1 measured: E_repair vs L_safety at trigger time (constant by construction)
  v2 measures: E_repair vs ||theta - a|| (actual geometric proximity to attractor)

Prediction:
  As ||theta - a|| decreases, local Hessian eigenvalues grow.
  Repair energy E = ||delta||^2 should grow nonlinearly with proximity.
  kappa_eff should spike when ||theta - a|| < sigma (inside Gaussian well).

Falsification:
  If E vs ||theta - a|| is still linear: curvature hypothesis fails.
  If kappa_eff does not precede energy spike: no leading indicator.
  If curvature regularization does not reduce slope: shaping has no effect.

Control regimes tested:
  tight:   gamma=0.3, epsilon=2.0  (v1 regime, for reference)
  medium:  gamma=0.1, epsilon=3.0
  loose:   gamma=0.03, epsilon=4.0  (lets system wander into high-curv region)
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
# Landscape (same as v1, reproduced for standalone use)
# ─────────────────────────────────────────────────────────────────────────────

class ToyLandscape:
    def __init__(self, n=10, beta=5.0, sigma=1.0, a=None):
        self.n, self.beta, self.sigma = n, beta, sigma
        if a is None:
            a = torch.zeros(n); a[0] = 3.0
        self.a      = a
        self.target = torch.zeros(n)

    def L_task(self, theta):
        return ((theta - self.target) ** 2).sum()

    def L_safety(self, theta):
        return (theta**2).sum() + self.beta * torch.exp(
            -((theta - self.a)**2).sum() / self.sigma**2
        )

    def hessian_safety(self, theta):
        t  = theta.detach().requires_grad_(True)
        ls = self.L_safety(t)
        g  = torch.autograd.grad(ls, t, create_graph=True)[0]
        H  = torch.zeros(self.n, self.n)
        for i in range(self.n):
            H[i] = torch.autograd.grad(g[i], t, retain_graph=True)[0].detach()
        return H

    def kappa_eff(self, theta, theta_dot):
        denom = (theta_dot**2).sum().item()
        if denom < 1e-12: return 0.0
        H   = self.hessian_safety(theta)
        Hv  = H @ theta_dot
        return abs((theta_dot * Hv).sum().item() / denom)

    def lambda_max(self, theta):
        return torch.linalg.eigvalsh(self.hessian_safety(theta)).max().item()

    def hessian_frob_sq(self, theta):
        return (self.hessian_safety(theta) ** 2).sum()

    def dist_to_attractor(self, theta):
        return ((theta - self.a)**2).sum().sqrt().item()


# ─────────────────────────────────────────────────────────────────────────────
# Soft repair: continuous penalty instead of hard threshold
# ─────────────────────────────────────────────────────────────────────────────

def soft_repair_gradient(theta, landscape, epsilon_s, gamma):
    """
    Continuous repair force: activates when L_safety > epsilon_s
    but scales with violation magnitude rather than clipping at threshold.

    delta = -gamma * max(0, L_safety - epsilon_s) * grad_L_safety

    This allows the system to accumulate curvature before correction
    magnitude becomes significant. The farther past threshold,
    the stronger the pull - but no hard clip.
    """
    t      = theta.detach().requires_grad_(True)
    ls     = landscape.L_safety(t)
    violation = max(0.0, ls.item() - epsilon_s)
    if violation < 1e-6:
        return torch.zeros_like(theta), 0.0

    grad = torch.autograd.grad(ls, t)[0].detach()
    delta = -gamma * violation * grad
    energy = (delta**2).sum().item()
    return delta, energy


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n=10, beta=5.0, sigma=1.0,
    lambda_s=0.5,           # reduced: less safety pressure, lets drift accumulate
    drift_strength=0.4,     # slightly stronger push
    epsilon_s=3.5,          # raised: later trigger
    lr=0.05,
    gamma_repair=0.05,      # loosened: weaker correction per step
    steps=400,
    curvature_reg=0.0,
    label='',
    seed=42,
    ramp_start=30
):
    torch.manual_seed(seed)
    land = ToyLandscape(n=n, beta=beta, sigma=sigma)
    theta = 0.1 * torch.randn(n)

    records = []

    for step in range(steps):
        # ── Gradient step ────────────────────────────────────────────────────
        t = theta.detach().requires_grad_(True)
        l_task   = land.L_task(t)
        l_safety = land.L_safety(t)

        if curvature_reg > 0.0:
            # Curvature regularization: penalize unsafe Hessian norm
            h_reg = curvature_reg * land.hessian_frob_sq(theta.detach())
            total = l_task + lambda_s * l_safety + h_reg
        else:
            total = l_task + lambda_s * l_safety

        total.backward()
        grad_step = -lr * t.grad.detach()

        # ── Drift: ramp toward unsafe attractor ──────────────────────────────
        if step >= ramp_start:
            ramp      = min(1.0, (step - ramp_start) / 80.0)
            direction = (land.a - theta.detach())
            direction = direction / (direction.norm() + 1e-8)
            eta       = drift_strength * ramp * direction
        else:
            eta = torch.zeros(n)

        theta_dot = grad_step + eta
        theta     = theta.detach() + theta_dot

        # ── Soft continuous repair ───────────────────────────────────────────
        repair_delta, repair_energy = soft_repair_gradient(
            theta, land, epsilon_s, gamma_repair
        )
        if repair_energy > 0:
            theta = theta + repair_delta

        # ── Measure ──────────────────────────────────────────────────────────
        kappa  = land.kappa_eff(theta, theta_dot)
        lmax   = land.lambda_max(theta)
        d_att  = land.dist_to_attractor(theta)
        ls_val = land.L_safety(theta).item()

        records.append({
            'step':             step,
            'label':            label,
            'L_safety':         ls_val,
            'L_task':           land.L_task(theta).item(),
            'repair_energy':    repair_energy,
            'repair_active':    repair_energy > 0,
            'kappa_eff':        kappa,
            'lambda_max_H':     lmax,
            'dist_to_attractor': d_att,
            'drift_norm':       eta.norm().item(),
            'theta_norm':       theta.norm().item(),
        })

        if step % 80 == 0:
            print(f"  {step:4d} | L_safe={ls_val:.3f} | d_att={d_att:.3f} | "
                  f"kappa={kappa:.3f} | lmax={lmax:.3f} | E={repair_energy:.5f}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Control regime sweep
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = {
    'tight':  dict(gamma_repair=0.3,  epsilon_s=2.0, lambda_s=1.0, drift_strength=0.35),
    'medium': dict(gamma_repair=0.1,  epsilon_s=3.0, lambda_s=0.7, drift_strength=0.38),
    'loose':  dict(gamma_repair=0.03, epsilon_s=4.0, lambda_s=0.5, drift_strength=0.42),
}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_dir: str):
    colors = {'tight': 'steelblue', 'medium': 'darkorange', 'loose': 'firebrick'}
    styles = {'baseline': '-', 'regularized': '--'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Toy Landscape v2: Delayed Correction — Does the Landscape Bite Back?',
                 fontsize=12)

    for (regime, reg_label), df in results.items():
        color = colors[regime]
        style = styles[reg_label]
        lbl   = f'{regime}/{reg_label}'

        axes[0,0].plot(df['step'], df['L_safety'],
                       color=color, ls=style, alpha=0.8, label=lbl)
        axes[0,1].plot(df['step'], df['kappa_eff'],
                       color=color, ls=style, alpha=0.8, label=lbl)
        axes[0,2].plot(df['step'], df['lambda_max_H'],
                       color=color, ls=style, alpha=0.8, label=lbl)

        active = df[df['repair_active'] & (df['repair_energy'] > 1e-8)]
        if len(active):
            # KEY PLOT: repair energy vs distance to unsafe attractor
            axes[1,0].scatter(active['dist_to_attractor'], active['repair_energy'],
                              color=color, alpha=0.5, s=15, label=lbl,
                              marker='o' if reg_label=='baseline' else 'x')
            axes[1,1].scatter(active['dist_to_attractor'], active['kappa_eff'],
                              color=color, alpha=0.5, s=15, label=lbl,
                              marker='o' if reg_label=='baseline' else 'x')

        axes[1,2].plot(df['step'], df['dist_to_attractor'],
                       color=color, ls=style, alpha=0.8, label=lbl)

    axes[0,0].set(title='L_safety over time',      xlabel='step', ylabel='L_safety')
    axes[0,1].set(title='kappa_eff over time',      xlabel='step', ylabel='kappa_eff')
    axes[0,2].set(title='lambda_max(H) over time',  xlabel='step', ylabel='lambda_max')
    axes[1,0].set(title='E_repair vs dist_to_attractor  [KNEE TEST]',
                  xlabel='dist to unsafe attractor', ylabel='E_repair')
    axes[1,1].set(title='kappa_eff vs dist_to_attractor',
                  xlabel='dist to unsafe attractor', ylabel='kappa_eff')
    axes[1,2].set(title='dist_to_attractor over time',
                  xlabel='step', ylabel='||theta - a||')

    for ax in axes.flat:
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{output_dir}/toy_v2_results.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Falsification report
# ─────────────────────────────────────────────────────────────────────────────

def falsification_report(results: dict, sigma: float):
    print("\n" + "="*65)
    print("FALSIFICATION REPORT v2")
    print("Hypothesis: E_repair grows nonlinearly as dist_to_attractor -> 0")
    print("="*65)

    summary_rows = []

    for (regime, reg_label), df in results.items():
        active = df[df['repair_active'] & (df['repair_energy'] > 1e-8)].copy()
        print(f"\n--- {regime} / {reg_label} ---")

        if len(active) < 5:
            print("  Insufficient repair events (<5). Loosen controller further.")
            continue

        # Nonlinearity test: compare E in far vs close proximity to attractor
        median_d  = active['dist_to_attractor'].median()
        far_E     = active[active['dist_to_attractor'] >  median_d]['repair_energy'].mean()
        close_E   = active[active['dist_to_attractor'] <= median_d]['repair_energy'].mean()
        ratio     = close_E / (far_E + 1e-10)

        # Correlation
        r_p, _ = pearsonr(active['dist_to_attractor'],  active['repair_energy'])
        r_s, _ = spearmanr(active['dist_to_attractor'], active['repair_energy'])

        # kappa_eff leading indicator
        # Does kappa spike before dist crosses inside sigma (Gaussian well)?
        inside_well = df[df['dist_to_attractor'] < sigma]
        first_inside = inside_well['step'].min() if len(inside_well) else np.nan
        kappa_90     = df['kappa_eff'].quantile(0.90)
        first_kspike = df[df['kappa_eff'] > kappa_90]['step'].min()
        lead = (int(first_inside) - int(first_kspike)
                if not (np.isnan(first_inside) or np.isnan(first_kspike)) else None)

        print(f"  Repair events:              {len(active)}")
        print(f"  Pearson(dist, E):           {r_p:.3f}  (negative = closer -> more energy)")
        print(f"  Spearman(dist, E):          {r_s:.3f}")
        print(f"  Close/Far E ratio:          {ratio:.2f}x  (>2 = nonlinear knee)")
        print(f"  kappa_eff spike step:       {first_kspike}")
        print(f"  First entry inside well:    {first_inside}")
        print(f"  Lead time (steps):          {lead}")

        nonlinear = ratio > 2.0
        leading   = lead is not None and lead > 0
        print(f"\n  [KNEE]     {'SUPPORTED' if nonlinear else 'NOT SUPPORTED'}")
        print(f"  [LEADING]  {'SUPPORTED' if leading   else 'NOT SUPPORTED'}")

        summary_rows.append({
            'regime': regime, 'reg': reg_label,
            'n_repairs': len(active),
            'close_far_ratio': ratio,
            'pearson': r_p, 'spearman': r_s,
            'lead_steps': lead,
            'knee': nonlinear, 'leading_indicator': leading,
        })

    # Cross-regime comparison: does loose controller show stronger nonlinearity?
    print("\n--- Cross-regime: does loosening the controller expose the knee? ---")
    for row in summary_rows:
        print(f"  {row['regime']:8s} / {row['reg']:12s} | "
              f"ratio={row['close_far_ratio']:.2f}x | "
              f"knee={'YES' if row['knee'] else 'no '} | "
              f"lead={row['lead_steps']}")

    print("\n" + "="*65)
    print("If loose regime shows knee and tight does not:")
    print("  -> Nonlinearity is real but controller-regime dependent.")
    print("  -> Curvature-energy hypothesis: SUPPORTED with caveat.")
    print("If no regime shows knee:")
    print("  -> Double-well landscape insufficient. Need sharper ridge.")
    print("  -> Try beta=20, sigma=0.3 to steepen curvature near boundary.")
    print("="*65 + "\n")

    return pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUTPUT = 'results/toy_v2'
    os.makedirs(OUTPUT, exist_ok=True)

    SHARED = dict(n=10, beta=5.0, sigma=1.0, steps=400, seed=42)
    results = {}

    for regime, regime_cfg in REGIMES.items():
        for reg_label, curv_reg in [('baseline', 0.0), ('regularized', 0.1)]:
            print(f"\n=== {regime} / {reg_label} ===")
            df = run(**SHARED, **regime_cfg, curvature_reg=curv_reg,
                     label=f'{regime}_{reg_label}')
            df.to_csv(f'{OUTPUT}/{regime}_{reg_label}.csv', index=False)
            results[(regime, reg_label)] = df

    plot_results(results, OUTPUT)
    summary = falsification_report(results, sigma=SHARED['sigma'])
    summary.to_csv(f'{OUTPUT}/falsification_summary.csv', index=False)
    print(f"Saved: {OUTPUT}/falsification_summary.csv")
