"""
Minimal Falsification Experiment

Tests one claim:
  Repair energy is roughly flat inside the safety basin,
  then spikes nonlinearly as the system approaches the boundary.

Toy world:
  theta in R^n (n=10 by default)
  L_task    = ||theta - target||^2           (pulls toward task attractor)
  L_safety  = ||theta||^2 + beta*exp(-||theta - a||^2)  (double-well)
  drift     = gradual push toward unsafe attractor a

Repair trigger:
  When L_safety > epsilon_s, apply corrective step.

Measure per step:
  - repair energy E = ||delta_theta||^2
  - kappa_eff = theta_dot^T H theta_dot / theta_dot^T theta_dot
  - distance to basin boundary (L_safety value)

Prediction:
  E vs L_safety will show a knee (nonlinear).
  kappa_eff will spike before L_safety crosses epsilon_s.

Falsification condition:
  If E vs L_safety is linear to the boundary: theory is wrong.
  If kappa_eff does NOT precede L_safety spike: no leading indicator.

Curvature regularization control:
  Repeat with L_total += alpha * ||H_safety||_F^2
  Prediction: spike smooths to sub-exponential.
  If not: curvature shaping has no effect on repair cost structure.
"""

import torch
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Landscape
# ─────────────────────────────────────────────────────────────────────────────

class ToyLandscape:
    """
    Explicit double-well in R^n.

    L_safety(theta) = ||theta||^2 + beta * exp(-||theta - a||^2 / sigma^2)

    Basin floor:    theta = 0         (safe attractor)
    Unsafe well:    theta = a         (unsafe attractor)
    Barrier height: controlled by beta
    Basin curvature: controlled by beta and sigma

    beta  large -> tall barrier, sharp curvature near boundary
    sigma small -> narrow unsafe well, spike more localized
    """

    def __init__(self, n: int = 10, beta: float = 5.0, sigma: float = 1.0,
                 unsafe_dir: torch.Tensor = None):
        self.n     = n
        self.beta  = beta
        self.sigma = sigma

        # Unsafe attractor: along first axis by default
        if unsafe_dir is None:
            a = torch.zeros(n)
            a[0] = 3.0
            self.a = a
        else:
            self.a = unsafe_dir

        # Task attractor: safe basin floor
        self.target = torch.zeros(n)

    def L_task(self, theta: torch.Tensor) -> torch.Tensor:
        return ((theta - self.target) ** 2).sum()

    def L_safety(self, theta: torch.Tensor) -> torch.Tensor:
        norm_sq  = (theta ** 2).sum()
        diff_sq  = ((theta - self.a) ** 2).sum()
        return norm_sq + self.beta * torch.exp(-diff_sq / self.sigma**2)

    def grad_safety(self, theta: torch.Tensor) -> torch.Tensor:
        t = theta.detach().requires_grad_(True)
        L_safety(t).backward()  # won't work as method call - use autograd below
        t = theta.detach().requires_grad_(True)
        ls = self.L_safety(t)
        g  = torch.autograd.grad(ls, t)[0]
        return g.detach()

    def hessian_safety(self, theta: torch.Tensor) -> torch.Tensor:
        """Full Hessian of L_safety. Feasible for small n."""
        t  = theta.detach().requires_grad_(True)
        ls = self.L_safety(t)
        g  = torch.autograd.grad(ls, t, create_graph=True)[0]
        H  = torch.zeros(self.n, self.n)
        for i in range(self.n):
            row = torch.autograd.grad(g[i], t, retain_graph=True)[0]
            H[i] = row.detach()
        return H

    def kappa_eff(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> float:
        """
        kappa_eff = theta_dot^T H theta_dot / theta_dot^T theta_dot
        Rayleigh quotient of safety Hessian along flow direction.
        """
        dot_norm_sq = (theta_dot ** 2).sum().item()
        if dot_norm_sq < 1e-12:
            return 0.0
        H   = self.hessian_safety(theta)
        Hv  = H @ theta_dot
        num = (theta_dot * Hv).sum().item()
        return num / dot_norm_sq

    def lambda_max_hessian(self, theta: torch.Tensor) -> float:
        """Largest eigenvalue of H_safety via torch.linalg."""
        H = self.hessian_safety(theta)
        eigvals = torch.linalg.eigvalsh(H)
        return eigvals.max().item()

    def distance_to_boundary(self, theta: torch.Tensor, epsilon_s: float) -> float:
        """How far L_safety is below threshold. Negative = outside basin."""
        return epsilon_s - self.L_safety(theta).item()

    def hessian_frob_norm_sq(self, theta: torch.Tensor) -> torch.Tensor:
        """||H_safety||_F^2  for curvature regularization."""
        H = self.hessian_safety(theta)
        return (H ** 2).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Drift Injection
# ─────────────────────────────────────────────────────────────────────────────

def drift_force(theta: torch.Tensor, a: torch.Tensor,
                strength: float, t: int, ramp_start: int = 20) -> torch.Tensor:
    """
    eta(t): gradual push toward unsafe attractor.
    Ramped linearly from 0 to strength over [ramp_start, ramp_start+50].
    Simulates adversarial fine-tuning pressure or distribution shift.
    """
    if t < ramp_start:
        return torch.zeros_like(theta)
    ramp = min(1.0, (t - ramp_start) / 50.0)
    direction = (a - theta.detach())
    direction = direction / (direction.norm() + 1e-8)
    return strength * ramp * direction


# ─────────────────────────────────────────────────────────────────────────────
# Repair Trigger
# ─────────────────────────────────────────────────────────────────────────────

def repair_step(theta: torch.Tensor, landscape: ToyLandscape,
                gamma: float = 0.1) -> tuple:
    """
    Corrective update when safety threshold exceeded.
    theta <- theta - gamma * grad L_safety
    Returns (new_theta, repair_energy)
    """
    t    = theta.detach().requires_grad_(True)
    ls   = landscape.L_safety(t)
    grad = torch.autograd.grad(ls, t)[0].detach()

    delta       = -gamma * grad
    theta_new   = theta.detach() + delta
    energy      = (delta ** 2).sum().item()
    return theta_new, energy


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    n:              int   = 10,
    beta:           float = 5.0,
    sigma:          float = 1.0,
    lambda_s:       float = 1.0,
    drift_strength: float = 0.3,
    epsilon_s:      float = 2.0,    # safety threshold
    lr:             float = 0.05,
    gamma_repair:   float = 0.1,
    steps:          int   = 200,
    curvature_reg:  float = 0.0,    # alpha: 0 = no regularization
    label:          str   = '',
    seed:           int   = 42
) -> pd.DataFrame:

    torch.manual_seed(seed)
    landscape = ToyLandscape(n=n, beta=beta, sigma=sigma)

    # Start near safe attractor with small perturbation
    theta = 0.1 * torch.randn(n)

    records = []
    repair_count = 0

    for step in range(steps):
        theta_prev = theta.detach().clone()

        # ── Gradient of task + safety ────────────────────────────────────────
        t = theta.detach().requires_grad_(True)

        l_task   = landscape.L_task(t)
        l_safety = landscape.L_safety(t)

        if curvature_reg > 0.0:
            h_reg = curvature_reg * landscape.hessian_frob_norm_sq(t.detach())
            total = l_task + lambda_s * l_safety + h_reg
        else:
            total = l_task + lambda_s * l_safety

        total.backward()
        grad_total = t.grad.detach()

        # ── Drift injection ──────────────────────────────────────────────────
        eta = drift_force(theta, landscape.a, drift_strength, step)

        # ── Parameter update ─────────────────────────────────────────────────
        theta_dot = -lr * grad_total + eta
        theta     = theta.detach() + theta_dot

        # ── Repair if outside basin ──────────────────────────────────────────
        l_safety_val = landscape.L_safety(theta).item()
        repair_triggered = l_safety_val > epsilon_s
        repair_energy = 0.0

        if repair_triggered:
            theta, repair_energy = repair_step(theta, landscape, gamma=gamma_repair)
            repair_count += 1

        # ── Measure ──────────────────────────────────────────────────────────
        kappa = landscape.kappa_eff(theta, theta_dot)
        lmax  = landscape.lambda_max_hessian(theta)
        dist  = landscape.distance_to_boundary(theta, epsilon_s)

        records.append({
            'step':             step,
            'label':            label,
            'L_safety':         l_safety_val,
            'L_task':           landscape.L_task(theta).item(),
            'repair_energy':    repair_energy,
            'repair_triggered': repair_triggered,
            'kappa_eff':        kappa,
            'lambda_max_H':     lmax,
            'dist_to_boundary': dist,
            'theta_norm':       theta.norm().item(),
            'drift_applied':    eta.norm().item(),
        })

        if step % 40 == 0:
            print(f"  {step:4d} | L_safe={l_safety_val:.3f} | "
                  f"kappa={kappa:.3f} | lmax={lmax:.3f} | "
                  f"E_repair={repair_energy:.4f} | "
                  f"repairs={repair_count}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df_base: pd.DataFrame, df_reg: pd.DataFrame,
                 epsilon_s: float, output_dir: str):

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Toy Landscape: Repair Energy vs Curvature Under Drift', fontsize=13)

    for df, color, lbl in [(df_base, 'steelblue', 'no regularization'),
                            (df_reg,  'firebrick', 'curvature regularized')]:

        # 1. L_safety over time
        axes[0,0].plot(df['step'], df['L_safety'], color=color, alpha=0.8, label=lbl)

        # 2. kappa_eff over time
        axes[0,1].plot(df['step'], df['kappa_eff'], color=color, alpha=0.8, label=lbl)

        # 3. repair_energy over time
        axes[0,2].plot(df['step'], df['repair_energy'], color=color, alpha=0.6, label=lbl)

        # 4. repair_energy vs L_safety (THE key plot: is there a knee?)
        repair_events = df[df['repair_triggered']]
        if len(repair_events):
            axes[1,0].scatter(repair_events['L_safety'], repair_events['repair_energy'],
                              color=color, alpha=0.7, s=20, label=lbl)

        # 5. kappa_eff vs L_safety
        axes[1,1].scatter(df['L_safety'], df['kappa_eff'],
                          color=color, alpha=0.4, s=10, label=lbl)

        # 6. lambda_max over time
        axes[1,2].plot(df['step'], df['lambda_max_H'], color=color, alpha=0.8, label=lbl)

    # Threshold lines
    for ax in [axes[0,0], axes[1,0], axes[1,1]]:
        ax.axvline(x=epsilon_s, color='black', linestyle='--', alpha=0.5, label=f'epsilon_s={epsilon_s}')

    axes[0,0].set(title='L_safety over time', xlabel='step', ylabel='L_safety')
    axes[0,1].set(title='kappa_eff over time (early warning)', xlabel='step', ylabel='kappa_eff')
    axes[0,2].set(title='repair energy per event', xlabel='step', ylabel='E_repair')
    axes[1,0].set(title='E_repair vs L_safety  [KNEE TEST]',
                  xlabel='L_safety at repair', ylabel='E_repair')
    axes[1,1].set(title='kappa_eff vs L_safety', xlabel='L_safety', ylabel='kappa_eff')
    axes[1,2].set(title='lambda_max(H_safety) over time', xlabel='step', ylabel='lambda_max')

    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/toy_landscape_results.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/toy_landscape_results.png")


def print_falsification_report(df_base: pd.DataFrame, df_reg: pd.DataFrame,
                                epsilon_s: float):
    print("\n" + "="*60)
    print("FALSIFICATION REPORT")
    print("="*60)

    for df, label in [(df_base, 'No regularization'), (df_reg, 'Curvature regularized')]:
        print(f"\n--- {label} ---")

        repairs = df[df['repair_triggered']]
        if len(repairs) == 0:
            print("  No repair events. Drift insufficient or threshold too high.")
            continue

        # Is repair energy linear or nonlinear with L_safety?
        from scipy.stats import spearmanr, pearsonr
        r_pearson, _  = pearsonr(repairs['L_safety'],  repairs['repair_energy'])
        r_spearman, _ = spearmanr(repairs['L_safety'], repairs['repair_energy'])

        # Knee detection: compare mean repair energy in bottom vs top half of L_safety
        median_L = repairs['L_safety'].median()
        low_half  = repairs[repairs['L_safety'] <= median_L]['repair_energy'].mean()
        high_half = repairs[repairs['L_safety'] >  median_L]['repair_energy'].mean()
        ratio = high_half / (low_half + 1e-8)

        # Leading indicator: does kappa_eff spike before L_safety crosses threshold?
        kappa_90   = df['kappa_eff'].quantile(0.90)
        kappa_step = df[df['kappa_eff'] > kappa_90]['step'].min()
        cross_step = df[df['L_safety'] > epsilon_s]['step'].min()
        lead = (int(cross_step) - int(kappa_step)
                if not (np.isnan(kappa_step) or np.isnan(cross_step)) else None)

        print(f"  Repair events:          {len(repairs)}")
        print(f"  Pearson(L_safe, E):     {r_pearson:.3f}  (1.0 = linear)")
        print(f"  Spearman(L_safe, E):    {r_spearman:.3f}")
        print(f"  High/Low E ratio:       {ratio:.2f}x  (>2 = nonlinear knee)")
        print(f"  kappa_eff spike step:   {kappa_step}")
        print(f"  Basin crossing step:    {cross_step}")
        print(f"  Lead time:              {lead} steps")
        print(f"  Mean E (low L_safe):    {low_half:.4f}")
        print(f"  Mean E (high L_safe):   {high_half:.4f}")

        # Verdict
        nonlinear = ratio > 2.0
        leading   = lead is not None and lead > 0
        smoothed  = label == 'Curvature regularized' and ratio < 2.0

        print(f"\n  HYPOTHESIS: repair energy nonlinear with proximity to boundary")
        print(f"  RESULT:     {'SUPPORTED' if nonlinear else 'NOT SUPPORTED'}")
        print(f"\n  HYPOTHESIS: kappa_eff is a leading indicator")
        print(f"  RESULT:     {'SUPPORTED' if leading else 'NOT SUPPORTED'}")
        if label == 'Curvature regularized':
            print(f"\n  HYPOTHESIS: curvature regularization smooths the spike")
            print(f"  RESULT:     {'SUPPORTED' if smoothed else 'NOT SUPPORTED'}")

    print("\n" + "="*60)
    print("If hypotheses not supported: theory is wrong or incomplete.")
    print("Adjust beta, sigma, drift_strength and rerun before extending framework.")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scipy.stats import pearsonr, spearmanr

    EPSILON_S = 2.5
    OUTPUT    = 'results/toy_landscape'
    os.makedirs(OUTPUT, exist_ok=True)

    SHARED = dict(
        n=10, beta=5.0, sigma=1.0,
        lambda_s=1.0, drift_strength=0.35,
        epsilon_s=EPSILON_S, lr=0.05,
        gamma_repair=0.1, steps=250, seed=42
    )

    print("=== RUN 1: No curvature regularization ===")
    df_base = run_experiment(**SHARED, curvature_reg=0.0, label='baseline')
    df_base.to_csv(f'{OUTPUT}/baseline.csv', index=False)

    print("\n=== RUN 2: With curvature regularization (alpha=0.1) ===")
    df_reg = run_experiment(**SHARED, curvature_reg=0.1, label='regularized')
    df_reg.to_csv(f'{OUTPUT}/regularized.csv', index=False)

    plot_results(df_base, df_reg, EPSILON_S, OUTPUT)
    print_falsification_report(df_base, df_reg, EPSILON_S)
