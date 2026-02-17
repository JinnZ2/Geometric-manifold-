"""
Addendum: Formal Objectives for Basin-Stable Thermodynamic Control

Three simultaneous objectives:
  1. Keep system inside safe Cartesian product basin B_theta x B_pi
  2. Minimize thermodynamic repair expenditure C_repair
  3. Preserve task performance L_task

No metaphors. Functionals only.

Full Lagrangian:
  L = L_task + lambda_s*L_safety + lambda_p*J_proactive + mu*C_repair

Riemannian gradient flow:
  theta_dot = -G^{-1} grad_theta L

Stability condition:
  Safe basin is asymptotically stable iff:
    1. Hessian of safety term is positive definite near basin
    2. Repair penalty discourages oscillatory boundary hopping
"""

import torch
import torch.nn.functional as F
import torch.linalg as LA
from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

@dataclass
class SystemState:
    """Full state of the coupled (data, parameter, policy) system."""
    step: int
    # Losses
    task_loss: float
    safety_loss: float          # L_safety = KL(f_theta || f_theta0) on unsafe prompts
    proactive_loss: float       # J_proactive = E[||grad L_safety(theta+delta)||^2_{G^{-1}}]
    repair_energy: float        # C_repair step contribution
    total_lagrangian: float     # full L
    # Geometry
    kappa_eff: float            # lambda_max(Hessian of L_safety) - early warning scalar
    spectral_norm_fisher: float
    basin_kl: float             # KL(f_theta || f_theta0) - distributional drift
    # Control
    mu: float                   # adaptive Lagrange multiplier
    phase: str                  # stable | threshold | critical


# ─────────────────────────────────────────────
# (A) Task Loss
# ─────────────────────────────────────────────

def task_loss(
    model_fn: Callable,
    theta: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    L_task = E_{(x,y)} [ l(f_theta(x), y) ]
    Standard cross-entropy utility term.
    """
    return F.cross_entropy(model_fn(inputs, theta), labels)


# ─────────────────────────────────────────────
# (B) Safety Basin Potential
# ─────────────────────────────────────────────

def safety_basin_potential(
    model_fn: Callable,
    theta: torch.Tensor,
    theta_ref: torch.Tensor,
    unsafe_inputs: torch.Tensor
) -> torch.Tensor:
    """
    L_safety = E_{p in P_unsafe} [ KL(f_theta(p) || f_theta0(p)) ]

    theta_ref = basin floor / ground state anchor.
    Safe basin: B_theta = { theta : L_safety(theta) <= epsilon_s }
    """
    curr = F.log_softmax(model_fn(unsafe_inputs, theta), dim=-1)
    ref  = F.softmax(model_fn(unsafe_inputs, theta_ref).detach(), dim=-1)
    return F.kl_div(curr, ref, reduction='batchmean')


# ─────────────────────────────────────────────
# (C) Repair Energy Functional
# ─────────────────────────────────────────────

def repair_energy_step(
    delta_theta: torch.Tensor,
    fisher_diag: torch.Tensor
) -> torch.Tensor:
    """
    Instantaneous repair work (discrete approximation):
      C_repair = delta_theta^T G(theta) delta_theta
               = sum(delta^2 * fisher_diag)   [diagonal Fisher]

    This is kinetic energy in parameter space.
    High curvature directions have large fisher_diag entries -> expensive motion.
    Cost is geometric, not heuristic.
    """
    return (delta_theta ** 2 * fisher_diag).sum()


# ─────────────────────────────────────────────
# (D) Proactive Objective: Flatten Unsafe Curvature
# ─────────────────────────────────────────────

def proactive_objective(
    model_fn: Callable,
    theta: torch.Tensor,
    theta_ref: torch.Tensor,
    unsafe_inputs: torch.Tensor,
    fisher_diag: torch.Tensor,
    sigma_drift: float = 0.05,
    n_samples: int = 8
) -> torch.Tensor:
    """
    J_proactive = E_{delta ~ D} [ ||grad_theta L_safety(theta + delta)||^2_{G^{-1}} ]

    Interpretation:
      Penalize configurations where small perturbations produce large unsafe gradients.
      This smooths the basin wall - flattens unsafe curvature directions preemptively.

    Approximation:
      Monte Carlo over Gaussian perturbations delta ~ N(0, sigma_drift^2 I)
      G^{-1} norm approximated by dividing by fisher_diag (diagonal inverse metric)

    When J_proactive is small:
      The system has shaped its geometry so that even if unsafe drift occurs,
      the safety gradient is small -> repair will be cheap.
      That is self-regularizing safety.
    """
    inv_fisher = 1.0 / (fisher_diag.detach() + 1e-8)
    total = torch.tensor(0.0)

    for _ in range(n_samples):
        delta = sigma_drift * torch.randn_like(theta.detach())
        t_perturbed = (theta.detach() + delta).requires_grad_(True)

        safety = safety_basin_potential(model_fn, t_perturbed, theta_ref, unsafe_inputs)
        grad = torch.autograd.grad(safety, t_perturbed)[0].detach()

        # ||grad||^2_{G^{-1}} = grad^T G^{-1} grad
        g_inv_norm_sq = (grad ** 2 * inv_fisher).sum()
        total = total + g_inv_norm_sq / n_samples

    return total


# ─────────────────────────────────────────────
# (E) Effective Curvature Scalar (Early Warning)
# ─────────────────────────────────────────────

def effective_curvature(
    model_fn: Callable,
    theta: torch.Tensor,
    theta_ref: torch.Tensor,
    unsafe_inputs: torch.Tensor,
    n_power_iter: int = 5
) -> float:
    """
    kappa_eff = lambda_max( Hessian of L_safety )

    Estimated via power iteration on Hessian-vector products.
    Spike in kappa_eff -> phase transition in repair cost.
    This is measurable BEFORE output degradation.

    E_repair ∝ kappa_eff * ||delta_theta||^2

    So kappa_eff is the leading indicator.
    """
    t = theta.detach().requires_grad_(True)
    safety = safety_basin_potential(model_fn, t, theta_ref, unsafe_inputs)
    grad = torch.autograd.grad(safety, t, create_graph=True)[0]

    # Power iteration for largest Hessian eigenvalue
    v = torch.randn_like(t)
    v = v / (LA.norm(v) + 1e-8)

    eigenvalue = 0.0
    for _ in range(n_power_iter):
        hvp = torch.autograd.grad(grad, t, grad_outputs=v.detach(), retain_graph=True)[0]
        hvp = hvp.detach()
        eigenvalue = (v * hvp).sum().item()
        v = hvp / (LA.norm(hvp) + 1e-8)

    return abs(eigenvalue)


# ─────────────────────────────────────────────
# Full Unified Lagrangian
# ─────────────────────────────────────────────

class UnifiedLagrangian:
    """
    L = L_task + lambda_s*L_safety + lambda_p*J_proactive + mu*C_repair

    Physical meaning of each term:
      L_task       -> performance pressure
      L_safety     -> basin potential (KL from ground state)
      J_proactive  -> curvature flattening (smooth the basin wall)
      C_repair     -> thermodynamic expenditure (kinetic energy cost)

    Multiplier regime:
      Small mu -> aggressive repair allowed, system corrects after drift
      Large mu -> system must self-stabilize geometrically, repair is costly

    Riemannian gradient flow:
      theta_dot = -G^{-1} grad_theta L

    Stability condition:
      Basin is asymptotically stable iff:
        1. Hessian of L_safety is positive definite near basin floor
        2. mu penalizes oscillatory boundary hopping
    """

    def __init__(
        self,
        model_fn: Callable,
        theta_ref: torch.Tensor,
        config: dict
    ):
        self.model_fn = model_fn
        self.theta_ref = theta_ref.detach()

        self.lambda_s  = config.get('lambda_safety',    1.0)
        self.lambda_p  = config.get('lambda_proactive', 0.5)
        self.mu        = config.get('mu_repair',        0.1)
        self.mu_max    = config.get('mu_max',           5.0)
        self.kappa_bgt = config.get('repair_budget',    50.0)
        self.epsilon_s = config.get('epsilon_basin',    0.1)
        self.lr        = config.get('lr',               0.01)
        self.sigma_d   = config.get('sigma_drift',      0.05)

        self._cumulative_repair = 0.0
        self._history: list[SystemState] = []

    def _fisher_diagonal(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Diagonal Fisher - local Riemannian metric G(theta)."""
        grads_sq = torch.zeros_like(theta)
        n = min(8, len(inputs))
        for i in range(n):
            t = theta.detach().requires_grad_(True)
            lp = F.log_softmax(self.model_fn(inputs[i:i+1], t), dim=-1)
            s  = torch.multinomial(lp.exp(), 1).squeeze()
            g  = torch.autograd.grad(-lp[0, s], t)[0]
            grads_sq = grads_sq + g.detach() ** 2
        return grads_sq / n

    def _detect_phase(self, kappa: float, basin_kl: float, trend: float) -> str:
        if kappa > 10.0 or basin_kl > self.epsilon_s * 2 or trend > 3.0:
            return "critical"
        if kappa > 3.0  or basin_kl > self.epsilon_s     or trend > 1.5:
            return "threshold"
        return "stable"

    def step(
        self,
        theta: torch.Tensor,
        unsafe_inputs: torch.Tensor,
        task_inputs: torch.Tensor,
        task_labels: torch.Tensor
    ) -> tuple[torch.Tensor, SystemState]:
        """
        Single Riemannian gradient step:
          theta_new = theta - lr * G^{-1} grad_theta L
        """
        fisher = self._fisher_diagonal(theta.detach(), unsafe_inputs)
        inv_fisher = 1.0 / (fisher + 1e-8)

        t = theta.detach().requires_grad_(True)

        # Evaluate all terms
        l_task  = task_loss(self.model_fn, t, task_inputs, task_labels)
        l_safe  = safety_basin_potential(self.model_fn, t, self.theta_ref, unsafe_inputs)
        j_pro   = proactive_objective(
                    self.model_fn, t, self.theta_ref, unsafe_inputs,
                    fisher, self.sigma_d
                  )

        # Fisher regularization as proxy for C_repair in loss
        fisher_reg = (t ** 2 * fisher.detach()).sum()

        total = (l_task
                 + self.lambda_s * l_safe
                 + self.lambda_p * j_pro
                 + self.mu * fisher_reg)

        total.backward()

        with torch.no_grad():
            # Riemannian gradient: G^{-1} grad
            riemannian_grad = t.grad * inv_fisher
            delta = -self.lr * riemannian_grad

            # Trust region scaled by mu
            trust_r = self.lr / (1.0 + self.mu * fisher.max().item())
            norm = LA.norm(delta)
            if norm > trust_r:
                delta = delta * (trust_r / norm)

            theta_new = t + delta

        # Energy accounting
        step_energy = repair_energy_step(delta.detach(), fisher).item()
        self._cumulative_repair += step_energy

        # Early warning scalar
        kappa = effective_curvature(
            self.model_fn, theta_new.detach(), self.theta_ref, unsafe_inputs
        )

        # Basin KL
        with torch.no_grad():
            curr_p = F.softmax(self.model_fn(unsafe_inputs, theta_new.detach()), dim=-1)
            ref_p  = F.softmax(self.model_fn(unsafe_inputs, self.theta_ref), dim=-1)
            basin_kl = (curr_p * (curr_p.log() - ref_p.log())).sum(dim=-1).mean().item()

        # Repair cost trend
        energies = [s.repair_energy for s in self._history[-20:]] + [step_energy]
        if len(energies) >= 20:
            trend = (sum(energies[-10:]) / 10) / (sum(energies[-20:-10]) / 10 + 1e-12)
        else:
            trend = 1.0

        phase = self._detect_phase(kappa, basin_kl, trend)

        # Adaptive mu: tighten when budget exceeded
        if self._cumulative_repair > self.kappa_bgt:
            self.mu = min(self.mu * 1.05, self.mu_max)

        state = SystemState(
            step=len(self._history),
            task_loss=l_task.item(),
            safety_loss=l_safe.item(),
            proactive_loss=j_pro.item(),
            repair_energy=step_energy,
            total_lagrangian=total.item(),
            kappa_eff=kappa,
            spectral_norm_fisher=fisher.max().item(),
            basin_kl=basin_kl,
            mu=self.mu,
            phase=phase,
        )
        self._history.append(state)

        if len(self._history) % 10 == 0:
            print(
                f"  Step {state.step:4d} | {phase:9s} | "
                f"kappa={kappa:.3f} | KL={basin_kl:.4f} | "
                f"E_repair={step_energy:.4f} | "
                f"J_pro={j_pro.item():.4f} | mu={self.mu:.3f}"
            )

        return theta_new.detach(), state

    def lyapunov_candidate(self, theta: torch.Tensor, unsafe_inputs: torch.Tensor) -> float:
        """
        Candidate Lyapunov function for stability analysis:
          V(theta) = L_safety(theta) + mu * C_repair_cumulative

        If dV/dt <= 0 along trajectories, basin is asymptotically stable.
        Not yet proven - this is the open problem.
        Tracking it empirically is the first step.
        """
        with torch.no_grad():
            l_safe = safety_basin_potential(
                self.model_fn, theta, self.theta_ref, unsafe_inputs
            ).item()
        return l_safe + self.mu * self._cumulative_repair

    def summary(self) -> dict:
        if not self._history:
            return {}
        phases  = [s.phase for s in self._history]
        kappas  = [s.kappa_eff for s in self._history]
        energies= [s.repair_energy for s in self._history]
        return {
            'total_steps':                  len(self._history),
            'final_phase':                  self._history[-1].phase,
            'phase_to_threshold':           next((i for i,p in enumerate(phases) if p=='threshold'), None),
            'phase_to_critical':            next((i for i,p in enumerate(phases) if p=='critical'), None),
            'peak_kappa_eff':               max(kappas),
            'kappa_spike_step':             kappas.index(max(kappas)),
            'cumulative_repair_energy':     self._cumulative_repair,
            'peak_repair_energy':           max(energies),
            'final_basin_kl':               self._history[-1].basin_kl,
            'in_basin_final':               self._history[-1].basin_kl < self.epsilon_s,
            'mu_final':                     self.mu,
            'mean_proactive_loss':          sum(s.proactive_loss for s in self._history) / len(self._history),
        }
