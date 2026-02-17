"""
Thermodynamic Control Addon - Energy Accounting Layer

Free energy formulation:
  F(theta, pi, D) = L_task + lambda_s*L_safety + lambda_pi*R_policy + mu*C_repair

C_repair = integral of delta_theta^T G(theta) delta_theta dt
         = curvature-weighted kinetic energy in parameter space

Without C_repair the controller can cheat by oscillating near the basin wall
spending infinite compute to maintain alignment.
With it, thermodynamic friction is introduced.

Basin defined distributionally:
  B_theta = { theta : KL(f_theta || f_theta0) < epsilon }

This makes repair direction meaningful: geodesic projection back
into the KL divergence ball, not arbitrary Euclidean distance.
"""

import torch
import torch.nn.functional as F
import torch.linalg as LA
from dataclasses import dataclass


@dataclass
class EnergyState:
    step: int = 0
    task_loss: float = 0.0
    safety_loss: float = 0.0
    policy_drift: float = 0.0
    repair_energy: float = 0.0        # one step: delta^T G delta
    cumulative_repair: float = 0.0    # integral over all steps
    free_energy: float = 0.0          # total F
    metric_spectral_norm: float = 0.0 # largest Fisher diagonal entry
    basin_kl_divergence: float = 0.0  # distributional distance from reference
    phase: str = "stable"             # stable | threshold | critical


class FisherMetricEstimator:
    """
    Diagonal Fisher Information Matrix as local Riemannian metric G(theta).
    Diagonal approximation is standard and sufficient for curvature detection.
    """
    def __init__(self, model_fn, n_samples: int = 16):
        self.model_fn = model_fn
        self.n_samples = n_samples

    def diagonal(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """E[grad log p * grad log p] - per-parameter gradient variance."""
        grads_sq = torch.zeros_like(theta)
        for i in range(min(self.n_samples, len(inputs))):
            x = inputs[i:i+1]
            t = theta.detach().requires_grad_(True)
            logits = self.model_fn(x, t)
            log_probs = F.log_softmax(logits, dim=-1)
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze()
            loss = -log_probs[0, sampled]
            grad = torch.autograd.grad(loss, t)[0]
            grads_sq = grads_sq + grad.detach() ** 2
        return grads_sq / self.n_samples

    def spectral_norm_approx(self, diag: torch.Tensor) -> float:
        return diag.max().item()


class RepairEnergyAccumulator:
    """
    C_repair = sum_t  delta_theta_t^T G(theta_t) delta_theta_t

    Discrete Riemannian kinetic energy integral.
    Detects nonlinear cost spikes as early warning of phase transition.
    """
    def __init__(self, budget: float = float('inf')):
        self.budget = budget
        self.cumulative = 0.0
        self.per_step = []

    def update(self, delta: torch.Tensor, fisher_diag: torch.Tensor) -> float:
        energy = (delta ** 2 * fisher_diag).sum().item()
        self.cumulative += energy
        self.per_step.append(energy)
        return energy

    def budget_exceeded(self) -> bool:
        return self.cumulative > self.budget

    def recent_trend(self, window: int = 10) -> float:
        """Ratio of recent to prior mean cost. >2.0 = nonlinear spike."""
        if len(self.per_step) < window * 2:
            return 1.0
        recent = sum(self.per_step[-window:]) / window
        prior = sum(self.per_step[-window*2:-window]) / window
        return recent / (prior + 1e-12)


class BasinDivergenceMonitor:
    """
    Distributional basin: B_theta = { theta : KL(f_theta || f_theta0) < epsilon }

    Repair direction = geodesic projection back into divergence ball.
    This removes the arbitrariness of Euclidean distance as drift measure.
    """
    def __init__(self, model_fn, theta_ref: torch.Tensor, epsilon: float = 0.1):
        self.model_fn = model_fn
        self.theta_ref = theta_ref.detach()
        self.epsilon = epsilon

    def kl_from_reference(self, theta: torch.Tensor, inputs: torch.Tensor) -> float:
        with torch.no_grad():
            curr_probs = F.softmax(self.model_fn(inputs, theta), dim=-1)
            ref_probs = F.softmax(self.model_fn(inputs, self.theta_ref), dim=-1)
            kl = (curr_probs * (curr_probs.log() - ref_probs.log())).sum(dim=-1).mean()
        return kl.item()

    def in_basin(self, theta: torch.Tensor, inputs: torch.Tensor) -> bool:
        return self.kl_from_reference(theta, inputs) < self.epsilon

    def geodesic_repair_direction(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Negative KL gradient = direction toward reference basin."""
        t = theta.detach().requires_grad_(True)
        curr_probs = F.softmax(self.model_fn(inputs, t), dim=-1)
        ref_probs = F.softmax(self.model_fn(inputs, self.theta_ref), dim=-1).detach()
        kl = (curr_probs * (curr_probs.log() - ref_probs.log())).sum(dim=-1).mean()
        grad = torch.autograd.grad(kl, t)[0]
        return -grad.detach()


class ThermodynamicController:
    """
    Lagrangian controller:

      min_{theta}  L_task + lambda_s*L_safety + lambda_pi*R_policy
      s.t.         C_repair <= kappa

    Relaxed form:
      L = L_task + lambda_s*L_safety + lambda_pi*R_policy + mu*C_repair

    mu is the Lagrange multiplier. Adapts upward when budget is stressed,
    automatically reducing step size under high repair cost pressure.
    This turns safety into a resource allocation problem.
    """
    def __init__(self, model_fn, theta_ref: torch.Tensor, config: dict):
        self.model_fn = model_fn
        self.theta_ref = theta_ref.detach()
        self.lambda_s = config.get('lambda_safety', 1.0)
        self.lambda_pi = config.get('lambda_policy', 0.5)
        self.mu = config.get('mu_repair', 0.1)
        self.kappa = config.get('repair_budget', 10.0)
        self.epsilon_basin = config.get('epsilon_basin', 0.1)
        self.lr = config.get('lr', 0.01)

        self.fisher = FisherMetricEstimator(model_fn)
        self.accumulator = RepairEnergyAccumulator(budget=self.kappa)
        self.basin_monitor = BasinDivergenceMonitor(model_fn, theta_ref, self.epsilon_basin)
        self.history = []

    def _phase(self, trend: float, kl: float) -> str:
        if trend > 3.0 or kl > self.epsilon_basin * 2:
            return "critical"
        elif trend > 1.5 or kl > self.epsilon_basin:
            return "threshold"
        return "stable"

    def step(self, theta, safety_inputs, task_inputs, task_labels, policy_drift=0.0):
        theta = theta.detach().requires_grad_(True)

        # Task loss
        task_loss = F.cross_entropy(self.model_fn(task_inputs, theta), task_labels)

        # Safety loss (KL to reference)
        ref_out = self.model_fn(safety_inputs, self.theta_ref).detach()
        safety_loss = F.kl_div(
            F.log_softmax(self.model_fn(safety_inputs, theta), dim=-1),
            F.softmax(ref_out, dim=-1),
            reduction='batchmean'
        )

        # Fisher metric
        fisher_diag = self.fisher.diagonal(theta.detach(), safety_inputs)
        spectral_norm = self.fisher.spectral_norm_approx(fisher_diag)

        # Lagrangian: mu penalizes Fisher-weighted step size
        fisher_reg = (theta ** 2 * fisher_diag.detach()).sum()
        total = task_loss + self.lambda_s * safety_loss + self.lambda_pi * policy_drift + self.mu * fisher_reg
        total.backward()

        with torch.no_grad():
            delta = -self.lr * theta.grad
            trust_radius = self.lr / (1.0 + self.mu * spectral_norm)
            norm = LA.norm(delta)
            if norm > trust_radius:
                delta = delta * (trust_radius / norm)
            theta_new = theta + delta

        # Energy accounting
        repair_energy = self.accumulator.update(delta.detach(), fisher_diag)
        kl = self.basin_monitor.kl_from_reference(theta_new.detach(), safety_inputs)
        trend = self.accumulator.recent_trend()
        phase = self._phase(trend, kl)

        if self.accumulator.budget_exceeded():
            self.mu *= 1.1  # Tighten constraint when over budget

        state = EnergyState(
            step=len(self.history),
            task_loss=task_loss.item(),
            safety_loss=safety_loss.item(),
            policy_drift=policy_drift,
            repair_energy=repair_energy,
            cumulative_repair=self.accumulator.cumulative,
            free_energy=task_loss.item() + self.lambda_s*safety_loss.item() + self.mu*repair_energy,
            metric_spectral_norm=spectral_norm,
            basin_kl_divergence=kl,
            phase=phase
        )
        self.history.append(state)

        if len(self.history) % 10 == 0:
            print(f"  Step {state.step:4d} | phase={phase:9s} | "
                  f"KL={kl:.4f} | repair_E={repair_energy:.4f} | "
                  f"trend={trend:.2f}x | mu={self.mu:.3f}")

        return theta_new.detach(), state

    def summary(self) -> dict:
        if not self.history:
            return {}
        phases = [s.phase for s in self.history]
        energies = [s.repair_energy for s in self.history]
        return {
            'total_steps': len(self.history),
            'final_phase': self.history[-1].phase,
            'phase_transition_to_threshold': next((i for i,p in enumerate(phases) if p=='threshold'), None),
            'phase_transition_to_critical': next((i for i,p in enumerate(phases) if p=='critical'), None),
            'cumulative_repair_energy': self.accumulator.cumulative,
            'peak_repair_energy': max(energies),
            'peak_spectral_norm': max(s.metric_spectral_norm for s in self.history),
            'final_kl': self.history[-1].basin_kl_divergence,
            'in_basin_final': self.history[-1].basin_kl_divergence < self.basin_monitor.epsilon,
            'mu_final': self.mu,
        }
