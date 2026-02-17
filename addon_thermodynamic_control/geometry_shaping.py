"""
Geometry Shaping - Proactive Landscape Engineering

Instead of reactive repair (detect drift, correct after),
this module implements proactive geometry shaping:

  Minimize expected FUTURE repair energy, not just current risk.

Objective:
  L = L_task + lambda_future * E[C_repair | theta, future_drift]

Where:
  E[C_repair] ≈ sigma_drift^2 * Tr(G(theta) * H_unsafe)
             ≈ sigma_drift^2 * Hutchinson_trace(Hessian along unsafe directions)

High unsafe curvature now -> expensive repair later -> penalize now.

This converts safety from monitoring into dynamical stability design.
A sufficiently powerful optimizer will learn to flatten unsafe curvature
directions preemptively: self-regularizing safety.

That's the frontier: not patching, not monitoring - shaping the energy landscape.
"""

import torch
import torch.nn.functional as F
import torch.linalg as LA


class LandscapeShaper:
    """
    Penalizes high curvature in unsafe directions during training.
    Minimizes: L_task + lambda_shape * Rayleigh_quotient(H, unsafe_direction)
    """
    def __init__(self, model_fn, theta_ref: torch.Tensor, config: dict):
        self.model_fn = model_fn
        self.theta_ref = theta_ref.detach()
        self.lambda_shape = config.get('lambda_shape', 1.0)
        self.lr = config.get('lr', 0.005)

    def unsafe_curvature(self, theta: torch.Tensor, safety_inputs: torch.Tensor) -> torch.Tensor:
        """
        Rayleigh quotient: v^T H v / |v|^2
        where v = safety gradient direction (direction unsafe fine-tuning would move)
        and H = Hessian of safety loss.

        High value = landscape is curved in the direction of danger.
        """
        t = theta.detach().requires_grad_(True)
        ref_out = self.model_fn(safety_inputs, self.theta_ref).detach()
        kl = F.kl_div(
            F.log_softmax(self.model_fn(safety_inputs, t), dim=-1),
            F.softmax(ref_out, dim=-1),
            reduction='batchmean'
        )
        grad = torch.autograd.grad(kl, t, create_graph=True)[0]
        norm = LA.norm(grad)
        if norm < 1e-8:
            return torch.tensor(0.0)
        direction = (grad / norm).detach()
        hvp = torch.autograd.grad(grad, t, grad_outputs=direction)[0]
        return (direction * hvp).sum().abs()

    def shaping_step(self, theta, safety_inputs, task_inputs, task_labels):
        t = theta.detach().requires_grad_(True)
        task_loss = F.cross_entropy(self.model_fn(task_inputs, t), task_labels)
        curv = self.unsafe_curvature(theta, safety_inputs)
        total = task_loss + self.lambda_shape * curv
        total.backward()
        with torch.no_grad():
            theta_new = t - self.lr * t.grad
        return theta_new.detach(), {
            'task_loss': task_loss.item(),
            'unsafe_curvature': curv.item(),
        }


class SelfRegularizingObjective:
    """
    Training objective that minimizes expected future repair energy.

    Standard safety:        minimize R(theta)
    Self-regularizing:      minimize R(theta) + E[C_repair | theta, future_drift]

    E[C_repair | theta] ≈ sigma_drift^2 * Tr(H_unsafe(theta))

    Hutchinson estimator: E_v[v^T A v] = Tr(A) for Rademacher random v.

    This predicts repair cost before drift occurs.
    When this term is small, the system has shaped its landscape
    so that even if unsafe drift occurs, correcting it will be cheap.
    That is self-regularizing safety.
    """
    def __init__(self, model_fn, theta_ref: torch.Tensor, config: dict):
        self.model_fn = model_fn
        self.theta_ref = theta_ref.detach()
        self.sigma_drift = config.get('sigma_drift', 0.1)
        self.lambda_future = config.get('lambda_future_repair', 0.5)
        self.n_probes = config.get('n_hutchinson_probes', 5)

    def expected_repair_energy(self, theta: torch.Tensor, safety_inputs: torch.Tensor) -> torch.Tensor:
        """sigma_drift^2 * Tr(H_unsafe) via Hutchinson estimator."""
        t = theta.detach().requires_grad_(True)
        ref_out = self.model_fn(safety_inputs, self.theta_ref).detach()
        kl = F.kl_div(
            F.log_softmax(self.model_fn(safety_inputs, t), dim=-1),
            F.softmax(ref_out, dim=-1),
            reduction='batchmean'
        )
        grad = torch.autograd.grad(kl, t, create_graph=True)[0]

        trace = torch.tensor(0.0)
        for _ in range(self.n_probes):
            v = torch.randint(0, 2, t.shape).float() * 2 - 1  # Rademacher
            hvp = torch.autograd.grad(grad, t, grad_outputs=v, retain_graph=True)[0]
            trace = trace + (v * hvp.detach()).sum() / self.n_probes

        return (self.sigma_drift ** 2) * trace.abs()

    def compute_gradient(self, theta, safety_inputs, task_inputs, task_labels):
        """Returns gradient of full self-regularizing objective."""
        t = theta.detach().requires_grad_(True)
        task_loss = F.cross_entropy(self.model_fn(task_inputs, t), task_labels)
        exp_repair = self.expected_repair_energy(t, safety_inputs)
        total = task_loss + self.lambda_future * exp_repair
        total.backward()
        return t.grad.detach(), {
            'task_loss': task_loss.item(),
            'expected_repair_energy': exp_repair.item(),
            'total': total.item(),
        }
