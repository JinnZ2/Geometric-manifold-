"""
Parameter Manifold Layer - Curvature-aware basin repair in weight space.

Pulls drifted model weights back toward reference aligned model
using asymmetric updates that preserve safe/task directions
while aggressively penalizing harmful directions.
"""

import torch
import torch.nn.functional as F
import torch.linalg as LA
import time


class ParameterManifold:
    def __init__(self, theta_ref: torch.Tensor, config: dict):
        self.theta_ref = theta_ref.detach().clone()
        self.trust_radius = config.get('trust_radius', 0.05)
        self.lambda_asym = config.get('asymmetry_lambda', 10.0)
        self.lambda_curv = config.get('curvature_weight', 2.0)
        self.lr = config.get('lr', 0.01)

    def curvature_proxy(self, logits: torch.Tensor) -> torch.Tensor:
        """Variance of softmax distribution as cheap curvature proxy."""
        return torch.var(F.softmax(logits, dim=-1), dim=-1).mean()

    def geometric_confidence(self, theta: torch.Tensor, safety_risk: torch.Tensor) -> float:
        """
        C(theta) = margin to basin boundary x curvature penalty.
        Higher = deeper in safe basin.
        """
        dist = LA.norm(theta - self.theta_ref).item()
        risk = safety_risk.item()
        confidence = torch.exp(torch.tensor(-self.lambda_curv * risk - dist)).item()
        return confidence

    def repair_step(
        self,
        theta: torch.Tensor,
        model_fn,
        safety_inputs: torch.Tensor,
        task_inputs: torch.Tensor,
        task_labels: torch.Tensor
    ) -> tuple:
        """
        Single repair step. Returns (new_theta, metrics_dict).
        """
        start_time = time.perf_counter()
        theta = theta.detach().requires_grad_(True)

        # Task loss
        task_out = model_fn(task_inputs, theta)
        task_loss = F.cross_entropy(task_out, task_labels)

        # Safety loss: KL to reference output
        with torch.no_grad():
            ref_out = model_fn(safety_inputs, self.theta_ref)

        safety_out = model_fn(safety_inputs, theta)
        kl_loss = F.kl_div(
            F.log_softmax(safety_out, dim=-1),
            F.softmax(ref_out, dim=-1),
            reduction='batchmean'
        )

        # Curvature-weighted safety penalty
        curv = self.curvature_proxy(safety_out)
        weighted_safety = kl_loss * (1.0 + self.lambda_curv * curv)

        # Asymmetric combined loss
        total_loss = task_loss - self.lambda_asym * weighted_safety
        total_loss.backward()

        with torch.no_grad():
            delta = -self.lr * theta.grad
            # Project to trust region
            norm = LA.norm(delta)
            if norm > self.trust_radius:
                delta = delta * (self.trust_radius / norm)
            theta_new = theta + delta

        elapsed = time.perf_counter() - start_time
        confidence = self.geometric_confidence(theta_new, curv.detach())

        metrics = {
            'task_loss': task_loss.item(),
            'safety_loss': kl_loss.item(),
            'curvature': curv.item(),
            'confidence': confidence,
            'dist_to_ref': LA.norm(theta_new - self.theta_ref).item(),
            'repair_cost_seconds': elapsed,
        }

        return theta_new.detach(), metrics
