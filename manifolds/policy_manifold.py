"""
Policy Manifold Layer - Trajectory-level alignment.

Treats agent trajectories as points on a policy manifold.
Triggers re-anchoring when geometric confidence drops below threshold.
"""

import torch
import torch.nn.functional as F


class PolicyManifold:
    def __init__(self, config: dict):
        self.confidence_threshold = config.get('confidence_threshold', 0.4)
        self.reanchor_strength = config.get('reanchor_strength', 0.1)

    def trajectory_confidence(
        self,
        action_probs: torch.Tensor,
        ref_action_probs: torch.Tensor
    ) -> float:
        """
        Confidence = 1 - JS divergence between current and reference policy.
        High confidence = trajectory stays near reference policy basin.
        """
        m = 0.5 * (action_probs + ref_action_probs)
        jsd = 0.5 * F.kl_div(action_probs.log(), m, reduction='batchmean') + \
              0.5 * F.kl_div(ref_action_probs.log(), m, reduction='batchmean')
        return max(0.0, 1.0 - jsd.item())

    def needs_repair(self, confidence: float) -> bool:
        return confidence < self.confidence_threshold

    def reanchor(
        self,
        action_probs: torch.Tensor,
        ref_action_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Blend current policy toward reference policy by reanchor_strength.
        """
        return (1 - self.reanchor_strength) * action_probs + \
               self.reanchor_strength * ref_action_probs
