"""
Unified geometric confidence interface across all three manifolds.
Each manifold defines its own confidence metric,
but this module normalizes them to [0,1] for the monitor.
"""

import torch


class GeometricConfidence:
    def __init__(self):
        pass

    def normalize(self, value: float, min_val=0.0, max_val=1.0) -> float:
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val + 1e-8)))

    def combined(self, data_conf: float, param_conf: float, policy_conf: float,
                 weights=(0.2, 0.5, 0.3)) -> float:
        """Weighted combination of all three manifold confidences."""
        return (weights[0] * data_conf +
                weights[1] * param_conf +
                weights[2] * policy_conf)
