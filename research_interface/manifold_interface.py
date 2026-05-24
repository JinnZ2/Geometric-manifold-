"""
Manifold Research Interface — optimization environment for autonomous bridge-geometry search.

An external AI (or any optimizer) treats this as a pure mathematical landscape:
  - Input:   bridge_matrix W (dims × dims) — the geometric lens to vary
  - Outputs: net_viability (maximize), prediction_error, heat_leak (both minimize)

Mathematical model
------------------
Given sensory flux S ∈ R^d and physical metrics P ∈ R^d:

  warped      = W @ S                        (linear projection through bridge)
  compressed  = tanh(warped)                 (squash to (-1, 1) — bounded sensory encoding)
  effective_P = P ⊙ (1 + compressed)        (sensory modulation of physical metrics)

  prediction_error = mean( (compressed - P̂)^2 )   where P̂ = P / max(|P|)
    → 0 when the bridge perfectly maps sensory patterns onto normalised physical space.

  heat_leak = Σ|P| - Σ|effective_P|
    → positive = sensory compression suppressed physical metrics (energy lost)
    → negative = compression amplified them (gain)

  net_viability = Σ(effective_P) - w_pred · prediction_error - w_heat · |heat_leak|
    → the AI's sole objective to maximise.

The tension between minimising prediction_error and not inflating |heat_leak| is the
core difficulty of the landscape: perfectly matching P̂ via tanh implies effective_P > P
(compression > 0), which makes heat_leak negative but incurs an abs() penalty.

Gradient computation
--------------------
Uses central finite differences on each element of W (same correctness principle as
fd_hvp: use the step directly, divide by 2ε, no renormalisation).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


class ManifoldResearchInterface:
    """
    Optimization environment: bridge-matrix search over a Riemannian sensory–physical manifold.
    """

    def __init__(
        self,
        manifold_dimensions: int = 4,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self.dims = manifold_dimensions
        self.metric_space = np.eye(self.dims)  # base Euclidean metric

        # Penalty weights — configurable so the researching AI can vary them
        self.w_pred = cfg.get("prediction_error_penalty", 50.0)
        self.w_heat = cfg.get("heat_leak_penalty", 1.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Core evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_bridge_geometry(
        self,
        bridge_matrix: np.ndarray,
        sensory_flux: np.ndarray,
        physical_metrics: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Score a bridge-geometry hypothesis.

        Returns
        -------
        net_viability     : float — objective to maximise
        prediction_error  : float — MSE between compressed sensory and normalised physical
        heat_leak         : float — Σ|P| - Σ|effective_P| (positive = loss, negative = gain)
        manifold_coordinates : list[float] — tanh(W @ S), the compressed sensory encoding
        effective_metrics    : list[float] — P ⊙ (1 + compressed)
        """
        S = np.array(sensory_flux, dtype=float).reshape(-1, 1)
        P = np.array(physical_metrics, dtype=float).reshape(-1, 1)
        W = np.array(bridge_matrix, dtype=float)

        if W.shape != (self.dims, self.dims):
            raise ValueError(f"bridge_matrix must be ({self.dims},{self.dims}), got {W.shape}")
        if S.shape[0] != self.dims or P.shape[0] != self.dims:
            raise ValueError(f"sensory_flux and physical_metrics must have {self.dims} elements")

        # 1. Project sensory data through the bridge geometry
        warped = W @ S                          # (dims, 1)
        compressed = np.tanh(warped)            # bounded in (-1, 1)

        # 2. Sensory modulation of physical metrics
        effective_P = P * (1.0 + compressed)   # scales P into (0, 2P)

        # 3. Loss terms
        p_scale = np.max(np.abs(P)) + 1e-8     # safe normalisation — never divide by zero
        p_hat = P / p_scale                     # normalised physical target ∈ (-1, 1)
        prediction_error = float(np.mean(np.square(compressed - p_hat)))

        heat_leak = float(np.sum(np.abs(P)) - np.sum(np.abs(effective_P)))

        net_viability = (
            float(np.sum(effective_P))
            - self.w_pred * prediction_error
            - self.w_heat * abs(heat_leak)
        )

        return {
            "net_viability": round(net_viability, 6),
            "prediction_error": round(prediction_error, 6),
            "heat_leak": round(heat_leak, 6),
            "manifold_coordinates": compressed.flatten().tolist(),
            "effective_metrics": effective_P.flatten().tolist(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Gradient (central finite differences on W)
    # ──────────────────────────────────────────────────────────────────────────

    def gradient(
        self,
        bridge_matrix: np.ndarray,
        sensory_flux: np.ndarray,
        physical_metrics: np.ndarray,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """
        ∂(net_viability) / ∂W via central finite differences.

        For each element W[i,j]:
          grad[i,j] = (f(W + ε·E_ij) − f(W − ε·E_ij)) / 2ε

        Uses W directly (no normalisation) and divides by 2ε only — same
        correctness principle as fd_hvp in stability.py.
        """
        W = np.array(bridge_matrix, dtype=float)
        grad = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_p = W.copy(); W_p[i, j] += epsilon
                W_m = W.copy(); W_m[i, j] -= epsilon
                f_p = self.evaluate_bridge_geometry(W_p, sensory_flux, physical_metrics)["net_viability"]
                f_m = self.evaluate_bridge_geometry(W_m, sensory_flux, physical_metrics)["net_viability"]
                grad[i, j] = (f_p - f_m) / (2.0 * epsilon)
        return grad

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def identity_bridge(self) -> np.ndarray:
        """Start from the identity transform — no sensory warping."""
        return np.eye(self.dims)

    def random_bridge(self, scale: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
        """Random initialisation drawn from N(0, scale²)."""
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, size=(self.dims, self.dims))

    def near_diagonal_bridge(self, diag_strength: float = 0.9, noise: float = 0.1,
                             seed: Optional[int] = None) -> np.ndarray:
        """Near-diagonal initialisation — weak off-diagonal coupling."""
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, noise, size=(self.dims, self.dims))
        np.fill_diagonal(W, diag_strength)
        return W
