"""
ParallelPullTransformer — non-linear, non-sequential signal compressor.

Architecture
------------
Two weight matrices operate concurrently (no sequential dependency):

  Sensory path:  compressed = tanh(S @ w_sensory)     ← bounded pattern compressor
  Science path:  processed  = P @ w_science            ← linear metric transform
  Cross-attention: output   = processed ⊙ (1 + compressed)

The cross-attention formula is identical to ManifoldResearchInterface:
  output_i = (Σ_j P_j w_science[j,i]) × (1 + tanh(Σ_j S_j w_sensory[j,i]))

Convention note
---------------
Both weight matrices use left-multiply (numpy row-vector convention):
  v @ W  ≡  W.T @ v  (column vector)
This is equivalent to ManifoldResearchInterface's W @ S with W = w_sensory.T.
If you want right-multiply semantics, transpose the matrices on construction.

Scoring
-------
The transformer exposes a scalar score via ManifoldResearchInterface using:
  bridge_matrix = w_sensory.T   (converts to right-multiply convention)
  physical_metrics = S_science  = P @ w_science  (pre-projected science vector)
This lets BridgeOptimizer optimise w_sensory; w_science is treated as a fixed
basis unless explicitly co-optimised.

Gradient
--------
Central FD on each element of w_sensory and w_science, consistent with fd_hvp:
  grad_ws[i,j] = (f(ws + ε·E_ij) − f(ws − ε·E_ij)) / 2ε
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from research_interface.manifold_interface import ManifoldResearchInterface


class ParallelPullTransformer:
    """
    Concurrent sensory + science signal processor with cross-attention modulation.
    """

    def __init__(
        self,
        feature_dimensions: int = 4,
        config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        cfg = config or {}
        self.dims = feature_dimensions
        scale = cfg.get("init_scale", 0.1)
        rng   = np.random.default_rng(seed)

        self.w_sensory = rng.normal(0.0, scale, size=(self.dims, self.dims))
        self.w_science = rng.normal(0.0, scale, size=(self.dims, self.dims))

        # Scoring uses ManifoldResearchInterface for consistent loss computation
        self._scorer = ManifoldResearchInterface(
            manifold_dimensions=self.dims,
            config={
                "prediction_error_penalty": cfg.get("prediction_error_penalty", 50.0),
                "heat_leak_penalty":        cfg.get("heat_leak_penalty",        1.5),
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def sensory_compressor(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        tanh(S @ w_sensory) — projects high-bandwidth patterns into a bounded
        state vector ∈ (−1, 1).  +1 = zero entropy / maximum alignment.
        −1 = maximum friction / predictive error.
        """
        S = np.asarray(sensory_input, dtype=float).flatten()
        return np.tanh(S @ self.w_sensory)

    def science_transformer(self, science_input: np.ndarray) -> np.ndarray:
        """P @ w_science — linear projection of rigid physical metrics."""
        P = np.asarray(science_input, dtype=float).flatten()
        return P @ self.w_science

    def cross_attention_pull(
        self,
        sensory_field: np.ndarray,
        science_field: np.ndarray,
    ) -> np.ndarray:
        """
        Non-sequential handshake:
          output = science_field ⊙ (1 + sensory_field)

        When sensory_field → +1 (low friction): output → 2 × science_field (amplify)
        When sensory_field → −1 (high friction): output → 0               (suppress)
        When sensory_field = 0 (neutral):        output = science_field    (passthrough)
        """
        return science_field * (1.0 + sensory_field)

    def Execute_Pull(
        self,
        sensory_input,
        science_input,
    ) -> np.ndarray:
        """
        Runs both paths concurrently and merges via cross-attention.
        Returns the unified viability vector (dims,).
        """
        compressed = self.sensory_compressor(sensory_input)
        processed  = self.science_transformer(science_input)
        output     = self.cross_attention_pull(compressed, processed)
        return np.round(output, 6)

    # Alias following Python naming conventions
    execute_pull = Execute_Pull

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring (via ManifoldResearchInterface for consistency)
    # ──────────────────────────────────────────────────────────────────────────

    def score(
        self,
        sensory_input,
        science_input,
        w_sensory: Optional[np.ndarray] = None,
        w_science: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Score the current (or proposed) weights using ManifoldResearchInterface.

        bridge_matrix = w_sensory.T  (right-multiply convention)
        physical_metrics = P @ w_science  (pre-projected science)

        This allows BridgeOptimizer to treat w_sensory as a bridge matrix directly.
        """
        ws = (w_sensory if w_sensory is not None else self.w_sensory)
        wp = (w_science  if w_science  is not None else self.w_science)
        S  = np.asarray(sensory_input, dtype=float).flatten()
        P  = np.asarray(science_input, dtype=float).flatten()
        effective_P = P @ wp
        return self._scorer.evaluate_bridge_geometry(ws.T, S, effective_P)

    # ──────────────────────────────────────────────────────────────────────────
    # Gradients (central FD)
    # ──────────────────────────────────────────────────────────────────────────

    def gradient_w_sensory(
        self,
        sensory_input,
        science_input,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """
        ∂(net_viability) / ∂w_sensory via central FD.
        Same correctness principle as fd_hvp: use the step directly, divide by 2ε.
        """
        grad = np.zeros_like(self.w_sensory)
        for i in range(self.dims):
            for j in range(self.dims):
                ws_p = self.w_sensory.copy(); ws_p[i, j] += epsilon
                ws_m = self.w_sensory.copy(); ws_m[i, j] -= epsilon
                f_p = self.score(sensory_input, science_input, w_sensory=ws_p)["net_viability"]
                f_m = self.score(sensory_input, science_input, w_sensory=ws_m)["net_viability"]
                grad[i, j] = (f_p - f_m) / (2.0 * epsilon)
        return grad

    def gradient_w_science(
        self,
        sensory_input,
        science_input,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """∂(net_viability) / ∂w_science via central FD."""
        grad = np.zeros_like(self.w_science)
        for i in range(self.dims):
            for j in range(self.dims):
                wp_p = self.w_science.copy(); wp_p[i, j] += epsilon
                wp_m = self.w_science.copy(); wp_m[i, j] -= epsilon
                f_p = self.score(sensory_input, science_input, w_science=wp_p)["net_viability"]
                f_m = self.score(sensory_input, science_input, w_science=wp_m)["net_viability"]
                grad[i, j] = (f_p - f_m) / (2.0 * epsilon)
        return grad

    def joint_step(
        self,
        sensory_input,
        science_input,
        lr: float = 0.01,
        trust_radius: float = 0.1,
        epsilon: float = 1e-4,
    ) -> dict:
        """
        One gradient-ascent step on both weight matrices jointly.
        Trust region applied per matrix (Frobenius norm ≤ trust_radius).
        Returns score metrics after the step.
        """
        g_s = self.gradient_w_sensory(sensory_input, science_input, epsilon)
        g_p = self.gradient_w_science(sensory_input, science_input, epsilon)

        for grad, attr in [(g_s, "w_sensory"), (g_p, "w_science")]:
            delta = lr * grad
            frob  = np.linalg.norm(delta)
            if frob > trust_radius:
                delta = delta * (trust_radius / frob)
            setattr(self, attr, getattr(self, attr) + delta)

        return self.score(sensory_input, science_input)
