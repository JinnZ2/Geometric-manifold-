"""
Bridge Optimizer — gradient-ascent search over the bridge-matrix space.

Design mirrors ParameterManifold.repair_step():
  - Gradient step via central FD (matches fd_hvp philosophy)
  - Trust-region clamp on ΔW (same hard guarantee as parameter manifold)
  - Returns (new_matrix, metrics_dict) for direct use in a research loop

The optimizer is intentionally stateless between steps so an external AI
can interleave its own logic (population-based search, meta-learning, etc.)
without fighting internal state. State accumulation is the caller's problem.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional

from research_interface.manifold_interface import ManifoldResearchInterface


class BridgeOptimizer:
    """
    Single-step and multi-step gradient ascent on net_viability.

    Parameters
    ----------
    interface     : ManifoldResearchInterface
    lr            : float — gradient step size
    trust_radius  : float — max Frobenius norm of ΔW per step
    momentum      : float — exponential moving average on gradient (0 = no momentum)
    """

    def __init__(
        self,
        interface: ManifoldResearchInterface,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self.interface = interface
        self.lr = cfg.get("lr", 0.01)
        self.trust_radius = cfg.get("trust_radius", 0.1)
        self.momentum = cfg.get("momentum", 0.0)
        self.fd_epsilon = cfg.get("fd_epsilon", 1e-4)

        self._velocity: Optional[np.ndarray] = None  # momentum buffer

    def step(
        self,
        bridge_matrix: np.ndarray,
        sensory_flux: np.ndarray,
        physical_metrics: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        One gradient-ascent step with trust-region projection.

        Returns
        -------
        W_new   : np.ndarray — updated bridge matrix
        metrics : dict       — evaluation at W_new + step diagnostics
        """
        t0 = time.perf_counter()

        grad = self.interface.gradient(
            bridge_matrix, sensory_flux, physical_metrics, epsilon=self.fd_epsilon
        )

        # Optional momentum
        if self.momentum > 0.0:
            if self._velocity is None:
                self._velocity = np.zeros_like(grad)
            self._velocity = self.momentum * self._velocity + (1.0 - self.momentum) * grad
            effective_grad = self._velocity
        else:
            effective_grad = grad

        # Gradient ascent step
        delta = self.lr * effective_grad

        # Trust-region clamp (Frobenius norm)
        frob = np.linalg.norm(delta)
        if frob > self.trust_radius:
            delta = delta * (self.trust_radius / frob)

        W_new = bridge_matrix + delta
        elapsed = time.perf_counter() - t0

        metrics = self.interface.evaluate_bridge_geometry(W_new, sensory_flux, physical_metrics)
        metrics.update({
            "grad_frob_norm": round(float(np.linalg.norm(grad)), 6),
            "delta_frob_norm": round(float(np.linalg.norm(delta)), 6),
            "trust_region_active": bool(frob > self.trust_radius),
            "step_seconds": round(elapsed, 4),
        })
        return W_new, metrics

    def run(
        self,
        initial_matrix: np.ndarray,
        sensory_flux: np.ndarray,
        physical_metrics: np.ndarray,
        n_steps: int = 50,
        log_interval: int = 10,
        early_stop_delta: float = 1e-5,
    ) -> list[dict]:
        """
        Full optimization loop. Returns history of per-step metrics.

        Stops early if net_viability changes less than early_stop_delta for
        3 consecutive steps (gradient flat — local optimum reached).
        """
        W = initial_matrix.copy()
        history = []
        flat_streak = 0

        baseline = self.interface.evaluate_bridge_geometry(W, sensory_flux, physical_metrics)
        if log_interval > 0:
            print(f"  Step    0 | viability={baseline['net_viability']:+.4f} "
                  f"| pred_err={baseline['prediction_error']:.4f} "
                  f"| heat={baseline['heat_leak']:+.4f}")

        prev_viability = baseline["net_viability"]

        for step in range(1, n_steps + 1):
            W, m = self.step(W, sensory_flux, physical_metrics)
            m["step"] = step
            history.append(m)

            if step % log_interval == 0 or step == n_steps:
                print(f"  Step {step:4d} | viability={m['net_viability']:+.4f} "
                      f"| pred_err={m['prediction_error']:.4f} "
                      f"| heat={m['heat_leak']:+.4f} "
                      f"| ΔW_frob={m['delta_frob_norm']:.5f} "
                      f"| trust={'*' if m['trust_region_active'] else ' '}")

            # Early stopping
            delta_v = abs(m["net_viability"] - prev_viability)
            flat_streak = (flat_streak + 1) if delta_v < early_stop_delta else 0
            if flat_streak >= 3:
                print(f"  Early stop at step {step} (viability flat for 3 steps)")
                break
            prev_viability = m["net_viability"]

        return history

    def reset_momentum(self) -> None:
        self._velocity = None
