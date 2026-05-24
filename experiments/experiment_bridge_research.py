"""
Autonomous Bridge-Geometry Research Experiment.

Demonstrates the ManifoldResearchInterface + BridgeOptimizer pipeline:
  1. Score two hand-crafted hypotheses (reproduces the original demo)
  2. Run gradient-ascent optimization from three starting points
  3. Export the best bridge's trajectory to science_constraint_bridge format
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_interface import ManifoldResearchInterface, BridgeOptimizer
from repair.science_constraint_bridge import to_constraint_state, to_coupling_vector


SENSORY_FLUX     = [0.9, -0.4, 0.7, -0.1]
PHYSICAL_METRICS = [120.0, 75.0, 0.35, 50.0]

# Hypothesis matrices from the original specification
HYPO_1 = np.array([
    [0.1, -0.2,  0.5,  0.0],
    [0.9,  0.1, -0.3,  0.2],
    [-0.4, 0.6,  0.1,  0.8],
    [0.2, -0.1,  0.7,  0.3],
])

HYPO_2 = np.array([
    [0.9,  0.0,  0.1, -0.8],
    [-0.1, 0.85, 0.0,  0.3],
    [0.2, -0.3,  0.95, 0.0],
    [0.0,  0.1, -0.2,  0.9],
])


def run():
    sandbox = ManifoldResearchInterface(manifold_dimensions=4)
    optimizer = BridgeOptimizer(sandbox, config={"lr": 0.02, "trust_radius": 0.1, "momentum": 0.3})

    print("=== PART 1: Hand-crafted Hypothesis Scoring ===\n")
    for i, W in enumerate([HYPO_1, HYPO_2], 1):
        result = sandbox.evaluate_bridge_geometry(W, SENSORY_FLUX, PHYSICAL_METRICS)
        print(f"Hypothesis #{i}:")
        print(f"  Net Viability:    {result['net_viability']:+.4f}")
        print(f"  Prediction Error: {result['prediction_error']:.4f}")
        print(f"  Heat Leak:        {result['heat_leak']:+.4f}")
        print(f"  Manifold Coords:  {[round(x,3) for x in result['manifold_coordinates']]}")
        print()

    print("=== PART 2: Autonomous Gradient Ascent ===\n")
    init_strategies = {
        "identity":     sandbox.identity_bridge(),
        "near_diagonal": sandbox.near_diagonal_bridge(seed=42),
        "random":       sandbox.random_bridge(scale=0.3, seed=0),
    }

    best_W = None
    best_viability = -np.inf
    best_history = []

    for name, W0 in init_strategies.items():
        print(f"--- Init: {name} ---")
        optimizer.reset_momentum()
        history = optimizer.run(W0, SENSORY_FLUX, PHYSICAL_METRICS, n_steps=60, log_interval=20)
        final = history[-1]["net_viability"]
        print(f"  Final viability: {final:+.4f}\n")
        if final > best_viability:
            best_viability = final
            best_history = history
            # Reconstruct best W by replaying (optimizer is stateless — store last W via history)
            W = W0.copy()
            for m in history:
                # delta is not stored, so we re-derive from gradient for the best run
                pass
            best_W = W0  # placeholder — see note below

    print(f"=== Best viability across all runs: {best_viability:+.4f} ===\n")

    print("=== PART 3: science_constraint_layers Export ===\n")
    # Convert best-run history to ConstraintState sequence
    for step, m in enumerate(best_history[:5]):
        cs = to_constraint_state(
            param_metrics={
                "task_loss":    m["prediction_error"],
                "safety_loss":  max(0.0, m["heat_leak"]),
                "curvature":    m.get("grad_frob_norm", 0.0),
                "confidence":   min(1.0, max(0.0, m["net_viability"] / 300.0 + 0.5)),
                "dist_to_ref":  m["delta_frob_norm"],
            },
            policy_conf=1.0 - m["prediction_error"],
            data_conf=min(1.0, max(0.0, 1.0 - abs(m["heat_leak"]) / 100.0)),
            step=step,
        )
        couplings = to_coupling_vector(cs)
        dominant = max(couplings, key=lambda c: c["strength"])
        print(f"  Step {step:2d}: viability={m['net_viability']:+.4f} | "
              f"violations={cs['violated']} | "
              f"dominant_coupling={dominant['type']} ({dominant['strength']:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    run()
