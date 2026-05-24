"""
IntegratedViabilityPipeline — connects ParallelPullTransformer → SystemicViabilityModel.

Architecture
------------

  Raw signals at step t
        │
        ▼
  ParallelPullTransformer.Execute_Pull(sensory_t, science_t)
        │   compressed viability vector (dims,)
        ▼
  SystemicViabilityModel.set_from_vector(step=t, vector=output_t)
        │   maps vector → {energy_input, local_efficiency, tech_maturity,
        │                   infrastructure_decay, trust_entropy}
        ▼
  SystemicViabilityModel.evaluate_viability()
        │   net viability index (steps,)
        ▼
  science_constraint_bridge.to_constraint_state(...)
        │   ConstraintState dict for language_codec / coupling monitors
        ▼
  optional: BridgeOptimizer / ParallelTransformer.joint_step()
        │   adapt w_sensory, w_science to improve net_scalar over time

The pipeline is stateless between calls to `run_step` — the transformer and
model carry their own state.  Call `run_step` once per environmental update,
then query `model.evaluate_viability()` for the updated trajectory.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from research_interface.systemic_viability import SystemicViabilityModel
from research_interface.parallel_transformer import ParallelPullTransformer
from repair.science_constraint_bridge import to_constraint_state, to_coupling_vector


class IntegratedViabilityPipeline:
    """
    Single coherent interface combining transformer signal compression,
    temporal viability accounting, and constraint monitoring.
    """

    def __init__(
        self,
        name: str = "integrated",
        timeline_steps: int = 10,
        feature_dimensions: int = 5,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self.dims = feature_dimensions
        self.steps = timeline_steps

        self.transformer = ParallelPullTransformer(
            feature_dimensions=feature_dimensions,
            config=cfg.get("transformer", {}),
            seed=cfg.get("seed"),
        )
        self.model = SystemicViabilityModel(
            name=name,
            timeline_steps=timeline_steps,
            config=cfg.get("model", {}),
        )

        self._history: list[dict] = []
        self._current_step = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Step-by-step ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def run_step(
        self,
        sensory_input: Sequence[float],
        science_input: Sequence[float],
        adapt_weights: bool = False,
        lr: float = 0.01,
    ) -> dict:
        """
        Process one environmental update.

        1. Transformer compresses raw signals → viability vector
        2. Systemic model ingests the vector at the current step
        3. Constraint state and coupling vector are computed
        4. Optionally adapt transformer weights (joint_step)

        Returns a metrics dict suitable for logging / export.
        """
        t = self._current_step
        if t >= self.steps:
            raise RuntimeError(
                f"Pipeline has {self.steps} steps; step {t} exceeds capacity. "
                "Construct a new pipeline or reset."
            )

        # 1. Compress signals
        output = self.transformer.Execute_Pull(sensory_input, science_input)

        # 2. Update systemic model — normalise to 5 variables
        # If dims > 5, take the first 5 components; if dims < 5, zero-pad
        padded = np.zeros(5)
        n = min(5, self.dims)
        padded[:n] = output[:n]
        self.model.set_from_vector(t, padded)

        # 3. Score the current transformer configuration
        score = self.transformer.score(sensory_input, science_input)

        # 4. Constraint state (uses current model variable values at step t)
        viability = self.model.evaluate_viability()
        waste     = self.model.calculate_dynamic_waste()
        param_metrics = {
            "task_loss":    float(waste[t]),
            "safety_loss":  float(max(0.0, -viability[t])),   # negative viability = safety risk
            "curvature":    score.get("prediction_error", 0.0),
            "confidence":   float(np.clip(viability[t] / (abs(viability[t]) + 1.0) + 0.5, 0.0, 1.0)),
            "dist_to_ref":  score.get("heat_leak", 0.0),
        }
        policy_conf = max(0.0, 1.0 - score.get("prediction_error", 0.0))
        data_conf   = float(1.0 - self.model.variables["trust_entropy"][t])
        cs = to_constraint_state(param_metrics, policy_conf=policy_conf, data_conf=data_conf, step=t)
        couplings = to_coupling_vector(cs)

        # 5. Optional weight adaptation
        if adapt_weights:
            score = self.transformer.joint_step(sensory_input, science_input, lr=lr)

        record = {
            "step":             t,
            "output_vector":    output.tolist(),
            "net_viability":    score.get("net_viability", 0.0),
            "prediction_error": score.get("prediction_error", 0.0),
            "heat_leak":        score.get("heat_leak", 0.0),
            "model_viability":  float(viability[t]),
            "waste":            float(waste[t]),
            "constraint_state": cs,
            "couplings":        couplings,
        }
        self._history.append(record)
        self._current_step += 1
        return record

    # ──────────────────────────────────────────────────────────────────────────
    # Batch ingestion (full timeline at once)
    # ──────────────────────────────────────────────────────────────────────────

    def run_timeline(
        self,
        sensory_timeline: list[list[float]],
        science_timeline: list[list[float]],
        adapt_weights: bool = False,
        log_interval: int = 1,
    ) -> list[dict]:
        """
        Process a full environmental timeline.
        Both input lists must have length equal to timeline_steps.
        """
        if len(sensory_timeline) != self.steps or len(science_timeline) != self.steps:
            raise ValueError(
                f"Timeline length must match steps={self.steps}; "
                f"got sensory={len(sensory_timeline)}, science={len(science_timeline)}"
            )

        print(f"=== {self.model.name.upper()} — Pipeline Run ({self.steps} steps) ===")
        records = []
        for t, (s, p) in enumerate(zip(sensory_timeline, science_timeline)):
            rec = self.run_step(s, p, adapt_weights=adapt_weights)
            if t % log_interval == 0 or t == self.steps - 1:
                print(
                    f"  Step {t:3d} | model_V={rec['model_viability']:+.3f} "
                    f"| transformer_V={rec['net_viability']:+.4f} "
                    f"| violations={rec['constraint_state']['violated']}"
                )
            records.append(rec)

        print(f"\n  Mean model viability:       {self.model.net_scalar():+.4f}")
        print(f"  Mean transformer viability: {float(np.mean([r['net_viability'] for r in records])):+.4f}")
        return records

    # ──────────────────────────────────────────────────────────────────────────
    # Inspection
    # ──────────────────────────────────────────────────────────────────────────

    def audit_report(self) -> None:
        """Print the systemic model's full ledger."""
        self.model.generate_audit_report()

    def constraint_history(self) -> list[dict]:
        """Extract constraint states from the run history for language_codec export."""
        return [r["constraint_state"] for r in self._history]

    def dominant_couplings(self) -> list[dict]:
        """Return the strongest coupling at each step."""
        return [
            max(r["couplings"], key=lambda c: c["strength"])
            for r in self._history
        ]
