"""
SystemicViabilityModel — Thermodynamic & Information-Design Framework.

Evaluates the net energetic and structural cost of a system over time using five
physically-grounded variables. Each variable is projected as a linear timeline
(or updated step-by-step from a transformer), then combined into a per-step
net viability index:

  useful_work  = energy_input × local_efficiency × w_efficiency
  waste        = infrastructure_decay × w_decay
               + trust_entropy × w_trust
               + (1 − tech_maturity) × w_maturity    ← speculative penalty
  viability[t] = useful_work[t] − waste[t]

The speculative penalty makes "head-in-the-air" technology (maturity → 0) expensive
in exactly the same way the manifold research interface penalises low-alignment
bridge geometries: the penalty is proportional to the gap between claimed and
delivered capability.

Variable ranges
---------------
  energy_input        ≥ 0          (Watts / normalised load)
  local_efficiency    ∈ [0, 1]     (fraction of energy retained locally)
  tech_maturity       ∈ [0, 1]     (0 = speculative, 1 = deployed, tested, working)
  infrastructure_decay ∈ [0, 1]   (0 = no wear, 1 = critical failure trajectory)
  trust_entropy       ∈ [0, 1]    (0 = solid network, 1 = total institutional conflict)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


_VARIABLE_NAMES = (
    "energy_input",
    "local_efficiency",
    "tech_maturity",
    "infrastructure_decay",
    "trust_entropy",
)

_BOUNDED_01 = ("local_efficiency", "tech_maturity", "infrastructure_decay", "trust_entropy")


class SystemicViabilityModel:
    """
    Temporal viability ledger: net energetic value minus systemic waste per step.
    """

    def __init__(self, name: str, timeline_steps: int = 10, config: Optional[dict] = None):
        cfg = config or {}
        self.name = name
        self.steps = timeline_steps

        self.variables: dict[str, np.ndarray] = {k: np.zeros(self.steps) for k in _VARIABLE_NAMES}

        self.weights = {
            "w_energy":     cfg.get("w_energy",     1.0),
            "w_efficiency": cfg.get("w_efficiency",  1.5),
            "w_maturity":   cfg.get("w_maturity",    2.0),  # high weight: penalises unapplied theory
            "w_decay":      cfg.get("w_decay",       1.2),
            "w_trust":      cfg.get("w_trust",       1.8),  # social friction = direct energy loss
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Variable setters
    # ──────────────────────────────────────────────────────────────────────────

    def set_variable_timeline(self, variable_name: str, start_val: float, end_val: float) -> None:
        """Linearly interpolate a variable across the full timeline."""
        if variable_name not in self.variables:
            raise ValueError(f"Unknown variable '{variable_name}'. Options: {list(self.variables)}")
        if variable_name in _BOUNDED_01 and not (0.0 <= start_val <= 1.0 and 0.0 <= end_val <= 1.0):
            raise ValueError(f"'{variable_name}' must stay in [0, 1]; got [{start_val}, {end_val}]")
        if variable_name == "energy_input" and (start_val < 0 or end_val < 0):
            raise ValueError("energy_input must be non-negative")
        self.variables[variable_name] = np.linspace(start_val, end_val, self.steps)

    def set_variable_at_step(self, variable_name: str, step: int, value: float) -> None:
        """Set a single step's value — used when transformer feeds step-by-step."""
        if variable_name not in self.variables:
            raise ValueError(f"Unknown variable '{variable_name}'")
        if not (0 <= step < self.steps):
            raise IndexError(f"step {step} out of range [0, {self.steps})")
        self.variables[variable_name][step] = float(value)

    def set_from_vector(self, step: int, vector: np.ndarray) -> None:
        """
        Ingest a 5-element normalised state vector at a single step.

        The vector is expected to have components in [-1, 1] (e.g., transformer output
        after tanh squashing).  Mapping:
          v[0] → energy_input   : abs(v[0]) × max_energy  (scaled by current max or 1.0)
          v[1] → local_efficiency: (v[1] + 1) / 2 ∈ [0, 1]
          v[2] → tech_maturity  : (v[2] + 1) / 2 ∈ [0, 1]
          v[3] → infrastructure_decay: (1 − v[3]) / 2 ∈ [0, 1]  (inverted: +1 = no decay)
          v[4] → trust_entropy  : (1 − v[4]) / 2 ∈ [0, 1]       (inverted: +1 = solid trust)
        """
        v = np.asarray(vector, dtype=float).flatten()
        if len(v) < 5:
            raise ValueError(f"vector must have at least 5 elements; got {len(v)}")
        self.variables["energy_input"][step]        = abs(float(v[0]))
        self.variables["local_efficiency"][step]    = float(np.clip((v[1] + 1.0) / 2.0, 0.0, 1.0))
        self.variables["tech_maturity"][step]       = float(np.clip((v[2] + 1.0) / 2.0, 0.0, 1.0))
        self.variables["infrastructure_decay"][step] = float(np.clip((1.0 - v[3]) / 2.0, 0.0, 1.0))
        self.variables["trust_entropy"][step]       = float(np.clip((1.0 - v[4]) / 2.0, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Computation
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_dynamic_waste(self) -> np.ndarray:
        """
        Per-step waste:
          physical_waste   = infrastructure_decay × w_decay
          social_waste     = trust_entropy × w_trust
          speculative_leak = (1 − tech_maturity) × w_maturity
        """
        physical_waste   = self.variables["infrastructure_decay"] * self.weights["w_decay"]
        social_waste     = self.variables["trust_entropy"]        * self.weights["w_trust"]
        speculative_leak = (1.0 - self.variables["tech_maturity"]) * self.weights["w_maturity"]
        return physical_waste + social_waste + speculative_leak

    def evaluate_viability(self) -> np.ndarray:
        """
        Net Systemic Viability Index per step:
          viability[t] = (energy_input[t] × local_efficiency[t] × w_efficiency) − waste[t]
        """
        useful_work = (
            self.variables["energy_input"]
            * self.variables["local_efficiency"]
            * self.weights["w_efficiency"]
        )
        return np.round(useful_work - self.calculate_dynamic_waste(), 6)

    def net_scalar(self) -> float:
        """Mean viability across the timeline — single score for optimisation."""
        return float(np.mean(self.evaluate_viability()))

    # ──────────────────────────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────────────────────────

    def generate_audit_report(self) -> None:
        """Raw physical ledger — no narrative, just numbers."""
        viability = self.evaluate_viability()
        waste     = self.calculate_dynamic_waste()
        print(f"\n=== SYSTEMIC AUDIT: {self.name.upper()} ===")
        print(f"{'Step':<6}{'Energy In':<12}{'Maturity':<10}{'Trust Ent':<11}{'Tot Waste':<11}{'NET VIABILITY'}")
        print("-" * 65)
        for t in range(self.steps):
            print(
                f"{t:<6}"
                f"{self.variables['energy_input'][t]:<12.3f}"
                f"{self.variables['tech_maturity'][t]:<10.3f}"
                f"{self.variables['trust_entropy'][t]:<11.3f}"
                f"{waste[t]:<11.3f}"
                f"{viability[t]:+12.3f}"
            )
        print("=" * 65)
        print(f"  Mean viability: {self.net_scalar():+.4f}  |  "
              f"Final viability: {viability[-1]:+.4f}")

    def to_dict(self) -> dict:
        """Serialise current state for science_constraint_bridge / monitoring."""
        viability = self.evaluate_viability()
        waste     = self.calculate_dynamic_waste()
        return {
            "name":           self.name,
            "steps":          self.steps,
            "net_scalar":     self.net_scalar(),
            "final_viability": float(viability[-1]),
            "mean_waste":      float(np.mean(waste)),
            "variables":      {k: v.tolist() for k, v in self.variables.items()},
            "viability":      viability.tolist(),
            "waste":          waste.tolist(),
        }
