"""
Bridge to science_constraint_layers (github.com/JinnZ2/JinnZ2/tree/main/science_constraint_layers).

The external project maintains a three-layer constraint stack:
  Layer 0 – per-domain ConstraintState (physics, biology, thermodynamics, mathematics)
  Layer 1 – cross-domain coupling detectors → IntegratedConstraintState
  Layer 2 – language_codec → human-readable narrative

This module maps the Basin Repair Framework onto that schema so the two
projects can share a common constraint representation:

  Data Manifold      ↔  biology domain      (population/minority dynamics)
  Parameter Manifold ↔  mathematics domain  (curvature, basin geometry)
  Policy Manifold    ↔  thermodynamics domain (entropy, free energy, drift)
  Confidence weights ↔  coupling strengths   (dominant: parameter 50%)

Usage
-----
  from repair.science_constraint_bridge import to_constraint_state, to_coupling_vector

  state_dict = to_constraint_state(param_metrics, policy_conf, data_conf, step=t)
  coupling_vec = to_coupling_vector(state_dict)
"""

from __future__ import annotations

import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ConstraintState schema
# Mirrors science_constraint_layers.science_transformers.ConstraintState.to_dict()
# ─────────────────────────────────────────────────────────────────────────────

def to_constraint_state(
    param_metrics: dict,
    policy_conf: float,
    data_conf: float,
    step: int = 0,
    kappa_eff: Optional[float] = None,
    basin_kl: Optional[float] = None,
) -> dict:
    """
    Convert per-step repair metrics into a ConstraintState-compatible dict.

    Maps each manifold layer to the closest domain in science_constraint_layers:

      Parameter manifold (curvature, dist_to_ref, confidence)
        → mathematics domain (curvature, metric_signature, euler_char proxy)

      Policy manifold (trajectory confidence, JS divergence proxy)
        → thermodynamics domain (entropy proxy, free energy, dS/dt)

      Data manifold (cleaning confidence scalar)
        → biology domain (population balance proxy)

    Returns a flat dict compatible with ConstraintState.to_dict():
      {
        "time": step,
        "domain": "basin_repair",
        "state_vector": [...],           # flattened numeric state
        "constraint_mask": [...],        # bool list — which constraints hold
        "violated": [...]                # names of violated constraints
      }
    """
    curvature = param_metrics.get("curvature", 0.0)
    task_loss = param_metrics.get("task_loss", 0.0)
    safety_loss = param_metrics.get("safety_loss", 0.0)
    dist_to_ref = param_metrics.get("dist_to_ref", 0.0)
    param_conf = param_metrics.get("confidence", 1.0)

    # Mathematics domain state (parameter manifold)
    math_state = {
        "curvature": curvature,
        "euler_char_proxy": 2.0 - 2.0 * min(curvature, 1.0),  # χ ≈ 2(1-g) for low-genus
        "metric_signature": 1.0 if curvature < 10.0 else -1.0,  # +1 = positive definite basin
        "dist_to_basin_floor": dist_to_ref,
    }

    # Thermodynamics domain state (policy manifold)
    # JS divergence proxy: invert confidence → divergence
    js_proxy = 1.0 - policy_conf
    entropy_proxy = -policy_conf * math.log(policy_conf + 1e-8)  # H(confidence) as scalar
    thermo_state = {
        "temperature": safety_loss,          # KL loss ~ thermodynamic temperature
        "entropy": entropy_proxy,
        "free_energy": task_loss + safety_loss,
        "dS_dt": js_proxy,                   # rate of policy drift ~ entropy production
    }

    # Biology domain state (data manifold)
    bio_state = {
        "minority_fraction": data_conf,      # high = minority well-represented
        "population_balance": 2.0 * data_conf - 1.0,  # ∈[-1,1], 0 = balanced
        "metabolic_rate": 1.0 - data_conf,   # cleaning cost proxy
    }

    # Constraint satisfaction
    constraints = {
        "curvature_bounded": curvature < 20.0,          # spectral bound C < 20
        "in_basin": (basin_kl or safety_loss) < 0.1,    # KL < epsilon_basin
        "confidence_above_threshold": param_conf > 0.4,
        "entropy_nondecreasing": js_proxy >= 0.0,       # second law proxy
        "minority_preserved": data_conf > 0.0,
    }
    mask = list(constraints.values())
    violated = [name for name, ok in constraints.items() if not ok]

    state_vector = [
        # mathematics
        math_state["curvature"],
        math_state["euler_char_proxy"],
        math_state["metric_signature"],
        math_state["dist_to_basin_floor"],
        # thermodynamics
        thermo_state["temperature"],
        thermo_state["entropy"],
        thermo_state["free_energy"],
        thermo_state["dS_dt"],
        # biology
        bio_state["minority_fraction"],
        bio_state["population_balance"],
        bio_state["metabolic_rate"],
        # global
        param_conf,
        policy_conf,
        data_conf,
    ]

    return {
        "time": step,
        "domain": "basin_repair",
        "state_vector": state_vector,
        "constraint_mask": mask,
        "violated": violated,
        # domain sub-states (matches ConstraintState structure)
        "mathematics": math_state,
        "thermodynamics": thermo_state,
        "biology": bio_state,
        "kappa_eff": kappa_eff,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coupling vector
# Mirrors science_constraint_layers.constraint_integration_layer.Coupling
# ─────────────────────────────────────────────────────────────────────────────

def to_coupling_vector(state: dict) -> list[dict]:
    """
    Compute inter-domain coupling strengths from a constraint state dict.

    Returns a list of Coupling-compatible dicts (one per pair), ordered to
    match the five detectors in constraint_integration_layer.py:

      thermo_bio       → policy drift × data cleaning interaction
      em_mechanical    → (not applicable; set to 0)
      math_physical    → curvature effect on parameter trajectory
      bio_physical     → data balance effect on safety loss
      thermo_physical  → free energy effect on repair energy
    """
    thermo = state.get("thermodynamics", {})
    bio = state.get("biology", {})
    math = state.get("mathematics", {})

    param_conf = state["state_vector"][11]
    policy_conf = state["state_vector"][12]
    data_conf = state["state_vector"][13]

    # thermo_bio: entropy production rate × metabolic proxy
    thermo_bio_strength = min(1.0, thermo.get("dS_dt", 0.0) * bio.get("metabolic_rate", 0.0) * 4.0)

    # math_physical: curvature scales repair difficulty
    curvature = math.get("curvature", 0.0)
    math_phys_strength = min(1.0, curvature / 20.0)  # normalised by C_bound

    # bio_physical: minority imbalance raises safety loss
    balance = bio.get("population_balance", 0.0)
    bio_phys_strength = min(1.0, abs(balance) * (1.0 - param_conf))

    # thermo_physical: free energy drives step energy
    free_energy = thermo.get("free_energy", 0.0)
    thermo_phys_strength = min(1.0, free_energy / (free_energy + 1.0))

    couplings = [
        {
            "type": "thermodynamic_biological",
            "strength": thermo_bio_strength,
            "direction": "thermo→bio",
            "satisfied": thermo_bio_strength < 0.7,
            "claim": "policy drift and data imbalance co-vary under adversarial drift",
        },
        {
            "type": "electromagnetic_mechanical",
            "strength": 0.0,
            "direction": "none",
            "satisfied": True,
            "claim": "not applicable to parameter-space framework",
        },
        {
            "type": "mathematical_physical",
            "strength": math_phys_strength,
            "direction": "math→physical",
            "satisfied": math_phys_strength < 1.0,
            "claim": "high safety-Hessian curvature predicts expensive repair steps",
        },
        {
            "type": "biological_physical",
            "strength": bio_phys_strength,
            "direction": "bio→physical",
            "satisfied": bio_phys_strength < 0.5,
            "claim": "data minority imbalance amplifies parameter-space safety loss",
        },
        {
            "type": "thermodynamic_physical",
            "strength": thermo_phys_strength,
            "direction": "thermo→physical",
            "satisfied": thermo_phys_strength < 0.8,
            "claim": "accumulated free energy predicts thermodynamic phase transition",
        },
    ]
    return couplings


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory export
# ─────────────────────────────────────────────────────────────────────────────

def export_trajectory(monitor_history: list[dict]) -> list[dict]:
    """
    Convert a list of per-step monitor records to a sequence of constraint
    states for batch import into science_constraint_layers' language_codec.

    Each element of monitor_history is expected to have keys matching the
    output of repair.monitors.RepairMonitor (task_loss, safety_loss,
    curvature, confidence, dist_to_ref, policy_confidence, data_confidence).
    """
    out = []
    for step, record in enumerate(monitor_history):
        param_metrics = {
            k: record.get(k, 0.0)
            for k in ("task_loss", "safety_loss", "curvature", "confidence", "dist_to_ref")
        }
        cs = to_constraint_state(
            param_metrics=param_metrics,
            policy_conf=record.get("policy_confidence", 1.0),
            data_conf=record.get("data_confidence", 1.0),
            step=step,
            kappa_eff=record.get("kappa_eff"),
            basin_kl=record.get("basin_kl"),
        )
        cs["couplings"] = to_coupling_vector(cs)
        out.append(cs)
    return out
