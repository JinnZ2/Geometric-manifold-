"""
Fieldlink export generator for Rosetta-Shape-Core integration.

Produces two JSON files that Rosetta-Shape-Core expects:
  1. manifold_invariants.json — mathematical contracts from the three-layer pipeline
  2. basin_topology.json — basin structure, trust regions, confidence geometry

These files let Rosetta validate that the geometric-manifold fieldlink source
satisfies its bridge contracts without running Python or importing torch.

Usage:
    python scripts/fieldlink_export.py
    python scripts/fieldlink_export.py --output-dir atlas/exports
"""

import argparse
import json
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "atlas" / "exports"


def generate_manifold_invariants() -> dict:
    """Export the mathematical invariants each manifold layer must satisfy.

    These mirror tests/test_invariants.py but in a machine-readable format
    that Rosetta's validator can consume.
    """
    return {
        "$schema": "urn:basin-repair:invariants:v1",
        "repo": "JinnZ2/Geometric-manifold-",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layers": {
            "parameter_manifold": {
                "id": "BASIN.PARAM_MANIFOLD",
                "rosetta_shape": "SHAPE.DODECA",
                "invariants": [
                    {
                        "name": "trust_region",
                        "description": "||delta|| <= trust_radius after every step",
                        "type": "hard_constraint",
                        "enforced_by": "manifolds/parameter_manifold.py:78-79",
                    },
                    {
                        "name": "confidence_bounded",
                        "description": "geometric_confidence() returns float in [0, 1]",
                        "type": "output_contract",
                        "range": [0.0, 1.0],
                    },
                    {
                        "name": "confidence_monotonic",
                        "description": "Higher drift from reference produces lower confidence",
                        "type": "monotonicity",
                        "direction": "decreasing_with_drift",
                    },
                    {
                        "name": "curvature_nonneg",
                        "description": "Variance-based curvature proxy is always >= 0",
                        "type": "bound",
                        "lower": 0.0,
                    },
                    {
                        "name": "finite_outputs",
                        "description": "Every repair step produces finite parameters and metrics",
                        "type": "finiteness",
                    },
                    {
                        "name": "bounded_drift",
                        "description": "Over N steps, distance grows by at most N * trust_radius",
                        "type": "lipschitz",
                        "bound_per_step": "trust_radius",
                    },
                ],
            },
            "policy_manifold": {
                "id": "BASIN.POLICY_MANIFOLD",
                "rosetta_shape": "SHAPE.OCTA",
                "invariants": [
                    {
                        "name": "js_self_zero",
                        "description": "JS(P, P) = 0, so confidence = 1.0 for identical distributions",
                        "type": "identity",
                    },
                    {
                        "name": "js_symmetry",
                        "description": "JS(P, Q) == JS(Q, P) within numerical tolerance",
                        "type": "symmetry",
                    },
                    {
                        "name": "reanchor_convex",
                        "description": "reanchor(P, Q) = (1-s)*P + s*Q exactly",
                        "type": "convex_combination",
                    },
                    {
                        "name": "reanchor_boundary",
                        "description": "s=0 returns P, s=1 returns Q",
                        "type": "boundary_condition",
                    },
                ],
            },
            "data_manifold": {
                "id": "BASIN.DATA_MANIFOLD",
                "rosetta_shape": "SHAPE.ICOSA",
                "invariants": [
                    {
                        "name": "minority_never_fully_dropped",
                        "description": "Every minority sample gets weight > 0",
                        "type": "preservation",
                    },
                    {
                        "name": "majority_pruning",
                        "description": "Some low-confidence majority samples are dropped",
                        "type": "filtering",
                    },
                    {
                        "name": "asymmetric_effect",
                        "description": "After rectification, minority fraction increases or stays same",
                        "type": "monotonicity",
                        "direction": "non_decreasing",
                    },
                ],
            },
            "confidence_aggregation": {
                "id": "BASIN.CONFIDENCE",
                "rosetta_shape": "SHAPE.TETRA",
                "invariants": [
                    {
                        "name": "confidence_bounded",
                        "description": "combined() output always in [0, 1]",
                        "type": "bound",
                        "range": [0.0, 1.0],
                    },
                    {
                        "name": "weights_sum_to_one",
                        "description": "Default weights (0.2, 0.5, 0.3) sum to 1.0",
                        "type": "normalization",
                    },
                    {
                        "name": "parameter_dominates",
                        "description": "Parameter confidence (50%) outweighs data (20%) and policy (30%)",
                        "type": "ordering",
                        "order": ["parameter", "policy", "data"],
                    },
                ],
            },
        },
    }


def generate_basin_topology() -> dict:
    """Export the basin structure: trust regions, curvature model, confidence geometry.

    This describes the mathematical shape of the basin repair landscape
    in terms Rosetta's bridge system can relate to polyhedral families.
    """
    return {
        "$schema": "urn:basin-repair:topology:v1",
        "repo": "JinnZ2/Geometric-manifold-",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basin": {
            "description": "Safe model configurations occupy basins in parameter space",
            "reference_point": "theta_ref — the aligned model's flattened parameter vector",
            "drift_model": "Additive Gaussian noise scaled by drift_strength",
        },
        "trust_region": {
            "type": "l2_ball",
            "radius_config_key": "manifolds.parameter.trust_radius",
            "default_radius": 0.05,
            "enforcement": "hard_clamp",
            "description": "Each repair step is projected onto the L2 ball of radius trust_radius",
        },
        "curvature": {
            "proxy": "softmax_variance",
            "description": "Variance of softmax distribution over safety outputs",
            "cost": "O(n) — no Hessian computation",
            "properties": ["non_negative", "finite"],
        },
        "confidence_geometry": {
            "formula": "C(theta) = exp(-lambda_curv * risk - dist_to_ref)",
            "inputs": {
                "risk": "curvature proxy value (softmax variance)",
                "dist_to_ref": "L2 distance from theta to theta_ref",
            },
            "range": [0.0, 1.0],
            "monotonicity": "decreasing with both risk and distance",
        },
        "loss_landscape": {
            "objective": "task_loss - asymmetry_lambda * curvature_weighted_safety_loss",
            "type": "saddle_point",
            "sign_convention": "MINUS — intentional adversarial tension",
            "resolution": "trust region prevents runaway; saddle-point explores landscape",
        },
        "pipeline": {
            "data_manifold": {
                "runs": "once_pre_loop",
                "space": "feature_space",
                "operation": "asymmetric_knn_cleaning",
            },
            "parameter_manifold": {
                "runs": "every_step",
                "space": "weight_space",
                "operation": "curvature_aware_gradient_step",
            },
            "policy_manifold": {
                "runs": "every_step",
                "space": "distribution_space",
                "operation": "js_divergence_monitoring",
            },
        },
        "configs": {
            "default": {
                "drift_strength": 0.3,
                "trust_radius": 0.05,
                "asymmetry_lambda": 10.0,
                "curvature_weight": 2.0,
                "lr": 0.01,
            },
            "adversarial": {
                "drift_strength": 0.8,
                "trust_radius": 0.03,
                "asymmetry_lambda": 20.0,
                "curvature_weight": 4.0,
                "lr": 0.005,
            },
        },
        "rosetta_bridge": {
            "shape_assignments": {
                "data_manifold": "SHAPE.ICOSA",
                "parameter_manifold": "SHAPE.DODECA",
                "policy_manifold": "SHAPE.OCTA",
                "confidence_aggregation": "SHAPE.TETRA",
                "thermodynamic_extension": "SHAPE.CUBE",
            },
            "namespace": "BASIN",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fieldlink exports for Rosetta")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Directory for export files (default: atlas/exports/)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    invariants = generate_manifold_invariants()
    invariants_path = args.output_dir / "manifold_invariants.json"
    invariants_path.write_text(json.dumps(invariants, indent=2) + "\n")
    print(f"Wrote {invariants_path}")

    topology = generate_basin_topology()
    topology_path = args.output_dir / "basin_topology.json"
    topology_path.write_text(json.dumps(topology, indent=2) + "\n")
    print(f"Wrote {topology_path}")


if __name__ == "__main__":
    main()
