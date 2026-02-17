"""
Main repair loop. Orchestrates all three manifold layers.
"""

import torch
from manifolds.data_manifold import DataManifold
from manifolds.parameter_manifold import ParameterManifold
from manifolds.policy_manifold import PolicyManifold
from repair.geometric_confidence import GeometricConfidence
from repair.monitors import Monitor


class Controller:
    def __init__(self, env, config: dict):
        self.env = env
        self.config = config
        self.steps = config['simulation']['steps']

        mfld_cfg = config['manifolds']
        mon_cfg = config['monitoring']

        self.data_layer = DataManifold(mfld_cfg['data']) \
            if mfld_cfg['data']['enabled'] else None
        self.param_layer = ParameterManifold(env.theta_ref, mfld_cfg['parameter']) \
            if mfld_cfg['parameter']['enabled'] else None
        self.policy_layer = PolicyManifold(mfld_cfg['policy']) \
            if mfld_cfg['policy']['enabled'] else None

        self.confidence = GeometricConfidence()
        self.monitor = Monitor(mon_cfg)
        self.model_fn = env.get_model_fn()

    def run(self) -> Monitor:
        print("=== Basin Repair Simulation ===")
        print(f"Steps: {self.steps} | Drift: {self.env.drift_strength}")
        print()

        theta = self.env.theta_drifted.clone()

        # Layer 1: Data manifold rectification (runs once pre-loop)
        data_conf = 1.0
        if self.data_layer:
            feat_clean, lbl_clean, weights = self.data_layer.rectify(
                self.env.features, self.env.labels
            )
            data_conf = weights.mean().item()
            print(f"Data rectification: {len(self.env.features)} → {len(feat_clean)} samples")
            print(f"Data manifold confidence: {data_conf:.3f}")
            print()

        for step in range(self.steps):
            metrics = {'repair_triggered': False, 'data_confidence': data_conf}

            # Layer 2: Parameter space repair
            if self.param_layer:
                theta, param_metrics = self.param_layer.repair_step(
                    theta,
                    self.model_fn,
                    self.env.safety_inputs,
                    self.env.task_inputs,
                    self.env.task_labels
                )
                metrics.update(param_metrics)
                metrics['repair_triggered'] = param_metrics['confidence'] < 0.5

            # Layer 3: Policy manifold check
            if self.policy_layer:
                with torch.no_grad():
                    curr_probs = torch.softmax(
                        self.model_fn(self.env.task_inputs, theta), dim=-1
                    )
                    ref_probs = torch.softmax(
                        self.model_fn(self.env.task_inputs, self.env.theta_ref), dim=-1
                    )
                policy_conf = self.policy_layer.trajectory_confidence(curr_probs, ref_probs)
                metrics['policy_confidence'] = policy_conf

                if self.policy_layer.needs_repair(policy_conf):
                    metrics['repair_triggered'] = True

            # Combined confidence
            param_conf = metrics.get('confidence', 1.0)
            policy_conf = metrics.get('policy_confidence', 1.0)
            metrics['combined_confidence'] = self.confidence.combined(
                data_conf, param_conf, policy_conf
            )

            self.monitor.log(step, metrics)
            self.monitor.detect_cost_spike()

        self.monitor.save()
        print("\n=== Summary ===")
        summary = self.monitor.summary()
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return self.monitor
