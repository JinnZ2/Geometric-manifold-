"""
Basin occupancy monitoring. Tracks metrics over time.
Flags potential nonlinear cost spikes.
"""

import pandas as pd
import numpy as np


class Monitor:
    def __init__(self, config: dict):
        self.log_interval = config.get('log_interval', 5)
        self.output_dir = config.get('output_dir', 'results/')
        self.records = []

    def log(self, step: int, metrics: dict):
        metrics['step'] = step
        self.records.append(metrics)
        if step % self.log_interval == 0:
            print(f"Step {step:4d} | conf={metrics.get('confidence', 0):.3f} | "
                  f"dist={metrics.get('dist_to_ref', 0):.3f} | "
                  f"cost={metrics.get('repair_cost_seconds', 0)*1000:.1f}ms | "
                  f"repair={'YES' if metrics.get('repair_triggered') else 'no'}")

    def detect_cost_spike(self, window=10, threshold=2.0) -> bool:
        """
        Returns True if recent repair costs have spiked nonlinearly.
        Uses rolling mean ratio as simple detector.
        """
        if len(self.records) < window * 2:
            return False
        costs = [r.get('repair_cost_seconds', 0) for r in self.records]
        recent = np.mean(costs[-window:])
        prior = np.mean(costs[-window*2:-window])
        if prior > 0 and recent / prior > threshold:
            print(f"  *** COST SPIKE DETECTED: {recent/prior:.2f}x increase ***")
            return True
        return False

    def save(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_csv(f"{self.output_dir}/metrics.csv", index=False)
        print(f"Saved metrics to {self.output_dir}/metrics.csv")

    def summary(self) -> dict:
        if not self.records:
            return {}
        costs = [r.get('repair_cost_seconds', 0) for r in self.records]
        confs = [r.get('confidence', 0) for r in self.records]
        repairs = [r.get('repair_triggered', False) for r in self.records]
        return {
            'total_steps': len(self.records),
            'repair_rate': sum(repairs) / len(repairs),
            'mean_confidence': np.mean(confs),
            'mean_cost_ms': np.mean(costs) * 1000,
            'max_cost_ms': np.max(costs) * 1000,
            'cost_spike_detected': self.detect_cost_spike(),
        }
