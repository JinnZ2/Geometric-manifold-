"""
Cost scaling experiment.
Increases drift strength incrementally and tracks repair cost per cycle.
Tests hypothesis: cost stays linear near basin, spikes nonlinearly at threshold.
"""

import yaml
import copy
import numpy as np
import pandas as pd
from simulation.environment import Environment
from simulation.controller import Controller


def run_cost_analysis(config_path='configs/default.yaml',
                      drift_values=None, steps=50):
    if drift_values is None:
        drift_values = np.linspace(0.0, 1.0, 11)

    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    results = []

    for drift in drift_values:
        config = copy.deepcopy(base_config)
        config['simulation']['drift_strength'] = float(drift)
        config['simulation']['steps'] = steps
        config['monitoring']['output_dir'] = f"results/cost_analysis/drift_{drift:.2f}/"
        config['monitoring']['log_interval'] = steps + 1  # suppress per-step output

        env = Environment(config['simulation'])
        ctrl = Controller(env, config)
        monitor = ctrl.run()
        summary = monitor.summary()

        results.append({
            'drift_strength': drift,
            'mean_cost_ms': summary.get('mean_cost_ms', 0),
            'max_cost_ms': summary.get('max_cost_ms', 0),
            'repair_rate': summary.get('repair_rate', 0),
            'mean_confidence': summary.get('mean_confidence', 0),
        })
        print(f"Drift {drift:.2f}: mean_cost={summary.get('mean_cost_ms',0):.2f}ms | "
              f"repair_rate={summary.get('repair_rate',0):.2f}")

    df = pd.DataFrame(results)
    df.to_csv('results/cost_analysis/cost_vs_drift.csv', index=False)
    print("\nSaved: results/cost_analysis/cost_vs_drift.csv")
    return df


if __name__ == '__main__':
    import os
    os.makedirs('results/cost_analysis', exist_ok=True)
    df = run_cost_analysis()
    print(df.to_string())
