"""
Energy accounting experiment.
Runs ThermodynamicController across drift levels.
Produces the phase transition signature:
  - stable phase: near-constant repair energy
  - threshold phase: trend ratio climbs
  - critical phase: spike, behavioral failure follows
"""

import yaml
import numpy as np
import pandas as pd
import os
from simulation.environment import Environment
from addon_thermodynamic_control.energy import ThermodynamicController

THERMO_CONFIG = {
    'lambda_safety': 1.0,
    'lambda_policy': 0.3,
    'mu_repair': 0.05,
    'repair_budget': 50.0,
    'epsilon_basin': 0.15,
    'lr': 0.01,
}

def run(drift_values=None, steps=80, base_sim_config=None):
    if drift_values is None:
        drift_values = np.linspace(0.05, 0.9, 10)
    if base_sim_config is None:
        base_sim_config = {'seed': 42}

    records = []
    for drift in drift_values:
        sim_config = {**base_sim_config, 'drift_strength': float(drift)}
        env = Environment(sim_config)
        ctrl = ThermodynamicController(env.get_model_fn(), env.theta_ref, THERMO_CONFIG)

        theta = env.theta_drifted.clone()
        for _ in range(steps):
            theta, _ = ctrl.step(theta, env.safety_inputs, env.task_inputs, env.task_labels)

        summary = ctrl.summary()
        records.append({'drift': drift, **summary})
        print(f"drift={drift:.2f} | phase={summary['final_phase']} | "
              f"peak_E={summary['peak_repair_energy']:.4f} | "
              f"KL_final={summary['final_kl']:.4f}")

    os.makedirs('results/energy_experiment', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv('results/energy_experiment/phase_transitions.csv', index=False)
    print("\nSaved: results/energy_experiment/phase_transitions.csv")
    return df

if __name__ == '__main__':
    run()
