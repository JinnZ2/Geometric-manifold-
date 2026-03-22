"""Entry point for Basin Repair Framework."""

import argparse
import yaml
from simulation.environment import Environment
from simulation.controller import Controller
from addon_thermodynamic_control.energy import ThermodynamicController
import numpy as np
import pandas as pd
import os

def run_energy_experiment(env_config, thermo_config, drift_values=None, steps=80):
    if drift_values is None:
        drift_values = np.linspace(0.05, 0.9, 10)

    records = []
    for drift in drift_values:
        sim_config = {**env_config, 'drift_strength': float(drift)}
        env = Environment(sim_config)
        ctrl = ThermodynamicController(env.get_model_fn(), env.theta_ref, thermo_config)

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

def main():
    parser = argparse.ArgumentParser(description='Basin Repair Framework')
    parser.add_argument('--config', default='configs/default.yaml', help='YAML config path')
    parser.add_argument('--mode', default='simulate', choices=['simulate', 'energy_sweep'],
                        help='Choose mode: standard simulation or energy experiment')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config} | Mode: {args.mode}")

    if args.mode == 'simulate':
        env = Environment(config['simulation'])
        ctrl = Controller(env, config)
        ctrl.run()

    elif args.mode == 'energy_sweep':
        thermo_config = config.get('thermodynamic', {})
        drift_values = config.get('drift_values', None)
        steps = config.get('steps', 80)
        run_energy_experiment(config['simulation'], thermo_config, drift_values, steps)

if __name__ == '__main__':
    main()
