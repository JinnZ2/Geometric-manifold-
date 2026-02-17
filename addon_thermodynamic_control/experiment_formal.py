"""
Experiment: Formal Objectives Validation

Tests the key prediction:
  kappa_eff spikes BEFORE behavioral failure (basin_kl exceeds epsilon).
  kappa_eff is a leading indicator, not a lagging one.

Also tests:
  J_proactive reduces kappa_eff over time when lambda_p > 0
  (geometry shaping actually flattens the landscape).
"""

import numpy as np
import pandas as pd
import os
from simulation.environment import Environment
from addon_thermodynamic_control.addendum_formal_objectives import UnifiedLagrangian

CONFIG_BASE = {
    'lambda_safety':    1.0,
    'lambda_proactive': 0.5,
    'mu_repair':        0.05,
    'mu_max':           5.0,
    'repair_budget':    100.0,
    'epsilon_basin':    0.15,
    'lr':               0.01,
    'sigma_drift':      0.05,
}

CONFIG_NO_PROACTIVE = {**CONFIG_BASE, 'lambda_proactive': 0.0}

def run_trial(drift: float, config: dict, steps: int = 100, label: str = '') -> pd.DataFrame:
    env = Environment({'drift_strength': drift, 'seed': 42})
    ctrl = UnifiedLagrangian(env.get_model_fn(), env.theta_ref, config)
    theta = env.theta_drifted.clone()

    records = []
    for _ in range(steps):
        theta, state = ctrl.step(
            theta, env.safety_inputs, env.task_inputs, env.task_labels
        )
        records.append({
            'step':             state.step,
            'drift':            drift,
            'config':           label,
            'kappa_eff':        state.kappa_eff,
            'basin_kl':         state.basin_kl,
            'repair_energy':    state.repair_energy,
            'proactive_loss':   state.proactive_loss,
            'task_loss':        state.task_loss,
            'phase':            state.phase,
        })

    return pd.DataFrame(records)


def run_all():
    os.makedirs('results/formal', exist_ok=True)
    results = []

    for drift in [0.2, 0.4, 0.6, 0.8]:
        print(f"\n=== Drift {drift:.1f} | With proactive ===")
        df = run_trial(drift, CONFIG_BASE, steps=100, label='proactive')
        results.append(df)

        print(f"\n=== Drift {drift:.1f} | Without proactive ===")
        df2 = run_trial(drift, CONFIG_NO_PROACTIVE, steps=100, label='no_proactive')
        results.append(df2)

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv('results/formal/unified_lagrangian.csv', index=False)
    print("\nSaved: results/formal/unified_lagrangian.csv")

    # Key check: does kappa spike before basin_kl crosses epsilon?
    print("\n=== Leading Indicator Check ===")
    for drift in [0.2, 0.4, 0.6, 0.8]:
        sub = all_results[(all_results['drift']==drift) & (all_results['config']=='proactive')]
        kappa_spike = sub[sub['kappa_eff'] > sub['kappa_eff'].quantile(0.9)]['step'].min()
        basin_fail  = sub[sub['basin_kl'] > 0.15]['step'].min()
        lead = basin_fail - kappa_spike if not np.isnan(kappa_spike) else 'no spike'
        print(f"  drift={drift:.1f}: kappa_spike@{kappa_spike} | basin_fail@{basin_fail} | lead={lead} steps")

    return all_results


if __name__ == '__main__':
    run_all()
