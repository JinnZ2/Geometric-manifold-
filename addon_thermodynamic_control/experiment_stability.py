"""
Stability Certificate Experiment

Tests:
  1. Is dV/dt <= 0 maintained along trajectory? (Lyapunov condition)
  2. Does spectral bound lambda_max(H_safety) < C hold?
  3. Does kappa_eff spike before basin_kl exceeds epsilon? (leading indicator)
  4. Does the proactive term reduce sup kappa_eff vs no proactive?

These four checks are the empirical preconditions for a
formal stability proof. If all hold across drift levels,
the framework is a candidate for an ISS theorem.
"""

import numpy as np
import pandas as pd
import os
from simulation.environment import Environment
from addon_thermodynamic_control.stability import CoupledDynamicalSystem

CONFIG = {
    'lambda_safety':        1.0,
    'lambda_proactive':     0.5,
    'lambda_policy':        0.3,
    'mu_repair':            0.05,
    'mu_max':               10.0,
    'repair_budget':        100.0,
    'epsilon_basin':        0.15,
    'spectral_C_bound':     20.0,
    'alpha_kl':             1.0,
    'lr':                   0.01,
    'sigma_drift':          0.05,
}

CONFIG_NO_PROACTIVE = {**CONFIG, 'lambda_proactive': 0.0}


def run_trial(drift, config, steps=100, label=''):
    env = Environment({'drift_strength': drift, 'seed': 42})
    sys = CoupledDynamicalSystem(
        env.get_model_fn(), env.theta_ref, env.task_inputs, config
    )
    theta = env.theta_drifted.clone()

    records = []
    for _ in range(steps):
        theta, state = sys.step(
            theta, env.safety_inputs, env.task_inputs, env.task_labels
        )
        records.append({
            'step':         state.step,
            'drift':        drift,
            'config':       label,
            'V':            state.V,
            'dV_dt':        state.dV_dt,
            'V_neg':        state.V_dot_negative,
            'kappa_eff':    state.kappa_eff,
            'lambda_max':   state.spectral_norm_hessian,
            'basin_kl':     state.basin_kl,
            'repair_E':     state.repair_energy_step,
            'phase':        state.phase,
            'spec_ok':      state.spectral_bound_satisfied,
        })

    cert = sys.lyapunov_certificate_summary()
    return pd.DataFrame(records), cert


def run_all(drift_values=None, steps=100):
    if drift_values is None:
        drift_values = [0.2, 0.4, 0.6, 0.8]

    os.makedirs('results/stability', exist_ok=True)
    all_records = []
    cert_records = []

    for drift in drift_values:
        for cfg, label in [(CONFIG, 'proactive'), (CONFIG_NO_PROACTIVE, 'no_proactive')]:
            print(f"\n=== drift={drift:.1f} | {label} ===")
            df, cert = run_trial(drift, cfg, steps=steps, label=label)
            all_records.append(df)

            # Leading indicator check
            kappa_90 = df['kappa_eff'].quantile(0.9)
            kappa_spike_step = df[df['kappa_eff'] > kappa_90]['step'].min()
            basin_fail_step  = df[df['basin_kl'] > CONFIG['epsilon_basin']]['step'].min()
            lead_steps = (int(basin_fail_step) - int(kappa_spike_step)
                          if not (np.isnan(kappa_spike_step) or np.isnan(basin_fail_step))
                          else None)

            cert_records.append({
                'drift':                drift,
                'config':               label,
                'lead_steps':           lead_steps,
                **cert
            })

            print(f"  Lyapunov violations:  {cert['V_dot_violations']}/{steps}")
            print(f"  Spectral violations:  {cert['spectral_violations']}/{steps}")
            print(f"  Empirical stability:  {cert['empirical_stability']}")
            print(f"  kappa_eff spike lead: {lead_steps} steps before basin failure")

    pd.concat(all_records).to_csv('results/stability/trajectory.csv', index=False)
    pd.DataFrame(cert_records).to_csv('results/stability/certificates.csv', index=False)
    print("\nSaved: results/stability/trajectory.csv")
    print("Saved: results/stability/certificates.csv")
    print("\nTo claim ISS theorem candidate: all 'empirical_stability' should be True.")


if __name__ == '__main__':
    run_all()
