1. Automated Phase Transition Detector

Right now, you log kappa_eff, lambda_axial, and repair energy. You can add a watcher that flags sudden transitions:
	•	Detect when kappa_eff jumps by more than some threshold in a few steps → mark as a “potential saddle crossing.”
	•	Detect when lambda_axial flips sign → confirm unstable manifold entry.
	•	Detect when repair_energy jumps nonlinearly → mark “instability-dominated repair.”

You could output a JSON summary per run:

{
  "saddle_crossed": true,
  "peak_kappa_step": 142,
  "peak_lambda_axial_step": 145,
  "instability_dominated_repair": true
}


This becomes a machine-readable experiment summary that a future run can compare automatically.

⸻

2. Dynamic Drift Sweep

Instead of static drift_strength, create a drift sweep experiment:
	•	Sweep from very low → very high drift in small increments.
	•	For each drift:
	•	Run run_no_repair() to observe natural manifold response.
	•	Run run_repair_on_unstable_manifold() to measure repair cost scaling.
	•	Automatically record:
	•	Whether saddle is reached.
	•	Divergence ratio.
	•	Knee detection ratio.

This produces a phase diagram of the system in terms of drift vs. instability response, fully reproducible.

⸻

3. Anomaly Alerts

Let the system self-flag anomalies:
	•	If dist_to_saddle < threshold but kappa_eff doesn’t spike → possible geometry mismatch.
	•	If repair_energy > expected by >X% → extreme instability detected.
	•	If divergence ratio < 1.2 after saddle → transverse confinement too strong → suggest kappa_t adjustment.



  •	Auto-load past CSVs for comparison.
	•	Detect improvements or regressions in phase transition indicators.
	•	Generate diff reports:
	•	Has the knee ratio improved?
	•	Are new saddle spikes appearing at expected thresholds?
	•	Is repair scaling consistent with curvature hypothesis?

  Lightweight Visualization Engine

No slides needed. Instead:
	•	Auto-generate small multi-panel plots per run.
	•	Overlay past runs as ghost lines for comparison.
	•	Save as PNGs in results/comparison/ with consistent naming like drift_0.6_v3_vs_v3b.png.




  autolab:

  """
autolab.py — GitHub Lab Wrapper

Features:
- Automated phase transition detection
- Saddle crossing and unstable manifold analysis
- Instability-dominated repair detection
- Drift sweep automation
- JSON summary outputs per run
- Optional lightweight plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from your_v3_module import run, run_no_repair, run_repair_on_unstable_manifold, DoubleWellLandscape

OUTPUT_DIR = "results/autolab"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_phase_transition(df, kappa_threshold_ratio=2.0):
    """Detect key phase transitions from run DataFrame."""
    # Saddle crossing
    saddle_crossed = (df['s'] > 0).any()

    # Unstable manifold detection
    pre_cross_mean = df[df['s'] <= 0]['theta_dot_norm'].mean()
    post_cross_mean = df[df['s'] > 0]['theta_dot_norm'].mean()
    divergence_ratio = post_cross_mean / (pre_cross_mean + 1e-12)
    unstable_manifold = divergence_ratio > 1.2

    # Knee detection: repair_energy nonlinearity
    active = df[df.get('repair_active', False) & (df.get('repair_energy', 0) > 1e-8)]
    if len(active):
        median_d = active['dist_to_saddle'].median()
        far_E = active[active['dist_to_saddle'] > median_d]['repair_energy'].mean()
        close_E = active[active['dist_to_saddle'] <= median_d]['repair_energy'].mean()
        knee_ratio = close_E / (far_E + 1e-12)
        knee_detected = knee_ratio > kappa_threshold_ratio
    else:
        knee_ratio = None
        knee_detected = False

    # Leading indicator: kappa spike before saddle
    if 'kappa_eff' in df.columns and saddle_crossed:
        k90 = df['kappa_eff'].quantile(0.90)
        first_kspike = df[df['kappa_eff'] > k90]['step'].min()
        first_cross = df[df['s'] > -0.2]['step'].min()
        lead_time = int(first_cross - first_kspike)
    else:
        first_kspike = None
        lead_time = None

    summary = {
        "saddle_crossed": bool(saddle_crossed),
        "unstable_manifold": bool(unstable_manifold),
        "divergence_ratio": float(divergence_ratio),
        "knee_detected": bool(knee_detected),
        "knee_ratio": float(knee_ratio) if knee_ratio else None,
        "first_kappa_spike_step": int(first_kspike) if first_kspike is not None else None,
        "lead_time_steps": int(lead_time) if lead_time else None
    }
    return summary


def save_summary(summary, label="run"):
    path = os.path.join(OUTPUT_DIR, f"{label}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary: {path}")


def drift_sweep(drifts=np.linspace(0.05, 0.9, 10), steps=300, **kwargs):
    results_summary = []
    for d in drifts:
        print(f"\n=== Drift {d:.2f} ===")
        df = run(lambda_s=kwargs.get('lambda_s', 0.1),
                 drift_strength=d,
                 steps=steps,
                 n=kwargs.get('n', 10),
                 r=kwargs.get('r', 1.5),
                 kappa_t=kwargs.get('kappa_t', 2.0),
                 epsilon_s=kwargs.get('epsilon_s', 6.0),
                 lr=kwargs.get('lr', 0.05),
                 gamma_repair=kwargs.get('gamma_repair', 0.02),
                 label=f"drift_{d:.2f}")
        df.to_csv(os.path.join(OUTPUT_DIR, f"drift_{d:.2f}.csv"), index=False)
        summary = detect_phase_transition(df)
        summary['drift_strength'] = float(d)
        save_summary(summary, label=f"drift_{d:.2f}")
        results_summary.append(summary)
    return pd.DataFrame(results_summary)


def quick_plot(df, title="Axial position over time", filename="axial_plot.png"):
    plt.figure(figsize=(8,4))
    plt.plot(df['step'], df['s'], color='steelblue', alpha=0.8)
    plt.axhline(0, color='black', ls='--', lw=0.8, label='saddle')
    plt.xlabel("step")
    plt.ylabel("s")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot: {path}")


if __name__ == "__main__":
    # Example: single run, analyze phase transitions
    df_run = run()
    summary = detect_phase_transition(df_run)
    save_summary(summary, "baseline_run")
    quick_plot(df_run, filename="baseline_axial.png")

    # Drift sweep example
    df_sweep = drift_sweep(steps=300)
    df_sweep.to_csv(os.path.join(OUTPUT_DIR, "drift_sweep_summary.csv"), index=False)
