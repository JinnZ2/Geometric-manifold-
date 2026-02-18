# experiments/cost_analysis.py
"""
Cost Analysis Experiment: Detect nonlinearity in repair compute as function of basin distance.
Prediction: Linear cost when close to reference, quadratic spike at basin boundary.
"""
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List
from manifolds.parameter_manifold import ParameterManifold
from simulation.controller import Controller
from simulation.environment import Environment

def run_cost_scaling_experiment(
    drift_strengths: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
    steps_per_drift: int = 50,
    n_repeats: int = 5,
    config_path: str = "configs/default.yaml"
) -> pd.DataFrame:
    """Main experiment: measure compute cost vs basin distance across drift levels"""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for drift_strength in drift_strengths:
        print(f"\n=== Drift strength: {drift_strength} ===")
        
        for repeat in range(n_repeats):
            # Fresh environment per repeat
            env = Environment(drift_strength=drift_strength, seed=42+repeat)
            controller = Controller.from_config(config)
            
            metrics_history = []
            total_cost = 0.0
            
            for step in range(steps_per_drift):
                start_time = time.perf_counter()
                
                # Single controller step
                metrics = controller.step(
                    theta=controller.current_theta,
                    safety_prompts=env.safety_set,
                    task_prompts=env.task_set
                )
                
                step_cost = time.perf_counter() - start_time  # Proxy for compute cost
                total_cost += step_cost
                
                # Key metrics for analysis
                metrics_history.append({
                    'drift_strength': drift_strength,
                    'step': step,
                    'basin_distance': metrics['dist_to_ref'],
                    'confidence': metrics['confidence'],
                    'repair_triggered': metrics['repair_triggered'],
                    'step_cost': step_cost,
                    'cumulative_cost': total_cost,
                    'safety_loss': metrics['safety_loss']
                })
            
            results.extend(metrics_history)
            print(f"Repeat {repeat+1}: avg_cost/step={total_cost/steps_per_drift:.4f}s")
    
    return pd.DataFrame(results)

def analyze_nonlinearity(df: pd.DataFrame, output_dir: Path):
    """Detect cost nonlinearity and generate key plots"""
    
    # Fit polynomial to cost vs distance
    df_agg = df.groupby('basin_distance')['step_cost'].mean().reset_index()
    
    # Quadratic fit
    coeffs = np.polyfit(df_agg['basin_distance'], df_agg['step_cost'], 2)
    quadratic = np.poly1d(coeffs)
    
    # Find inflection (nonlinearity spike)
    distances = np.linspace(df_agg['basin_distance'].min(), 
                          df_agg['basin_distance'].max(), 100)
    costs_pred = quadratic(distances)
    
    inflection_detected = np.argmax(np.diff(np.gradient(costs_pred))) / 100
    print(f"Nonlinearity spike detected at basin_distance ≈ {inflection_detected:.3f}")
    
    # PLOTS
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cost vs Basin Distance (money plot)
    axes[0,0].scatter(df['basin_distance'], df['step_cost'], alpha=0.5)
    axes[0,0].plot(distances, costs_pred, 'r-', lw=3, label='Quadratic fit')
    axes[0,0].axvline(inflection_detected, color='orange', ls='--', 
                     label=f'Spike at {inflection_detected:.3f}')
    axes[0,0].set_xlabel('Basin Distance')
    axes[0,0].set_ylabel('Step Cost (s)')
    axes[0,0].legend()
    axes[0,0].set_title('Cost Nonlinearity Detection')
    
    # 2. Repair frequency vs distance
    repair_freq = df.groupby('basin_distance')['repair_triggered'].mean()
    axes[0,1].plot(repair_freq.index, repair_freq.values, 'g-o')
    axes[0,1].set_xlabel('Basin Distance')
    axes[0,1].set_ylabel('Repair Frequency')
    axes[0,1].set_title('Repair Triggers Explode at Boundary')
    
    # 3. Cumulative cost by drift strength
    for drift in df['drift_strength'].unique():
        subset = df[df['drift_strength'] == drift]
        axes[1,0].plot(subset['step'], subset['cumulative_cost'], 
                      label=f'drift={drift}')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_ylabel('Cumulative Cost (s)')
    axes[1,0].legend()
    axes[1,0].set_title('Total Repair Budget by Drift')
    
    # 4. Confidence collapse
    conf_by_dist = df.groupby('basin_distance')['confidence'].mean()
    axes[1,1].plot(conf_by_dist.index, conf_by_dist.values, 'purple')
    axes[1,1].set_xlabel('Basin Distance')
    axes[1,1].set_ylabel('Geometric Confidence')
    axes[1,1].set_title('Confidence → 0 at Boundary')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_analysis_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw data
    df.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Summary stats
    summary = {
        'inflection_point': float(inflection_detected),
        'max_cost_per_step': float(df['step_cost'].max()),
        'repair_frequency_at_spike': float(df[df['basin_distance'] > inflection_detected]['repair_triggered'].mean()),
        'cost_savings_prediction': 'Data rectification should shift spike rightward'
    }
    
    with open(output_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f)
    
    return summary

def main():
    output_dir = Path('results/cost_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running basin repair cost scaling experiment...")
    df = run_cost_scaling_experiment()
    
    print("\nAnalyzing nonlinearity...")
    summary = analyze_nonlinearity(df, output_dir)
    
    print("\n=== RESULTS ===")
    print(yaml.dump(summary, default_flow_style=False))
    print(f"\nKey plots saved to {output_dir}/")
    print("Next: run `full_pipeline.py --compare_to baseline_no_rectification`")

if __name__ == "__main__":
    main()
