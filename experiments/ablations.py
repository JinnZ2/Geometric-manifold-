"""
Ablation runner. Disable individual layers and compare outcomes.
"""

import yaml
import copy
import sys
from simulation.environment import Environment
from simulation.controller import Controller


def run_ablation(base_config_path: str, disable_layer: str):
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config_ablated = copy.deepcopy(config)
    config_ablated['manifolds'][disable_layer]['enabled'] = False
    config_ablated['monitoring']['output_dir'] = f"results/ablation_{disable_layer}/"

    print(f"\n{'='*50}")
    print(f"ABLATION: disabling {disable_layer} layer")
    print(f"{'='*50}")

    env = Environment(config_ablated['simulation'])
    ctrl = Controller(env, config_ablated)
    monitor = ctrl.run()
    return monitor.summary()


if __name__ == '__main__':
    layer = sys.argv[1] if len(sys.argv) > 1 else 'data'
    config_path = 'configs/default.yaml'
    result = run_ablation(config_path, layer)
    print(f"\nAblation result ({layer} disabled): {result}")
