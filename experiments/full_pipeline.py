"""Full pipeline: all three layers active."""

import yaml
from simulation.environment import Environment
from simulation.controller import Controller

if __name__ == '__main__':
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)

    env = Environment(config['simulation'])
    ctrl = Controller(env, config)
    ctrl.run()
