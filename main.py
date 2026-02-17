"""Entry point."""

import argparse
import yaml
from simulation.environment import Environment
from simulation.controller import Controller


def main():
    parser = argparse.ArgumentParser(description='Basin Repair Framework')
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    env = Environment(config['simulation'])
    ctrl = Controller(env, config)
    ctrl.run()


if __name__ == '__main__':
    main()
