import argparse
from pathlib import Path

from baselines.common.parallel_run import execute_with_runner
from baselines.common.run import run_with_environment
from baselines.common.schema_environment import environment_definition
from baselines.common.tasks import read


def run(specification, output, backend=None):
    factory, definition = environment_definition(specification)
    return run_with_environment(specification, output, backend, environment_factory=factory,
                                environment_contract={'environment_execution': definition})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-settings', type=Path, required=True)
    args = parser.parse_args()
    specification = read(args.specification)
    environment_definition(specification)
    execute_with_runner(specification, args.output, read(args.parallel_settings), run)


if __name__ == '__main__':
    main()
