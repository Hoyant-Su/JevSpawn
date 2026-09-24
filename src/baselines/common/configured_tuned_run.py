import argparse
from pathlib import Path

from baselines.common.configured_environment_run import run
from baselines.common.tuned_parallel_run import execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parallel-settings', required=True)
    parser.add_argument('--tuning', required=True)
    execute(parser.parse_args(), run)


if __name__ == '__main__':
    main()
