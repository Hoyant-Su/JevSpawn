import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from candidate import gates_kernel as dynamic_kernel
from jev_spawn.infra.qwen35 import gates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    specification = json.loads(Path(settings['specification']).read_text())
    assert settings['shared_config'] == specification['shared_config']
    gates.gates_kernel = {'reference': gates.gates_kernel, 'dynamic': dynamic_kernel}[settings['kernel']]
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
