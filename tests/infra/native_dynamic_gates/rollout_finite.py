import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from candidate import gdn_gates as dynamic_gates
from jev_spawn.infra.qwen35 import ragged_gdn
from jev_spawn.infra.qwen35.gates import gdn_gates as native_gates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    specification = json.loads(Path(settings['specification']).read_text())
    assert settings['shared_config'] == specification['shared_config']
    ragged_gdn.gdn_gates = {'reference': native_gates,
                          'dynamic': dynamic_gates}[settings['kernel']]
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
