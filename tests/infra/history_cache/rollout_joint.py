import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from jev_spawn.infra.configuration import RESOURCES
from jev_spawn.rollout import branching
from tests.infra.history_cache.joint_control import select_control


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    RESOURCES['readout_labels'] = settings['readout_labels']
    if settings['joint_control']:
        branching.select_control = select_control
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
