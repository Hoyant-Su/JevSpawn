import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from tests.infra.action_prefix.runtime import install


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    if settings['action_prefix']:
        install(json.loads(Path(settings['action_prefix_settings']).read_text()))
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
