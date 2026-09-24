import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from tests.infra.native_state_context.candidate import install


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    install(settings['context_view'])
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
