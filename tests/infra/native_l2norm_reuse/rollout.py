import argparse
import faulthandler
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from tests.infra.native_l2norm_reuse.candidate import CANDIDATE, MODULE, REFERENCE


def run(settings):
    faulthandler.enable(all_threads=False)
    MODULE.l2norm_fwd_kernel = {
        'reference': REFERENCE, 'runtime_bucket': CANDIDATE,
    }[settings['arm']]
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
