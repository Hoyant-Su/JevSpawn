import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

from baselines.common.parallel_run import execute
from baselines.latentmas import common_service
from jev_spawn.infra.prompts import load_prompt
from tests.infra.latent_history.candidate import CachedRoleTransport
from tests.infra.latent_history.checkpoints import CheckpointRoleZeroHistory


def run(settings):
    specification = json.loads(Path(settings['specification']).read_text())
    method = json.loads(Path(specification['method']).read_text())
    load_prompt(method['prompts'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    service = common_service.LatentMASService
    original_init, original_generate = service.__init__, service._generate_batch

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.role_zero_history = CheckpointRoleZeroHistory(self.backend, settings['cache'])
        self.execution_metadata['role_zero_history'] = settings['cache']

    def generate(self, batch):
        start = len(self.role_zero_history.records)
        constructor = partial(CachedRoleTransport, history=self.role_zero_history,
                              sequences=[request.role_ids[settings['role_zero']] for request in batch])
        with patch.object(common_service, 'CapturedPaddingTransport', constructor):
            result = original_generate(self, batch)
        if self.records:
            self.records[settings['last_index']]['role_zero_history'] = self.role_zero_history.records[start:]
        return result

    if settings['arm'] == settings['cached_arm']:
        with patch.object(service, '__init__', initialize), patch.object(service, '_generate_batch', generate):
            execute(specification, Path(settings['run']), parallel)
    else:
        execute(specification, Path(settings['run']), parallel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
