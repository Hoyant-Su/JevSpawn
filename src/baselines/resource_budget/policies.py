import importlib
import json
from pathlib import Path

import torch

from baselines.formal_choices.run import read, save
from jev_spawn.infra.prompts import load_prompt


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class AgentPrunePolicy:
    def __init__(self, backend, settings, service):
        self.core = importlib.import_module('baselines.agentprune.formal')
        self.settings, self.service = settings, service
        training = read(settings['training_config'])
        assert training['batch_size'] == settings['batch_size'] == backend.config['batch_size']
        assert training['training']['num_rounds'] == settings['num_rounds']
        assert training['generation'] == settings['generation']
        self.core.load_core(settings['source'])
        self.graph = self.core.load_checkpoint(Path(settings['checkpoint']), training)
        self.frozen = {name: getattr(self.graph, name).detach().tolist()
                       for name in ['spatial_logits', 'temporal_logits', 'spatial_masks', 'temporal_masks']}
        self.core.configure(service, settings['generation'], load_prompt(settings['prompts']))

    def execute(self, tasks, output, seed, offset, split):
        source = self.settings['warmup_tasks'] if split == 'development' else self.settings['tasks']
        result = self.core.block_run(self.graph, self.service, tasks, source, self.settings,
                                     output, seed, offset, split)
        assert all(getattr(self.graph, name).detach().tolist() == value for name, value in self.frozen.items())
        return [row['answer'] for row in result['result']['records']]

    def warmup(self, tasks, output, seed):
        self.execute(tasks, output, seed, 0, 'development')

    def batch(self, tasks, output, seed, offset):
        return self.execute(tasks, output, seed, offset, 'test')


class LatentMASPolicy:
    def __init__(self, backend, settings, tasks, warmup, output):
        self.core = importlib.import_module('baselines.latentmas.formal')
        self.backend, self.settings = backend, settings
        self.runner = self.core.method(backend, settings, tasks + warmup)
        self.runner.model._ensure_latent_realign_matrix(self.runner.model.model, backend.device, self.runner.args)
        save(output / 'latent-preflight.json', self.core.preflight(self.runner, tasks, settings, backend.config))

    @torch.inference_mode()
    def warmup(self, tasks, output, seed):
        output.mkdir()
        for size in self.settings['warmup_batch_sizes']:
            assert size <= len(tasks)
            result = self.core.measure(self.runner, self.backend, tasks[:size], self.settings, seed)
            save(output / f'batch-{size}.json', result)

    @torch.inference_mode()
    def batch(self, tasks, output, seed, offset):
        output.mkdir()
        assert seed == self.settings['seed']
        result = self.core.measure(self.runner, self.backend, tasks, self.settings, seed)
        save(output / 'complete.json', result)
        return [row['answer'] for row in result['predictions']]
