import argparse
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import patch

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute
from jev_spawn.runtime.prefix_cache import PrefixCache


def run(settings):
    definition = settings['implementation']
    tail_class = getattr(import_module(definition['module']), definition['class'])

    def make_tail(service, backend, shared, inference):
        service.execution_metadata['history_cache_implementation'] = definition
        return tail_class(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                          inference['state_copy'], inference['graph_shape'])

    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    with patch.object(StableGraphFiniteService, '_make_tail', make_tail):
        execute(specification, Path(settings['run_output']),
                json.loads(Path(settings['parallel_settings']).read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
