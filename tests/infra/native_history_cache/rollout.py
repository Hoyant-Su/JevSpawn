import argparse
import json
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.infra.history_cache import HistoryTail


def install_history_cache(settings):
    def make_tail(service, backend, shared, inference):
        return HistoryTail(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                           inference['state_copy'], inference['graph_shape'], settings['history_cache'])

    StableGraphFiniteService._make_tail = make_tail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    install_history_cache(settings)
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
