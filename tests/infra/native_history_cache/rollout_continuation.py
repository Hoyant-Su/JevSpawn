import argparse
import json
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.native_history_cache.continuation import ContinuationTail, install_messages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    continuation = json.loads(Path(settings['continuation']).read_text())
    install_messages(continuation)

    def make_tail(service, backend, shared, inference):
        return ContinuationTail(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
            inference['state_copy'], inference['graph_shape'], continuation=continuation)

    StableGraphFiniteService._make_tail = make_tail
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
