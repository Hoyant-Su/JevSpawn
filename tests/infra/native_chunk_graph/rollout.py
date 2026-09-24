import argparse
import json
from pathlib import Path

import torch.distributed as dist

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute
from tests.infra.native_chunk_graph.candidate import install


def run(settings):
    original = StableGraphFiniteService._make_tail
    operators = []

    def make_tail(service, backend, shared, inference):
        graphs = install(settings['chunk_graph'], backend.device)
        operators.append((dist.get_rank(), graphs))
        return original(service, backend, shared, inference)

    StableGraphFiniteService._make_tail = make_tail
    execute(json.loads(Path(settings['specification']).read_text()),
            Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
    for rank, graphs in operators:
        path = Path(settings['run_output']) / settings['operator_file'].format(rank=rank)
        path.write_text(json.dumps({'rank': rank, 'captures': graphs.captures,
            'replays': graphs.replays, 'retained_graphs': len(graphs.entries)}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
