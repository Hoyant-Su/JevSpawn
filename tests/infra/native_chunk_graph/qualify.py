import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch.distributed as dist

from jev_spawn.infra.qwen35 import ragged_gdn
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_chunk_graph.candidate import ChunkGraphs
from tests.infra.native_direct_finite import qualify as paired


def run(settings):
    observations = []

    class GraphTail(StableFiniteGraphTail):
        def __init__(self, backend, runtime, *args):
            super().__init__(backend, runtime, *args)
            self.operator = ChunkGraphs(ragged_gdn.chunk_gated_delta_rule,
                                         settings['chunk_graph'], backend.device)

        def score(self, *args, **kwargs):
            captures, replays = self.operator.captures, self.operator.replays
            started = time.perf_counter()
            with patch.object(ragged_gdn, 'chunk_gated_delta_rule', self.operator):
                result = super().score(*args, **kwargs)
            observations.append({'rank': dist.get_rank(),
                'elapsed_host_seconds': time.perf_counter() - started,
                'new_chunk_graphs': self.operator.captures - captures,
                'chunk_replays': self.operator.replays - replays,
                'computed_input_tokens': result['computed_input_tokens']})
            return result

    paired.DirectFiniteTail = GraphTail
    paired.run(settings)
    rank, = {row['rank'] for row in observations}
    path = Path(settings['output']) / settings['operator_file'].format(rank=rank)
    path.write_text(json.dumps(observations, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
