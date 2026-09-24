import argparse
from collections import defaultdict
import json
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.history_cache.replay_fused_conv import FusedCheckpoint


class RecordedCheckpoint(FusedCheckpoint):
    def get_many(self, prefixes, prefill):
        self.requested_prefix_lengths = list(map(len, prefixes))
        return super().get_many(prefixes, prefill)

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        result = super().score(requests, base_lengths, base_cache, physical_batch_size)
        groups = defaultdict(list)
        for request, root in zip(requests, base_lengths, strict=True):
            groups[(request.task_id, request.field['context'])].append(root)
        counts = [len(roots) for roots in groups.values()]
        roots = [lengths[0] for lengths in groups.values()]
        result.update(history_root_lengths=roots, history_effective_prefix_lengths=self.requested_prefix_lengths,
            history_hit_rows=sum(count for count, root, prefix in zip(
                counts, roots, self.requested_prefix_lengths, strict=True) if prefix > root),
            history_eligible_rows=sum(counts),
            history_cache_bytes=self.manager.cache.bytes)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    if settings['same_pass_history']:
        def make_tail(service, backend, shared, inference):
            return RecordedCheckpoint(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                inference['state_copy'], inference['graph_shape'], settings['history_cache'])
        StableGraphFiniteService._make_tail = make_tail
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
