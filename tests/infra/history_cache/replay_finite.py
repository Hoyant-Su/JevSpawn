import argparse
import json
from pathlib import Path

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.history_cache import HistoryTail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.infra.history_prefill import HistoryPrefill
from tests.infra.native_history_cache import replay


class SharedHistoryFiniteTail(HistoryTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings, history_settings, manager):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings, history_settings)
        self.manager = manager

    def get_many(self, prefixes, prefill):
        hits = [self.manager.cache.prefix(prefix) == tuple(prefix) for prefix in prefixes]
        states = self.manager.compute(prefixes)
        self.cache_work = self.manager.records.pop()
        return states, hits

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        lengths = []
        block = self.history_settings['chunk_tokens']
        for request, base in zip(requests, base_lengths, strict=True):
            system, = [message['content'] for message in request.messages if message['role'] == 'system']
            tokens = request.admitted.tokens
            boundary = common_prefix([tokens, self.history_tokens(
                system, request.field['context'], request.field['history'])])
            assert boundary >= base
            lengths.append(base + (boundary - base) // block * block)
        result = StableFiniteGraphTail.score(self, requests, lengths, self, physical_batch_size)
        for name in ('computed_input_tokens', 'padded_input_tokens'):
            result[name] += self.cache_work[name]
        result.update(reused_state_tokens=self.cache_work['matched_prefix_tokens'],
            reused_root_tokens=self.history_settings['initial_count'],
            persistent_prefix_scope='shared_exact_task_and_history_prefixes',
            history_prefill=dict(self.cache_work), history_checkpoint_tokens=lengths)
        for request in requests:
            prefix = tuple(request.admitted.tokens[:-1])
            self.manager.cache.store(prefix, self.prefix_cache.entries[(prefix,)], None)
        return result

class SharedTail(SharedHistoryFiniteTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings, history_settings):
        manager = HistoryPrefill(backend, json.loads(Path(history_settings['manager']).read_text()))
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings, history_settings, manager)


def reset_history(tail):
    tail.manager.cache.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    replay.HistoryTail = SharedTail
    replay.reset_history = reset_history
    replay.run(json.loads(parser.parse_args().config.read_text()))
