from collections import defaultdict
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import patch

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra import cached_suffix
from jev_spawn.infra.history_cache import HistoryTail
from jev_spawn.infra.history_prefill import HistoryPrefill
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_chunk_checkpoint.integration import CheckpointRecorder


FLA_CHUNK = import_module('fla.ops.gated_delta_rule.chunk')


class SharedCheckpointTail(HistoryTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings, history_settings):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings, history_settings)
        self.manager = HistoryPrefill(backend, json.loads(Path(history_settings['manager']).read_text()))
        self.capture_settings = json.loads(Path(history_settings['capture_settings']).read_text())
        self.extend_states = self.extend_with_checkpoint

    def get_many(self, prefixes, prefill):
        keys = list(map(tuple, prefixes))
        hits = [key in self.manager.cache.entries for key in keys]
        retained = {key: self.manager.cache.read(key) for key, hit in zip(keys, hits, strict=True) if hit}
        missing = [list(key) for key in dict.fromkeys(keys) if key not in self.manager.cache.entries]
        states, root_hits = self.roots.get_many(missing, prefill) if missing else ([], [])
        reused_roots = sum(len(key) for key, hit in zip(missing, root_hits, strict=True) if hit)
        for key, state in zip(missing, states, strict=True):
            self.manager.cache.store(key, state, None)
            retained[tuple(key)] = state
        self.reused_roots = reused_roots
        return [retained[key] for key in keys], hits

    def extend_with_checkpoint(self, backend, states, tails, work):
        if not self.pending_capture:
            return cached_suffix.ragged_suffix(backend, states, tails, work)
        self.pending_capture = False
        active = [row for row, tail in enumerate(tails) if tail]
        offsets = [self.boundaries[row] - states[row].get_seq_length() for row in active]
        offsets = [offset if 0 < offset < len(tails[row]) else None
                   for row, offset in zip(active, offsets, strict=True)]
        if not any(offset is not None for offset in offsets):
            return cached_suffix.ragged_suffix(backend, states, tails, work)
        recorder = CheckpointRecorder([len(tails[row]) for row in active], offsets, backend.device,
            {'chunk_size': self.history_settings['chunk_tokens'], 'capture': self.capture_settings})
        roots = [states[row].get_seq_length() for row in active]
        split = cached_suffix.split_native_cache_at

        def retain(cache, lengths, stops):
            widths = {stop - len(tails[row]) for row, stop in zip(active, stops, strict=True)}
            width, = widths
            snapshots = recorder.compact(cache, roots, width, self.state_copy_settings)
            for selected, state in zip(recorder.rows, snapshots, strict=True):
                row = active[selected]
                self.manager.cache.store(self.checkpoints[row], state, None)
            return split(cache, lengths, stops)

        with patch.object(gdn, 'ragged_gdn_forward', recorder.patched_forward), \
             patch.object(FLA_CHUNK, 'chunk_gated_delta_rule_fwd_h', recorder.capture_native), \
             patch.object(cached_suffix, 'split_native_cache_at', retain):
            return cached_suffix.ragged_suffix(backend, states, tails, work)

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        groups = defaultdict(list)
        for row, request in enumerate(requests):
            groups[(request.task_id, request.field['context'])].append(row)
        lengths = list(base_lengths)
        self.boundaries, self.checkpoints = [], []
        reused = self.history_settings['initial_count']
        block = self.history_settings['chunk_tokens']
        for rows in groups.values():
            request = requests[rows[0]]
            tokens = request.admitted.tokens
            shared = common_prefix([requests[row].admitted.tokens[:-1] for row in rows])
            prior = self.manager.cache.prefix(tokens[:shared])
            base = base_lengths[rows[0]]
            prefix = prior if len(prior) >= base else tokens[:base]
            reused += len(prior)
            system, = [m['content'] for m in request.messages if m['role'] == 'system']
            boundary = min(shared, common_prefix([tokens, self.history_tokens(
                system, request.field['context'], request.field['history'])]))
            stop = len(prefix) + max((boundary - len(prefix)) // block, 0) * block
            self.boundaries.append(stop)
            self.checkpoints.append(tokens[:stop])
            for row in rows:
                lengths[row] = len(prefix)
        self.roots = base_cache
        self.pending_capture = True
        result = StableFiniteGraphTail.score(self, requests, lengths, self, physical_batch_size)
        result.update(reused_state_tokens=reused, reused_root_tokens=self.reused_roots,
            persistent_prefix_scope='shared_same_pass_native_history')
        for request in requests:
            key = tuple(request.admitted.tokens[:-1])
            self.manager.cache.store(key, self.prefix_cache.entries[(key,)], None)
        return result
