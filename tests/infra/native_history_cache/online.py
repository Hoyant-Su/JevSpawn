from importlib import import_module
import json
from pathlib import Path
import time
from unittest.mock import patch

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra import cached_suffix
from jev_spawn.infra.history_cache import HistoryTail
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_chunk_checkpoint.integration import CheckpointRecorder


FLA_CHUNK = import_module('fla.ops.gated_delta_rule.chunk')


class RetainedRoots:
    def __init__(self, roots, entries, mapping):
        self.roots, self.entries, self.mapping = roots, entries, mapping

    def get_many(self, prefixes, prefill):
        keys = [self.mapping[tuple(prefix)] for prefix in prefixes]
        states, hits = self.roots.get_many(keys, prefill)
        self.reused_root_tokens = sum(len(root) for root, hit in zip(keys, hits, strict=True) if hit)
        return [self.entries[root][1] if len(prefix) > len(root) else state
                for prefix, root, state in zip(prefixes, keys, states, strict=True)], hits


class OnlineHistoryTail(HistoryTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings, history_settings):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings, history_settings)
        self.capture_settings = json.loads(Path(history_settings['capture_settings']).read_text())
        self.extend_states = self.extend_with_checkpoint

    def extend_with_checkpoint(self, backend, states, tails, work):
        if not self.pending_capture:
            return cached_suffix.ragged_suffix(backend, states, tails, work)
        self.pending_capture = False
        active = [index for index, tail in enumerate(tails) if tail]
        offsets = [self.boundaries[self.call_roots[index]] - states[index].get_seq_length()
                   for index in active]
        offsets = [offset if 0 < offset < len(tails[index]) else None
                   for index, offset in zip(active, offsets, strict=True)]
        if not any(offset is not None for offset in offsets):
            return cached_suffix.ragged_suffix(backend, states, tails, work)
        recorder = CheckpointRecorder([len(tails[index]) for index in active], offsets, backend.device,
            {'chunk_size': self.history_settings['chunk_tokens'], 'capture': self.capture_settings})
        root_lengths = [states[index].get_seq_length() for index in active]
        split = cached_suffix.split_native_cache_at

        def retain(cache, lengths, stops):
            prefix_widths = {stop - len(tails[index]) for index, stop in zip(active, stops, strict=True)}
            prefix_width, = prefix_widths
            snapshots = recorder.compact(cache, root_lengths, prefix_width, self.state_copy_settings)
            for row, snapshot in zip(recorder.rows, snapshots, strict=True):
                root = self.call_roots[active[row]]
                self.history.entries[root] = (self.checkpoint_tokens[root], snapshot)
                self.history.entries.move_to_end(root)
            while len(self.history.entries) > self.runtime.root_batch_size:
                self.history.entries.popitem(last=False)
            return split(cache, lengths, stops)

        with patch.object(gdn, 'ragged_gdn_forward', recorder.patched_forward), \
             patch.object(FLA_CHUNK, 'chunk_gated_delta_rule_fwd_h', recorder.capture_native), \
             patch.object(cached_suffix, 'split_native_cache_at', retain):
            return cached_suffix.ragged_suffix(backend, states, tails, work)

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        started = time.perf_counter()
        block = self.history_settings['chunk_tokens']
        lengths, mapping, roots, groups = [], {}, [], {}
        self.boundaries, self.checkpoint_tokens = {}, {}
        for request, base in zip(requests, base_lengths, strict=True):
            tokens = request.admitted.tokens
            root = tuple(tokens[:base])
            system, = [m['content'] for m in request.messages if m['role'] == 'system']
            boundary = common_prefix([tokens, self.history_tokens(
                system, request.field['context'], request.field['history'])])
            length = base + (boundary - base) // block * block
            self.boundaries[root] = length
            self.checkpoint_tokens[root] = tuple(tokens[:length])
            prior = self.history.entries.get(root)
            prefix = prior[0] if prior is not None and tuple(tokens[:len(prior[0])]) == prior[0] else root
            lengths.append(len(prefix))
            mapping[prefix] = root
            roots.append(root)
            groups[(request.task_id, request.field['context'])] = (root, len(prefix))
        self.call_roots = [root for root, _ in groups.values()]
        self.pending_capture = True
        retained = RetainedRoots(base_cache, self.history.entries, mapping)
        result = StableFiniteGraphTail.score(self, requests, lengths, retained, physical_batch_size)
        result.update(reused_state_tokens=sum(length - len(root) for root, length in groups.values()),
            reused_root_tokens=retained.reused_root_tokens,
            root_prefix_tokens=[len(root) for root, _ in groups.values()],
            persistent_prefix_scope='task_root_and_online_native_history',
            elapsed_seconds=time.perf_counter() - started)
        return result
