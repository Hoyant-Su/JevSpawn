from collections import OrderedDict
import time

import torch

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra import finite_batch
from jev_spawn.infra.cached_suffix import grouped_suffix, ragged_suffix
from jev_spawn.infra.finite_batch import score_finite_with_tail
from jev_spawn.runtime.finite_decoding import CapturedFinite


class FiniteGraphTail:
    extend_states = staticmethod(grouped_suffix)

    def __init__(self, backend, runtime, prefix_cache, state_copy_settings):
        self.backend, self.runtime, self.prefix_cache = backend, runtime, prefix_cache
        self.state_copy_settings = state_copy_settings
        self.graphs = OrderedDict()
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=backend.device)

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        result = score_finite_with_tail(self.backend, requests, base_lengths, base_cache,
                                        self, self.extend_states, common_prefix, self.extend_prefixes, physical_batch_size)
        result['graph_layout'] = dict(self.last_layout)
        return result

    def extend_prefixes(self, backend, states, prefixes, bases, work, extend_states):
        return finite_batch.extend_prefixes(backend, states, prefixes, bases, work, extend_states)

    def graph_layout(self, sequences, native_ids):
        block = self.runtime.graph_cache_block_tokens
        capacity = (max(map(len, sequences)) + block - 1) // block * block
        return (len(sequences), capacity, tuple(native_ids)), len(sequences), capacity, tuple(native_ids)

    def graph_inputs(self, states, last_tokens, physical_rows):
        assert len(states) == physical_rows
        return states, last_tokens

    def graph_outputs(self, logits, active_rows, native_ids):
        return logits

    def __call__(self, backend, sequences, prefixes, owners, states, suffixes, native_ids):
        started = time.perf_counter()
        work = {'suffix_pack_seconds': 0.0, 'suffix_forward_seconds': 0.0, 'suffix_split_seconds': 0.0,
                'computed_input_tokens': 0, 'padded_input_tokens': 0, 'graph_replays': 1, 'graph_captures': 0}
        full_prefixes = [sequence[:-1] for sequence in sequences]
        source_rows = {tuple(prefix): index for index, prefix in enumerate(full_prefixes)}

        def complete(missing):
            rows = [source_rows[tuple(prefix)] for prefix in missing]
            source_states = [states[owners[row]] for row in rows]
            tails = [suffixes[row][:-1] for row in rows]
            return self.extend_states(backend, source_states, tails, work)

        final_states, hits = self.prefix_cache.get_many(full_prefixes, complete)
        phases = {'suffix_prefill_seconds': time.perf_counter() - started}
        phase = time.perf_counter()
        key, physical_rows, capacity, graph_ids = self.graph_layout(sequences, native_ids)
        self.last_layout = {'key': key, 'active_rows': len(sequences), 'physical_rows': physical_rows,
                            'capacity': capacity, 'selected_head_rows': len(graph_ids),
                            'valid_output_mask': [index < len(sequences) for index in range(physical_rows)]}
        if key not in self.graphs:
            if len(self.graphs) == self.runtime.graph_cache_size:
                self.graphs.popitem(last=False)
            self.graphs[key] = CapturedFinite(backend, physical_rows, capacity, graph_ids, self.pool, self.stream,
                                                state_copy_settings=self.state_copy_settings)
        decoder = self.graphs[key]
        self.graphs.move_to_end(key)
        load_states, load_tokens = self.graph_inputs(final_states, [sequence[-1] for sequence in sequences], physical_rows)
        decoder.load(load_states, load_tokens)
        phases['cache_load_seconds'] = time.perf_counter() - phase
        phases['capture_seconds'] = 0.0
        if decoder.graph is None:
            phase = time.perf_counter()
            decoder.capture(self.runtime.graph_warmup_steps)
            phases['capture_seconds'] = time.perf_counter() - phase
            work['graph_captures'] += 1
        phase = time.perf_counter()
        decoder.graph.replay()
        phases['graph_replay_seconds'] = time.perf_counter() - phase
        phases['tiles_seconds'] = time.perf_counter() - started
        phases['readout_seconds'] = 0.0
        work['computed_input_tokens'] += len(sequences)
        work['padded_input_tokens'] += physical_rows
        self.last_logits = self.graph_outputs(decoder.logits, len(sequences), native_ids)
        return self.last_logits, work, phases


class RaggedFiniteGraphTail(FiniteGraphTail):
    extend_states = staticmethod(ragged_suffix)
