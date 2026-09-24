import time

import torch
import torch.nn.functional as F

from baselines.common import graph_finite_service, parallel_service
from baselines.common.graph_finite_service import GraphFiniteService
from jev_spawn.algo.structured import common_prefix, padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.infra.finite_batch import extend_prefixes, score_finite_with_tail
from jev_spawn.runtime.native_cache_batch import pack_native_caches
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


class DirectFiniteTail:
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings):
        self.backend = backend
        self.prefix_cache = prefix_cache
        self.graphs = {}

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        base_by_tokens = {request.admitted.tokens: length
                          for request, length in zip(requests, base_lengths, strict=True)}

        def prefix_length(sequences):
            if len(sequences) == 1:
                return base_by_tokens[tuple(sequences[0])]
            return common_prefix(sequences)

        return score_finite_with_tail(self.backend, requests, base_lengths, base_cache,
                                     self, ragged_suffix, prefix_length, extend_prefixes, physical_batch_size)

    def __call__(self, backend, sequences, prefixes, owners, states, suffixes, native_ids):
        started = time.perf_counter()
        selected = [states[owners[index]] for index in range(len(sequences))]
        cache, prefix_mask = pack_native_caches(selected)
        ids, mask = padded(suffixes, backend.tokenizer.pad_token_id, backend.device, 'right')
        full_mask = torch.cat((prefix_mask, mask), dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        lengths = list(map(len, suffixes))
        descriptor = RaggedSuffix(lengths, backend.device)
        attention = RaggedCacheAttention(descriptor, [state.get_seq_length() for state in selected])
        output = backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=cache, use_cache=True,
            ragged_suffix=descriptor, decode_attention=attention)
        ends = torch.tensor(lengths, device=backend.device) - 1
        hidden = output.last_hidden_state[torch.arange(len(sequences), device=backend.device), ends]
        self.last_logits = F.linear(hidden.float(), backend.finite_output_weights[:len(native_ids)])
        work = {'computed_input_tokens': sum(lengths), 'padded_input_tokens': ids.numel(),
                'graph_replays': 0, 'graph_captures': 0}
        phases = {'tiles_seconds': time.perf_counter() - started,
                  'readout_seconds': 0.0, 'capture_seconds': 0.0}
        return self.last_logits, work, phases


class DirectFiniteService(GraphFiniteService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(finite_decode_engine='native_full_suffix_finite',
            finite_graph_state='Native ragged suffix outputs supply selected-head logits directly')

    def _make_tail(self, backend, shared, settings):
        return DirectFiniteTail(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                                settings['state_copy'], settings['graph_shape'])

    def _execution_trace(self, result, shapes):
        return {'graph_capture_seconds': result['timings']['capture_seconds'],
                'decode_engine': 'native_full_suffix_finite', 'graph_replays': result['graph_replays'],
                'graph_captures': result['graph_captures']}


def install():
    original = graph_finite_service.StableGraphFiniteService
    source = original.__module__ + '.' + original.__qualname__
    target = DirectFiniteService.__module__ + '.' + DirectFiniteService.__qualname__
    parallel_service.SETTINGS['prefix_caches'][target] = parallel_service.SETTINGS['prefix_caches'][source]
    graph_finite_service.StableGraphFiniteService = DirectFiniteService
