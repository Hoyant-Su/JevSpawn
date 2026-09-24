import time

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.latentmas.adapter import HybridTransport
from jev_spawn.runtime.decoding import CapturedDecode


def padding_segments(mask):
    assert bool(((mask == 0) | (mask == 1)).all())
    assert bool((mask[:, 1:] >= mask[:, :-1]).all())
    assert bool(mask[:, -1].all())
    boundaries = sorted({0, mask.shape[1], *(mask == 0).sum(1).tolist()})
    return list(zip(boundaries, boundaries[1:]))


class ChunkedHybridTransport(HybridTransport):
    def _forward(self, embeddings, mask, cache):
        inactive = ~mask.bool().any(1)
        saved = []
        if cache is not None and embeddings.shape[1] > 1 and bool(inactive.any()):
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    saved.append((layer, layer.conv_states[0][inactive].clone(),
                                  layer.recurrent_states[0][inactive].clone()))
        output = super()._forward(embeddings, mask, cache)
        for layer, conv, recurrent in saved:
            layer.conv_states[0][inactive] = conv
            layer.recurrent_states[0][inactive] = recurrent
            assert torch.equal(layer.conv_states[0][inactive], conv)
            assert torch.equal(layer.recurrent_states[0][inactive], recurrent)
        if saved:
            self.restored_rows += int(inactive.sum())
            self.records[-1]['restored_rows'] = int(inactive.sum())
        self.records[-1]['inactive_rows'] = int(inactive.sum())
        self.records[-1]['active_rows'] = int(mask.bool().any(1).sum())
        return output

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                 past_key_values=None, **kwargs):
        if past_key_values is None:
            return super().__call__(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask, past_key_values=None, **kwargs)
        assert past_key_values is self.cache
        embeddings = self.get_input_embeddings()(input_ids) if input_ids is not None else inputs_embeds
        mask = attention_mask[:, -embeddings.shape[1]:]
        # A row never crosses its padding boundary inside a chunk, so restoring its
        # recurrent and convolution states once is equivalent to skipping every pad.
        if self.backend.config['world_size'] > 1:
            commands = self.backend.parallel_commands
            segments = commands.leader_value(padding_segments(mask) if commands.is_leader else None)
        else:
            segments = padding_segments(mask)
        for start, end in segments:
            output = self._forward(embeddings[:, start:end], mask[:, start:end], self.cache)
            self.records[-1]['prompt_chunk_start'] = start
            self.records[-1]['prompt_chunk_end'] = end
            self.records[-1]['prompt_chunk_count'] = len(segments)
        return output


class CapturedPadding(CapturedDecode):
    def configure_attention(self):
        # Role padding contains holes and uses the explicit masked SDPA in step.
        self.decode_kwargs = {}

    def __init__(self, backend, cache, batch_size, capacity, workspace_reserve,
                 graph_pool=None, graph_stream=None):
        tensors, linear = [], []
        for layer in cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                states = [layer.conv_states[0], layer.recurrent_states[0]]
                tensors.extend(states)
                linear.extend(states)
            else:
                tensors.extend([layer.keys, layer.values, layer.cumulative_length])
        cache_bytes = sum(t.numel() * t.element_size() for t in tensors)
        snapshot_bytes = sum(t.numel() * t.element_size() for t in linear)
        snapshot_bytes += sum((layer.keys[:, :, :1].numel() + layer.values[:, :, :1].numel())
                              * (layer.keys.element_size() + torch.tensor([], dtype=torch.long).element_size())
                              + layer.cumulative_length.element_size()
                              for layer in cache.layers if not isinstance(layer, LinearAttentionCacheLayerMixin))
        linear_bytes = sum(t.numel() * t.element_size() for t in linear)
        hidden_size = backend.model.get_input_embeddings().weight.shape[1]
        element_size = backend.model.get_input_embeddings().weight.element_size()
        vocab_size = backend.vocab_size
        buffers = batch_size * (16 + capacity + vocab_size * element_size + 2 * hidden_size * element_size + 1)
        free, _ = torch.cuda.mem_get_info(backend.device)
        reusable = torch.cuda.memory_reserved(backend.device) - torch.cuda.memory_allocated(backend.device)
        required = snapshot_bytes + linear_bytes + 2 * buffers + workspace_reserve
        assert free + reusable >= required, (
            f'Insufficient GPU memory for exact padding graph: need {required} additional bytes, '
            f'have {free + reusable}. CPU offload is forbidden.')
        super().__init__(backend, batch_size, capacity, graph_pool=graph_pool, graph_stream=graph_stream)
        self.cache = cache
        self.embeddings = torch.empty((batch_size, 1, hidden_size), device=backend.device,
                                      dtype=backend.model.get_input_embeddings().weight.dtype)
        self.hidden = torch.empty((batch_size, hidden_size), device=backend.device, dtype=self.embeddings.dtype)
        self.active = torch.empty(batch_size, device=backend.device, dtype=torch.bool)
        self.linear = [(state, torch.empty_like(state)) for state in linear]
        self.memory = {'cache_bytes': cache_bytes, 'linear_scratch_bytes': linear_bytes,
                       'snapshot_bytes': snapshot_bytes,
                       'workspace_reserve_bytes': workspace_reserve, 'required_extra_bytes': required}

    def step(self):
        for state, saved in self.linear:
            saved.copy_(state)
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(inputs_embeds=self.embeddings, position_ids=self.positions,
            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
            past_key_values=self.cache, use_cache=True, output_hidden_states=True, return_dict=True)
        for state, saved in self.linear:
            active = self.active.view(-1, *([1] * (state.ndim - 1)))
            torch.where(active, state, saved, out=state)
        self.hidden.copy_(output.last_hidden_state[:, -1])


class CapturedPaddingTransport(HybridTransport):
    def __init__(self, backend, decoder, warmup_steps, workspace_reserve, before_forward):
        super().__init__(backend)
        self.decoder = decoder
        self.warmup_steps, self.workspace_reserve = warmup_steps, workspace_reserve
        self.before_forward = before_forward

    def _forward(self, embeddings, mask, cache):
        self.before_forward()
        return super()._forward(embeddings, mask, self.decoder.cache if cache is None else cache)

    def run_padding(self, embeddings, mask, observer):
        history = torch.cat([self.mask, mask], dim=1)
        assert history.shape[1] <= self.backend.config['max_input_tokens']
        positions = (history.cumsum(1) - 1).clamp_min(0)[:, -mask.shape[1]:]
        initial_width = self.mask.shape[1]
        if self.decoder.padding_graph is None:
            self.decoder.padding_graph = CapturedPadding(self.backend, self.cache, embeddings.shape[0],
                self.decoder.capacity, self.workspace_reserve, graph_pool=self.decoder.graph_pool,
                graph_stream=self.decoder.graph_stream)
        graph = self.decoder.padding_graph
        assert graph.cache is self.cache
        graph.key_valid.fill_(True)
        graph.key_valid[:, :history.shape[1]].copy_(history)
        for offset in range(embeddings.shape[1]):
            self.before_forward()
            graph.embeddings.copy_(embeddings[:, offset:offset + 1])
            graph.positions.copy_(positions[:, offset:offset + 1])
            graph.active.copy_(mask[:, offset].bool())
            capture_seconds = 0.0
            if graph.graph is None:
                started = time.perf_counter()
                graph.capture(self.warmup_steps)
                capture_seconds = time.perf_counter() - started
            started = time.perf_counter()
            graph.graph.replay()
            torch.cuda.synchronize(self.backend.device)
            self.mask = history[:, :initial_width + offset + 1]
            self.last_hidden = graph.hidden
            inactive = int((~graph.active).sum())
            self.restored_rows += inactive
            self.records.append({'phase': self.phase, 'batch_size': embeddings.shape[0], 'tokens': 1,
                'seconds': time.perf_counter() - started, 'valid_tokens': int(graph.active.sum()),
                'restored_rows': inactive, 'padding_token_offset': offset,
                'decode_engine': 'cuda_graph_single_token', 'graph_capture_seconds': capture_seconds,
                'graph_input_shape': [embeddings.shape[0], 1]})
            if observer is not None:
                observer(graph.hidden, graph.positions)

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                 past_key_values=None, **kwargs):
        if past_key_values is None:
            return super().__call__(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask, past_key_values=None, **kwargs)
        assert past_key_values is self.cache
        embeddings = self.get_input_embeddings()(input_ids) if input_ids is not None else inputs_embeds
        mask = attention_mask[:, -embeddings.shape[1]:]
        if embeddings.shape[1] == 1:
            return self._forward(embeddings, mask, self.cache)
        assert bool((mask[:, 1:] >= mask[:, :-1]).all()) and bool(mask[:, -1].all())
        prefix = int((mask == 0).sum(1).max())
        if prefix:
            self.run_padding(embeddings[:, :prefix], mask[:, :prefix], observer=None)
        return self._forward(embeddings[:, prefix:], mask[:, prefix:], self.cache)
