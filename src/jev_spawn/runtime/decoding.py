import time

import torch
from transformers import StaticCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin, StaticLayer

from jev_spawn.infra.cached_attention import GroupedDecodeAttention


class CapturedDecode:
    @classmethod
    def from_rows(cls, backend, rows, key_positions, graph_pool, graph_stream):
        decoder = cls.__new__(cls)
        decoder.backend = backend
        decoder.trunk = backend.model.model.language_model
        decoder.capacity = len(key_positions)
        decoder.cache = rows.cache
        decoder.ids, decoder.positions = rows.ids, rows.positions
        decoder.key_valid, decoder.logits = rows.key_valid, rows.logits
        decoder.key_positions = key_positions
        decoder.graph = None
        decoder.graph_pool, decoder.graph_stream = graph_pool, graph_stream
        decoder.configure_attention()
        return decoder

    def __init__(self, backend, batch_size, capacity, arena=None, graph_pool=None, graph_stream=None):
        self.backend = backend
        self.trunk = backend.model.model.language_model
        self.capacity = capacity
        self.cache = (StaticCache(config=backend.cache_config, max_cache_len=capacity) if arena is None
                      else arena.bind(batch_size, capacity))
        self.ids = torch.empty((batch_size, 1), device=backend.device, dtype=torch.long)
        self.positions = torch.empty_like(self.ids)
        self.key_valid = torch.ones((batch_size, capacity), device=backend.device, dtype=torch.bool)
        self.key_positions = torch.arange(capacity, device=backend.device)
        self.logits = torch.empty((batch_size, backend.vocab_size),
                                  device=backend.device, dtype=backend.model.lm_head.weight.dtype)
        self.graph = None
        self.graph_pool, self.graph_stream = graph_pool, graph_stream
        self.configure_attention()

    def configure_attention(self):
        self.leftpad = torch.empty(self.ids.shape[0], device=self.ids.device, dtype=torch.int32)
        self.decode_kwargs = {}
        if self.backend.config['execution'] == 'qwen35_optimized':
            self.decode_kwargs['decode_attention'] = GroupedDecodeAttention(
                self.cache, self.leftpad, self.backend.config['decode_attention_splits'])

    def validate_attention(self):
        if self.decode_kwargs:
            lengths = self.cache.get_seq_length().expand(self.ids.shape[0])
            self.leftpad.copy_(self.key_valid.to(torch.int32).argmax(-1))
            positions = self.key_positions[None]
            visible = positions < lengths[:, None]
            expected = visible & (positions >= self.leftpad[:, None])
            assert bool((lengths > self.leftpad).all()) and torch.equal(self.key_valid & visible, expected), (
                'Grouped cache attention requires a nonempty contiguous valid interval per row; interior padding is unsupported.')

    def attention_arguments(self):
        if self.decode_kwargs:
            # Arena graphs survive row replacement; derive metadata from their current GPU masks.
            self.leftpad.copy_(self.key_valid.to(torch.int32).argmax(-1))
        return self.decode_kwargs

    def prefill(self, inputs):
        self.cache.reset()
        width = inputs['input_ids'].shape[1]
        self.key_valid.fill_(True)
        self.key_valid[:, :width].copy_(inputs['attention_mask'])
        positions = inputs['attention_mask'].cumsum(1) - 1
        positions.masked_fill_(inputs['attention_mask'] == 0, 1)
        output = self.trunk(**inputs, position_ids=positions, past_key_values=self.cache, use_cache=True)
        logits = self.backend.model.lm_head(output.last_hidden_state[:, -1])
        self.logits.copy_(logits)
        self.ids.copy_(logits.argmax(-1)[:, None])
        self.positions.copy_(inputs['attention_mask'].sum(1)[:, None])
        self.validate_attention()
        return self.ids[:, 0].clone()

    def step(self):
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
                            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
                            past_key_values=self.cache, use_cache=True, **self.attention_arguments())
        self.logits.copy_(self.backend.model.lm_head(output.last_hidden_state[:, -1]))
        self.ids.copy_(self.logits.argmax(-1)[:, None])
        self.positions.add_(1)

    def capture_snapshot(self):
        assert self.ids.shape[1] == 1
        state = [self.ids, self.positions, self.logits]
        kv = []
        full_bytes = sum(t.numel() * t.element_size() for t in state)
        for layer in self.cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                tensors = [*layer.conv_states.values(), *layer.recurrent_states.values()]
                state.extend(tensors)
            else:
                assert isinstance(layer, StaticLayer)
                tensors = [layer.keys, layer.values, layer.cumulative_length]
                state.append(layer.cumulative_length)
                positions = layer.cumulative_length.expand(self.ids.shape[0])[:, None, None, None]
                assert bool(((positions >= 0) & (positions < self.capacity)).all())
                for tensor in [layer.keys, layer.values]:
                    indices = positions.expand(-1, tensor.shape[1], 1, tensor.shape[3]).clone()
                    # Both StaticLayer and RowStaticLayer overwrite one position per row.
                    kv.append((tensor, indices, tensor.gather(2, indices)))
            full_bytes += sum(t.numel() * t.element_size() for t in tensors)
        saved = [tensor.clone() for tensor in state]
        self.capture_memory = {'full_state_bytes': full_bytes,
            'saved_state_bytes': sum(t.numel() * t.element_size() for t in saved)
                + sum((indices.numel() * indices.element_size() + values.numel() * values.element_size())
                      for _, indices, values in kv),
            'kv_positions_per_row': self.ids.shape[1], 'snapshot_device': str(self.ids.device)}
        return state, saved, kv

    @staticmethod
    def restore_capture_snapshot(snapshot):
        state, saved, kv = snapshot
        for destination, source in zip(state, saved, strict=True):
            destination.copy_(source)
        for destination, indices, source in kv:
            destination.scatter_(2, indices, source)

    def capture(self, warmup_steps):
        self.validate_attention()
        snapshot = self.capture_snapshot()

        def reset():
            self.restore_capture_snapshot(snapshot)

        stream = self.graph_stream if self.graph_stream is not None else torch.cuda.Stream(device=self.backend.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                reset()
                self.step()
            reset()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream, pool=self.graph_pool):
            self.step()
        reset()
        torch.cuda.synchronize()

    def generate_tokens(self, inputs, options, stopping, warmup_steps, on_tokens=None, between_steps=None):
        token = self.prefill(inputs)
        return self.decode_tokens(inputs, options, stopping, warmup_steps, token, on_tokens, between_steps)

    def decode_tokens(self, inputs, options, stopping, warmup_steps, token, on_tokens=None, between_steps=None):
        events = []
        batch, width = inputs['input_ids'].shape
        output = torch.full((batch, width + options['max_new_tokens']), options['pad_token_id'],
                            device=self.backend.device, dtype=torch.long)
        output[:, :width].copy_(inputs['input_ids'])
        self.validate_attention()
        capture_seconds = 0.0
        if self.graph is None:
            started = time.perf_counter()
            self.capture(warmup_steps)
            capture_seconds = time.perf_counter() - started
        finished = torch.zeros(batch, dtype=torch.bool, device=self.backend.device)
        delivered = set()
        for step in range(options['max_new_tokens']):
            if step:
                self.graph.replay()
                token = self.ids[:, 0].clone()
            if options['do_sample']:
                probabilities = (self.logits.float() / options['temperature']).softmax(-1)
                token = torch.multinomial(probabilities, 1).squeeze(1)
            token = torch.where(finished, options['pad_token_id'], token)
            self.ids.copy_(token[:, None])
            output[:, width + step].copy_(token)
            finished |= stopping(output[:, :width + step + 1], None)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append(event)
            if on_tokens is None:
                complete = bool(finished.all())
            else:
                completed = {index for index, done in enumerate(finished.tolist()) if done}
                newly_finished = sorted(completed - delivered)
                if newly_finished:
                    on_tokens(output[:, width:width + step + 1], newly_finished)
                delivered = completed
                complete = len(delivered) == batch
            if complete:
                break
            if between_steps is not None:
                between_steps()
        return output[:, :width + step + 1], events, capture_seconds
