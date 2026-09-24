import ast
from copy import copy
from importlib import import_module
import inspect
import textwrap
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, StaticLayer

from jev_spawn.infra import cached_suffix
from jev_spawn.infra.history_prefill import HistoryPrefill
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers
from tests.infra.native_chunk_checkpoint.capture import checkpoint_mapping
from tests.infra.native_chunk_checkpoint.integration import CheckpointRecorder


FLA_CHUNK = import_module('fla.ops.gated_delta_rule.chunk')


class ColdCheckpoint(CheckpointRecorder):
    def __init__(self, backend, inputs, settings):
        self.backend, self.settings = backend, settings
        lengths = inputs['attention_mask'].sum(-1).tolist()
        width = inputs['input_ids'].shape[-1]
        chunk = settings['chunk_size']
        self.boundary = (width - settings['reserved_tail_tokens']) // chunk * chunk
        self.starts = [width - length for length in lengths]
        self.rows = [row for row, start in enumerate(self.starts) if self.boundary > start]
        self.device_rows = torch.tensor(self.rows, device=backend.device)
        self.mapping = checkpoint_mapping([(width + chunk - 1) // chunk] * len(lengths),
            [(row, self.boundary // chunk) for row in self.rows],
            device=backend.device, settings=settings['capture'])
        self.native_signature = inspect.signature(inspect.unwrap(FLA_CHUNK.chunk_gated_delta_rule_fwd_h))
        self.conv, self.recurrent = {}, {}
        self.patched_forward = self.instrument_forward()

    def instrument_forward(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(gdn.gdn_forward)))
        function, = tree.body
        positions = [index for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign) and ast.unparse(node) == 'mixed_qkv = mixed_qkv.transpose(1, 2)']
        position, = positions
        function.body.insert(position + 1, ast.parse('recorder.capture_conv(module, mixed_qkv)').body[0])
        ast.fix_missing_locations(tree)
        namespace = dict(gdn.gdn_forward.__globals__, recorder=self)
        exec(compile(tree, inspect.getsourcefile(gdn.gdn_forward), 'exec'), namespace)
        return namespace[function.name]

    def capture_conv(self, module, values):
        self.active_layer = module.layer_idx
        self.conv[module.layer_idx] = values.index_select(0, self.device_rows)[
            :, :, self.boundary - module.conv_kernel_size:self.boundary].clone()

    def save(self, decoder, history):
        template = DynamicCache(config=self.backend.cache_config)
        pairs, entries = [], []

        def copied(source):
            target = torch.empty_like(source)
            pairs.append((target, source))
            return target

        for selected, row in enumerate(self.rows):
            layers = []
            for index, (original, current) in enumerate(zip(template.layers, decoder.cache.layers, strict=True)):
                names = ('keys', 'values') if type(original) is DynamicLayer else ('conv_states', 'recurrent_states')
                layer = copy(original)
                layer.__dict__ = _metadata(original if type(original) is DynamicLayer else current, names)
                if type(original) is DynamicLayer:
                    assert type(current) is StaticLayer
                    layer.dtype, layer.device = current.keys.dtype, current.keys.device
                    layer.is_initialized = current.is_initialized
                    for name in names:
                        setattr(layer, name, copied(getattr(current, name)[row:row + 1, :, self.starts[row]:self.boundary]))
                else:
                    layer.conv_states = {key: copied(self.conv[index][selected:selected + 1]) for key in current.conv_states}
                    layer.recurrent_states = {key: copied(self.recurrent[index][selected:selected + 1]) for key in current.recurrent_states}
                layers.append(layer)
            sequence = history.sequences[row][:self.boundary - self.starts[row]]
            entries.append((sequence, _with_layers(template, layers)))
        copy_states(pairs, history.settings['state_copy'])
        for sequence, state in entries:
            history.cache.store(sequence, state, None)


class CheckpointHistory(HistoryPrefill):
    def __init__(self, backend, settings):
        super().__init__(backend, settings)
        self.checkpoint = settings['checkpoint']

    def cold_prefill(self, decoder, inputs):
        recorder = ColdCheckpoint(self.backend, inputs, self.checkpoint)
        if not recorder.rows:
            return super().cold_prefill(decoder, inputs)
        trunk = self.backend.model.model.language_model
        bindings = [(layer.linear_attn, layer.linear_attn.forward) for layer in trunk.layers
                    if hasattr(layer, 'linear_attn')]
        with patch.object(FLA_CHUNK, 'chunk_gated_delta_rule_fwd_h', recorder.capture_native):
            for module, original in bindings:
                module.forward = recorder.patched_forward.__get__(module, type(module))
            token = super().cold_prefill(decoder, inputs)
            for module, original in bindings:
                module.forward = original
        recorder.save(decoder, self)
        return token

    def extend_states(self, backend, states, tails, work):
        active = [row for row, tail in enumerate(tails) if tail]
        chunk = self.checkpoint['chunk_size']
        offsets = [(len(tails[row]) - self.checkpoint['reserved_tail_tokens']) // chunk * chunk for row in active]
        offsets = [offset if offset > 0 else None for offset in offsets]
        if not any(offset is not None for offset in offsets):
            return cached_suffix.ragged_suffix(backend, states, tails, work)
        recorder = CheckpointRecorder([len(tails[row]) for row in active], offsets, backend.device, self.checkpoint)
        root_lengths = [states[row].get_seq_length() for row in active]
        split = cached_suffix.split_native_cache_at
        state_keys = {id(value): key for key, value in self.cache.entries.items()}

        def retain(cache, lengths, stops):
            prefix_widths = {stop - len(tails[row]) for row, stop in zip(active, stops, strict=True)}
            prefix_width, = prefix_widths
            snapshots = recorder.compact(cache, root_lengths, prefix_width, self.settings['state_copy'])
            for selected, snapshot in zip(recorder.rows, snapshots, strict=True):
                row = active[selected]
                sequence = (*state_keys[id(states[row])], *tails[row][:offsets[selected]])
                self.cache.store(sequence, snapshot, None)
            return split(cache, lengths, stops)

        with patch.object(gdn, 'ragged_gdn_forward', recorder.patched_forward), \
             patch.object(FLA_CHUNK, 'chunk_gated_delta_rule_fwd_h', recorder.capture_native), \
             patch.object(cached_suffix, 'split_native_cache_at', retain):
            return cached_suffix.ragged_suffix(backend, states, tails, work)
