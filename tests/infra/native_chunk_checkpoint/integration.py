import ast
from copy import copy
import inspect
import textwrap

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from transformers.cache_utils import DynamicLayer

from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers
from tests.infra.native_chunk_checkpoint.capture import capture_h, checkpoint_mapping


class CheckpointRecorder:
    def __init__(self, lengths, offsets, device, settings):
        chunk_size = settings['chunk_size']
        assert len(lengths) == len(offsets)
        assert all(0 < offset < length and offset % chunk_size == 0
                   for length, offset in zip(lengths, offsets, strict=True) if offset is not None)
        self.settings = settings
        self.offsets = offsets
        self.rows = [row for row, offset in enumerate(offsets) if offset is not None]
        self.device_rows = torch.tensor(self.rows, device=device)
        self.device_offsets = torch.tensor([offsets[row] for row in self.rows], device=device)
        self.mapping = checkpoint_mapping(
            [(length + chunk_size - 1) // chunk_size for length in lengths],
            [(row, offsets[row] // chunk_size) for row in self.rows],
            device=device, settings=settings['capture'])
        self.native_signature = inspect.signature(inspect.unwrap(chunk_gated_delta_rule_fwd_h))
        self.conv, self.recurrent = {}, {}
        self.patched_forward = self.instrument_forward()

    def instrument_forward(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(ragged_gdn_forward)))
        function, = tree.body
        positions = [index for index, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 'history' for target in node.targets)]
        position, = positions
        function.body.insert(position + 1, ast.parse('recorder.capture_conv(module, history)').body[0])
        ast.fix_missing_locations(tree)
        namespace = dict(ragged_gdn_forward.__globals__, recorder=self)
        exec(compile(tree, inspect.getsourcefile(ragged_gdn_forward), 'exec'), namespace)
        return namespace[function.name]

    def capture_conv(self, module, history):
        self.active_layer = module.layer_idx
        positions = self.device_offsets[:, None, None] + torch.arange(
            module.conv_kernel_size, device=history.device)[None, None, :]
        self.conv[module.layer_idx] = history[self.device_rows[:, None, None],
            torch.arange(history.shape[1], device=history.device)[None, :, None], positions]

    def capture_native(self, *args, **kwargs):
        bound = self.native_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        assert bound.arguments['chunk_size'] == self.settings['chunk_size']
        h, values, final, selected = capture_h(
            **bound.arguments, checkpoint_rows=self.mapping,
            checkpoint_count=len(self.rows), settings=self.settings['capture'])
        self.recurrent[self.active_layer] = selected
        return h, values, final

    def compact(self, cache, root_lengths, prefix_width, copy_settings):
        pairs, rows = [], []

        def allocate(source):
            destination = torch.empty_like(source)
            pairs.append((destination, source))
            return destination

        for selected, row in enumerate(self.rows):
            root, offset = root_lengths[row], self.offsets[row]
            layers = []
            for index, original in enumerate(cache.layers):
                names = ('keys', 'values') if type(original) is DynamicLayer else ('conv_states', 'recurrent_states')
                layer = copy(original)
                layer.__dict__ = _metadata(original, names)
                if type(original) is DynamicLayer:
                    for name in names:
                        source = getattr(original, name)[row:row + 1, :, prefix_width - root:prefix_width + offset]
                        setattr(layer, name, allocate(source))
                else:
                    assert list(original.conv_states) == list(original.recurrent_states) == [0]
                    layer.conv_states = {0: allocate(self.conv[index][selected:selected + 1])}
                    layer.recurrent_states = {0: allocate(self.recurrent[index][selected:selected + 1])}
                layers.append(layer)
            rows.append(_with_layers(cache, layers))
        copy_states(pairs, copy_settings)
        return rows
