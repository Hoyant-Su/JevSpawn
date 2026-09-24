import ast
from functools import cache, partial
import inspect
import textwrap

import torch
import triton
import triton.language as tl

from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward
from tests.infra.native_chunk_checkpoint.integration import CheckpointRecorder


@triton.jit
def extract_conv(History, Final, Selected, Lengths, Offsets, Slots,
                 CHANNELS: tl.constexpr, WINDOW: tl.constexpr,
                 HISTORY_ROW: tl.constexpr, HISTORY_CHANNEL: tl.constexpr,
                 HISTORY_TIME: tl.constexpr, FINAL_ROW: tl.constexpr,
                 FINAL_CHANNEL: tl.constexpr, FINAL_TIME: tl.constexpr,
                 UNSELECTED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < CHANNELS * WINDOW
    channel, step = index // WINDOW, index % WINDOW
    stop = tl.load(Lengths + row)
    origin = row * HISTORY_ROW + channel * HISTORY_CHANNEL
    value = tl.load(History + origin + (stop + step) * HISTORY_TIME, valid, other=0)
    tl.store(Final + row * FINAL_ROW + channel * FINAL_CHANNEL + step * FINAL_TIME, value, valid)
    slot = tl.load(Slots + row)
    if slot != UNSELECTED:
        stop = tl.load(Offsets + row)
        value = tl.load(History + origin + (stop + step) * HISTORY_TIME, valid, other=0)
        tl.store(Selected + slot * CHANNELS * WINDOW + index, value, valid)


@cache
def instrumented_forward():
    tree = ast.parse(textwrap.dedent(inspect.getsource(ragged_gdn_forward)))
    function, = tree.body
    function.args.kwonlyargs.append(ast.arg(arg='recorder'))
    function.args.kw_defaults.append(None)
    positions = [index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == 'positions' for target in node.targets)]
    index, = positions
    gather = function.body[index + 1]
    assert isinstance(gather, ast.Expr) and isinstance(gather.value, ast.Call)
    assert isinstance(gather.value.func, ast.Attribute) and gather.value.func.attr == 'copy_'
    function.body[index:index + 2] = ast.parse(
        'recorder.capture_conv(module, history, ragged_suffix, layer.conv_states[0])').body
    ast.fix_missing_locations(tree)
    namespace = dict(ragged_gdn_forward.__globals__)
    exec(compile(tree, inspect.getsourcefile(ragged_gdn_forward), 'exec'), namespace)
    return namespace[function.name]


class FusedConvRecorder(CheckpointRecorder):
    def __init__(self, lengths, offsets, device, settings):
        super().__init__(lengths, offsets, device, settings)
        selected = {row: index for index, row in enumerate(self.rows)}
        empty = settings['capture']['unselected_row']
        self.slots = torch.tensor([selected.get(row, empty) for row in range(len(lengths))],
                                  device=device, dtype=getattr(torch, settings['capture']['index_dtype']))
        self.offsets_gpu = torch.tensor([empty if offset is None else offset for offset in offsets],
                                        device=device, dtype=self.slots.dtype)

    def instrument_forward(self):
        return partial(instrumented_forward(), recorder=self)

    def capture_conv(self, module, history, descriptor, final):
        self.active_layer = module.layer_idx
        batch, channels, _ = history.shape
        window = module.conv_kernel_size
        selected = history.new_empty((len(self.rows), channels, window))
        settings = self.settings['capture']
        extract_conv[(triton.cdiv(channels * window, settings['block_size']), batch)](
            history, final, selected, descriptor.lengths, self.offsets_gpu, self.slots,
            channels, window, *history.stride(), *final.stride(), settings['unselected_row'],
            settings['block_size'], num_warps=settings['num_warps'], num_stages=settings['num_stages'])
        self.conv[module.layer_idx] = selected
