# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Upstream wrapper: fla/ops/common/chunk_delta_h.py, MIT licensed.
# https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE
# Native h keeps its dtype; selected FP32 recurrence registers are stored in-place.

import ast
from functools import cache
import inspect
from itertools import accumulate
import textwrap

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

from tests.infra.native_chunk_checkpoint.fused_kernel import fused_checkpoint_kernel


@cache
def fused_wrapper():
    original = inspect.unwrap(chunk_gated_delta_rule_fwd_h)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function, = tree.body
    function.decorator_list = []
    for name in ('checkpoint_rows', 'checkpoint_count', 'checkpoint_dtype', 'unselected_row'):
        function.args.kwonlyargs.append(ast.arg(arg=name))
        function.args.kw_defaults.append(None)
    launches = [(index, node.value) for index, node in enumerate(function.body)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Subscript)]
    (position, launch), = launches
    for name, value in [('selected', 'selected'), ('checkpoint_rows', 'checkpoint_rows'),
                        ('UNSELECTED', 'unselected_row')]:
        launch.keywords.append(ast.keyword(arg=name, value=ast.Name(id=value, ctx=ast.Load())))
    function.body[position:position] = ast.parse(
        'selected = h.new_empty((checkpoint_count, *h.shape[2:]), dtype=checkpoint_dtype)').body
    function.body[-1].value.elts.append(ast.Name(id='selected', ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__,
                     chunk_gated_delta_rule_fwd_kernel_h_blockdim64=fused_checkpoint_kernel)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    return namespace[function.name]


def checkpoint_mapping(chunk_counts, checkpoints, *, device, settings):
    """Map (sequence, chunk-start index) pairs to caller-ordered output rows."""
    offsets = list(accumulate(chunk_counts, initial=0))
    flat_indices = [offsets[sequence] + chunk for sequence, chunk in checkpoints]
    assert len(flat_indices) == len(set(flat_indices))
    assert all(0 <= sequence < len(chunk_counts) and 0 <= chunk < chunk_counts[sequence]
               for sequence, chunk in checkpoints)
    mapping = [settings['unselected_row']] * offsets[-1]
    for row, index in enumerate(flat_indices):
        mapping[index] = row
    return torch.tensor(mapping, dtype=getattr(torch, settings['index_dtype']), device=device)


def capture_h(k, w, u, *, g, gk, initial_state, output_final_state, chunk_size,
              save_new_value, state_v_first, cu_seqlens, cu_seqlens_cpu, chunk_indices,
              checkpoint_rows, checkpoint_count, settings):
    """Capture selected FP32 chunk-start states within the native recurrence.

    Checkpoint j follows j * chunk_size suffix tokens; final_state is the native
    state after the last chunk. The h allocation and recurrence are unchanged.
    """
    checkpoint_dtype = getattr(torch, settings['checkpoint_dtype'])
    assert checkpoint_dtype == torch.float32
    h, v_new, final_state, selected = fused_wrapper()(
        k=k, w=w, u=u, g=g, gk=gk, initial_state=initial_state,
        output_final_state=output_final_state, chunk_size=chunk_size,
        save_new_value=save_new_value, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices,
        checkpoint_rows=checkpoint_rows, checkpoint_count=checkpoint_count,
        checkpoint_dtype=checkpoint_dtype, unselected_row=settings['unselected_row'])
    assert checkpoint_rows.numel() == h.shape[0] * h.shape[1] and checkpoint_rows.device == h.device
    return h, v_new, final_state, selected
