import ast
from functools import cache
import inspect
import math
import textwrap

import triton
import triton.language as tl

from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


@triton.jit
def pack_kernel(Source, Target, Indices, COUNT: tl.constexpr, WIDTH: tl.constexpr,
                INNER: tl.constexpr, DIM: tl.constexpr, SB: tl.constexpr,
                ST: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr,
                REPEAT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, inner = index // INNER, index % INNER
    source_token = tl.load(Indices + token, index < COUNT, other=0)
    offset = (source_token // WIDTH) * SB + (source_token % WIDTH) * ST
    offset += (inner // DIM // REPEAT) * SH + (inner % DIM) * SD
    value = tl.load(Source + offset, index < COUNT, other=0)
    tl.store(Target + index, value, index < COUNT)


@triton.jit
def unpack_kernel(Source, Target, Lengths, Offsets, COUNT: tl.constexpr,
                  WIDTH: tl.constexpr, INNER: tl.constexpr, DIM: tl.constexpr,
                  ST: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, inner = index // INNER, index % INNER
    row, position = token // WIDTH, token % WIDTH
    length = tl.load(Lengths + row, index < COUNT, other=0)
    packed_token = tl.load(Offsets + row, index < COUNT, other=0) + position
    offset = packed_token * ST + (inner // DIM) * SH + (inner % DIM) * SD
    value = tl.load(Source + offset, (index < COUNT) & (position < length), other=0)
    tl.store(Target + index, value, index < COUNT)


class FusedRaggedSuffix(RaggedSuffix):
    def pack(self, tensor, repeat=1):
        trailing = (tensor.shape[2] * repeat, *tensor.shape[3:])
        output = tensor.new_empty((1, self.indices.numel(), *trailing))
        settings = self.kernel_settings
        shape = (*tensor.shape, *([1] * (4 - tensor.ndim)))
        strides = (*tensor.stride(), *([0] * (4 - tensor.ndim)))
        pack_kernel[(triton.cdiv(output.numel(), settings['block_size']),)](
            tensor, output, self.indices, output.numel(), self.width, math.prod(trailing),
            shape[-1], *strides, repeat, settings['block_size'],
            num_warps=settings['num_warps'])
        return output

    def unpack(self, tensor):
        output = tensor.new_empty((self.batch_size, self.width, *tensor.shape[2:]))
        settings = self.kernel_settings
        shape = (*tensor.shape, *([1] * (4 - tensor.ndim)))
        strides = (*tensor.stride(), *([0] * (4 - tensor.ndim)))
        unpack_kernel[(triton.cdiv(output.numel(), settings['block_size']),)](
            tensor, output, self.lengths, self.cu_seqlens, output.numel(), self.width,
            math.prod(tensor.shape[2:]), shape[-1], *strides[1:], settings['block_size'],
            num_warps=settings['num_warps'])
        return output


@cache
def fused_gdn_forward():
    tree = ast.parse(textwrap.dedent(inspect.getsource(ragged_gdn_forward)))
    function, = tree.body
    function.body = [node for node in function.body if not (
        isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == 'repeat')]
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'pack':
            argument, = node.args
            if isinstance(argument, ast.Name) and argument.id in ('query', 'key'):
                node.args.append(ast.Name(id='repeat', ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    namespace = dict(ragged_gdn_forward.__globals__)
    exec(compile(tree, inspect.getsourcefile(ragged_gdn_forward), 'exec'), namespace)
    return namespace[function.name]
