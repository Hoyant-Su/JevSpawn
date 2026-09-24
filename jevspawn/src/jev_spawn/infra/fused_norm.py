from types import MethodType

import torch
import triton
import triton.language as tl
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated


@triton.jit
def rms_kernel(X, W, G, Y, stride_x, stride_g, N: tl.constexpr, EPS: tl.constexpr,
               GATED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + column, column < N, 0).to(tl.float32)
    w = tl.load(W + column, column < N, 0).to(tl.float32)
    variance = tl.sum(x * x, 0) / N
    value = x * tl.rsqrt(variance + EPS)
    if GATED:
        # Preserve the two BF16 rounding boundaries in the pretrained module.
        value = value.to(X.dtype.element_ty).to(tl.float32)
        value = (value * w).to(X.dtype.element_ty).to(tl.float32)
        gate = tl.load(G + row * stride_g + column, column < N, 0).to(tl.float32)
        value = value * (gate / (1.0 + tl.exp(-gate)))
    else:
        value = value * (1.0 + w)
    tl.store(Y + row * N + column, value, column < N)


def rms_norm(x, weight, eps, gate=None):
    assert x.is_cuda and x.dtype == weight.dtype == torch.bfloat16
    assert not torch.is_grad_enabled()
    width = x.shape[-1]
    matrix = x.reshape(-1, width)
    assert matrix.stride(1) == 1
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    gates = gate.reshape(-1, width) if gate is not None else matrix
    assert gates.stride(1) == 1
    rms_kernel[(matrix.shape[0],)](
        matrix, weight, gates, result, matrix.stride(0), gates.stride(0),
        width, eps, gate is not None, triton.next_power_of_2(width), enable_fp_fusion=False)
    return result


def normalized(module, x):
    return rms_norm(x, module.weight, module.eps)


def gated(module, x, gate):
    return rms_norm(x, module.weight, module.variance_epsilon, gate)


def install(model):
    originals = []
    for name, module in model.named_modules():
        if type(module) in (Qwen3_5RMSNorm, Qwen3_5RMSNormGated):
            originals.append((name, module, module.forward))
            module.forward = MethodType(gated if type(module) is Qwen3_5RMSNormGated else normalized, module)
    assert originals
    return originals


def restore(originals):
    for _, module, forward in originals:
        module.forward = forward
