import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def gates_kernel(A, B, Decay, Bias, G, Beta, H: tl.constexpr, N: tl.constexpr,
                 A_ROW: tl.constexpr, A_COL: tl.constexpr,
                 B_ROW: tl.constexpr, B_COL: tl.constexpr,
                 THRESHOLD: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    head = index % H
    a = tl.load(A + index // H * A_ROW + head * A_COL, valid, 0).to(tl.float32)
    b = tl.load(B + index // H * B_ROW + head * B_COL, valid, 0).to(tl.float32)
    a = a + tl.load(Bias + head, valid, 0).to(tl.float32)
    decay = tl.load(Decay + head, valid, 0).to(tl.float32)
    softplus = tl.where(a <= THRESHOLD, libdevice.log1p(tl.exp(a)), a)
    tl.store(G + index, decay * softplus, valid)
    tl.store(Beta + index, tl.sigmoid(b), valid)


def gdn_gates(module, a, b):
    settings = module._execution_settings['gates']
    width = a.shape[-1]
    a_rows, b_rows = a.reshape(-1, width), b.reshape(-1, width)
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(b.shape, dtype=b.dtype, device=b.device)
    gates_kernel[(triton.cdiv(a.numel(), settings['block_size']),)](
        a_rows, b_rows, module._gdn_decay, module.dt_bias, g, beta, width, a.numel(),
        *a_rows.stride(), *b_rows.stride(), settings['softplus_threshold'], settings['block_size'],
        **settings['launch'])
    return g, beta
