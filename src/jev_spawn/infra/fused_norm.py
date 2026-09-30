import triton
import triton.language as tl


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
