# L2 normalization arithmetic follows FLA's MIT-licensed l2norm_fwd_kernel.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.

from copy import copy
import importlib

import triton
import triton.language as tl

from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail


@triton.jit(do_not_specialize=['T', 'NB'])
def l2norm_runtime_bucket(x, y, rstd, eps, T, D: tl.constexpr,
                         BD: tl.constexpr, NB, BT: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    columns = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (columns[None, :] < D)
    offsets = rows[:, None] * D + columns[None, :]
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse = 1 / tl.sqrt(tl.sum(values * values, 1) + eps)
    normalized = values * inverse[:, None]
    tl.store(y + offsets, normalized.to(y.dtype.element_ty), mask=mask)
    tl.store(rstd + rows, inverse.to(rstd.dtype.element_ty), mask=rows < T)


MODULE = importlib.import_module('fla.modules.l2norm')
REFERENCE = MODULE.l2norm_fwd_kernel
CANDIDATE = copy(REFERENCE)
CANDIDATE.fn = l2norm_runtime_bucket
CANDIDATE.base_fn = l2norm_runtime_bucket.fn


class ReferenceTail(StableFiniteGraphTail):
    def __init__(self, *args, **kwargs):
        MODULE.l2norm_fwd_kernel = REFERENCE
        super().__init__(*args, **kwargs)


class CandidateTail(StableFiniteGraphTail):
    def __init__(self, *args, **kwargs):
        # Keep the original tuning keys and selected launch configurations.
        CANDIDATE.cache = REFERENCE.cache
        MODULE.l2norm_fwd_kernel = CANDIDATE
        super().__init__(*args, **kwargs)
