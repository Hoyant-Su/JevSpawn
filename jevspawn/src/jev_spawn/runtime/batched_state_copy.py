import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(Descriptors, ROWS: tl.constexpr, RANK: tl.constexpr,
                 COLUMNS: tl.constexpr, ELEMENT: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    lower, upper = 0, ROWS
    while lower < upper:
        middle = (lower + upper) // 2
        end = tl.load(Descriptors + middle * COLUMNS + 1)
        if tile >= end:
            lower = middle + 1
        else:
            upper = middle
    descriptor = Descriptors + lower * COLUMNS
    start = tl.load(descriptor).to(tl.int32)
    source = tl.load(descriptor + 2).to(tl.pointer_type(ELEMENT))
    destination = tl.load(descriptor + 3).to(tl.pointer_type(ELEMENT))
    count = tl.load(descriptor + 4).to(tl.int32)
    linear = (tile - start) * BLOCK + tl.arange(0, BLOCK)
    remaining = linear
    source_offset = tl.full((BLOCK,), 0, tl.int64)
    destination_offset = tl.full((BLOCK,), 0, tl.int64)
    for dimension in tl.static_range(RANK - 1, -1, -1):
        size = tl.load(descriptor + 5 + dimension).to(tl.int32)
        coordinate = remaining % size
        remaining = remaining // size
        source_offset += coordinate * tl.load(descriptor + 5 + RANK + dimension)
        destination_offset += coordinate * tl.load(descriptor + 5 + 2 * RANK + dimension)
    value = tl.load(source + source_offset, linear < count, other=0)
    tl.store(destination + destination_offset, value, linear < count)


def copy_states(pairs, settings):
    assert pairs and settings['mode'] in ('scalar', 'triton_grouped')
    for destination, source in pairs:
        assert destination.shape == source.shape and source.numel() > 0
        assert destination.dtype == source.dtype and destination.device == source.device
    if settings['mode'] == 'scalar':
        for destination, source in pairs:
            destination.copy_(source)
        return
    element_types = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    device = pairs[0][0].device
    assert device.type == 'cuda'
    assert all(source.device == device and source.dtype in element_types for _, source in pairs)
    for dtype, element in element_types.items():
        group = [(destination, source) for destination, source in pairs if source.dtype == dtype]
        if not group:
            continue
        rank = max(source.ndim for _, source in group)
        descriptors = []
        tiles = 0
        for destination, source in group:
            assert source.numel() <= torch.iinfo(torch.int32).max
            count = triton.cdiv(source.numel(), settings['block_size'])
            padding = rank - source.ndim
            descriptors.append([tiles, tiles + count, source.data_ptr(), destination.data_ptr(), source.numel(),
                                *([1] * padding + list(source.shape)),
                                *([0] * padding + list(source.stride())),
                                *([0] * padding + list(destination.stride()))])
            tiles += count
        assert tiles <= torch.iinfo(torch.int32).max
        metadata = torch.tensor(descriptors, device=device, dtype=torch.int64)
        _copy_kernel[(tiles,)](metadata, len(group), rank, metadata.shape[1], element,
                              settings['block_size'], num_warps=settings['num_warps'])
