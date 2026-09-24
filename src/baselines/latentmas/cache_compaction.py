import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin


def compact_attention_history(cache, mask):
    width = mask.shape[1]
    positions = torch.arange(width, device=mask.device)
    permutation = torch.where(mask.bool(), positions, positions - width).argsort(dim=-1)
    for layer in cache.layers:
        if not isinstance(layer, LinearAttentionCacheLayerMixin):
            indices = permutation[:, None, :, None].expand(
                -1, layer.keys.shape[1], -1, layer.keys.shape[-1])
            for tensor in (layer.keys, layer.values):
                # Keys already contain RoPE; relocate history without rotating it again.
                tensor[:, :, :width].copy_(tensor[:, :, :width].gather(2, indices))
    return (positions >= (width - mask.sum(-1))[:, None]).to(mask.dtype)
