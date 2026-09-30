from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import torch


def grouped_cache_attention(query, key, value, lengths, leftpad, scaling, splits):
    return flash_attn_with_kvcache(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        cache_seqlens=lengths, cache_leftpad=leftpad,
        softmax_scale=scaling, num_splits=splits,
    )


class GroupedDecodeAttention:
    def __init__(self, cache, leftpad, splits):
        self.cache, self.leftpad, self.splits = cache, leftpad, splits

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        lengths = self.cache.layers[module.layer_idx].cumulative_length.expand(query.shape[0]).to(torch.int32)
        return grouped_cache_attention(query, key, value, lengths, self.leftpad,
                                       kwargs['scaling'], self.splits), None


class RaggedCacheAttention:
    def __init__(self, suffix, prefix_lengths):
        self.suffix = suffix
        prefix_width = max(prefix_lengths)
        width = prefix_width + suffix.width
        lengths = [prefix + length for prefix, length in
                   zip(prefix_lengths, suffix.cu_seqlens_cpu.diff().tolist(), strict=True)]
        self.key_width = max(lengths)
        self.key_offsets = torch.tensor([0, *lengths], dtype=torch.int32,
                                       device=suffix.lengths.device).cumsum(0, dtype=torch.int32)
        self.key_indices = torch.tensor([
            row * width + offset
            for row, (prefix, length) in enumerate(zip(prefix_lengths, lengths, strict=True))
            for offset in range(prefix_width - prefix, prefix_width - prefix + length)
        ], device=suffix.lengths.device, dtype=torch.long)

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        packed_query = self.suffix.pack(query.transpose(1, 2)).squeeze(0)
        packed_key = key.transpose(1, 2).flatten(0, 1).index_select(0, self.key_indices)
        packed_value = value.transpose(1, 2).flatten(0, 1).index_select(0, self.key_indices)
        output = flash_attn_varlen_func(
            packed_query, packed_key, packed_value,
            self.suffix.cu_seqlens, self.key_offsets, self.suffix.width, self.key_width,
            softmax_scale=kwargs['scaling'], causal=True,
        )
        return self.suffix.unpack(output.unsqueeze(0)), None
