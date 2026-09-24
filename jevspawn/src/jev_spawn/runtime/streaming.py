"""Consume each native layer's prefix cache before advancing to the next layer."""

from copy import deepcopy

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask


@torch.inference_mode()
def streamed_hidden(model, prefix_ids, prefix_mask, suffix_ids, suffix_mask, branches, branch_batch_size):
    text = model.model.language_model
    prefix = text.embed_tokens(prefix_ids)
    suffix = text.embed_tokens(suffix_ids)
    prefix_positions = (prefix_mask.cumsum(-1) - 1).clamp_min(0)
    mask = torch.cat([prefix_mask.index_select(0, branches), suffix_mask], dim=1)
    suffix_positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, -suffix.shape[1]:]
    prefix_rotary = text.rotary_emb(prefix, prefix_positions[None].expand(3, -1, -1))
    suffix_rotary = text.rotary_emb(suffix, suffix_positions[None].expand(3, -1, -1))
    prefix_masks = {
        'full_attention': create_causal_mask(text.config, prefix, prefix_mask, None, prefix_positions),
        'linear_attention': create_recurrent_attention_mask(text.config, prefix, prefix_mask),
    }
    recurrent_mask = create_recurrent_attention_mask(text.config, suffix, mask)
    cache, fork, empty = [DynamicCache(config=text.config) for _ in range(3)]
    for index, layer in enumerate(text.layers):
        kind = text.config.layer_types[index]
        prefix = layer(prefix, position_embeddings=prefix_rotary, attention_mask=prefix_masks[kind],
                       position_ids=prefix_positions, past_key_values=cache, use_cache=True)
        branch_mask = (create_causal_mask(text.config, suffix, mask, cache, suffix_positions, layer_idx=index)
                       if kind == 'full_attention' else recurrent_mask)
        for start in range(0, len(branches), branch_batch_size):
            end = start + branch_batch_size
            # Native cache updates mutate their tensors and containers; each tile owns its copy.
            fork.layers[index] = deepcopy(cache.layers[index])
            fork.layers[index].reorder_cache(branches[start:end])
            suffix[start:end] = layer(
                suffix[start:end], position_embeddings=tuple(value[start:end] for value in suffix_rotary),
                attention_mask=branch_mask[start:end] if branch_mask is not None else None,
                position_ids=suffix_positions[start:end], past_key_values=fork, use_cache=True,
            )
            fork.layers[index] = empty.layers[index]
        # No continuation is decoded, so this layer's state will never be read again.
        cache.layers[index] = empty.layers[index]
    return text.norm(suffix)
