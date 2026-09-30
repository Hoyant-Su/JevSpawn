from collections import defaultdict
from copy import copy

from flash_attn import flash_attn_varlen_func
import torch

from jev_spawn.infra.merged_layers import LayerMergedPrefill
from jev_spawn.runtime.native_cache_batch import _metadata


class SharedPrefixLayout:
    def __init__(self, nodes, device):
        grouped = defaultdict(list)
        offset = 0
        self.spans = []
        for node in nodes:
            stop = offset + len(node.tokens)
            grouped[node.parent].extend(range(offset, stop))
            self.spans.append((offset, stop))
            offset = stop
        self.parents = list(grouped)
        self.order = torch.tensor([position for indices in grouped.values() for position in indices],
            device=device, dtype=torch.long)
        self.inverse = self.order.argsort()
        qlengths = [len(indices) for indices in grouped.values()]
        klengths = [parent.get_seq_length() for parent in self.parents]
        self.qwidth, self.kwidth = max(qlengths), max(klengths)
        self.qoffsets = torch.tensor([0, *qlengths], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        self.koffsets = torch.tensor([0, *klengths], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)


class SharedPrefixMergedPrefill(LayerMergedPrefill):
    def attention_layout(self, suffix, prefixes, nodes):
        return SharedPrefixLayout(nodes, suffix.lengths.device)

    def attend_level(self, nodes, prefixes, suffix, attention, q, k, v, index, scale):
        sources = [parent.layers[index] for parent in attention.parents]
        keys, values = [torch.cat([getattr(source, name).squeeze(0).transpose(0, 1)
            for source in sources]) for name in ('keys', 'values')]
        prefix, prefix_lse, _ = flash_attn_varlen_func(q.squeeze(0).index_select(0, attention.order),
            keys, values, attention.qoffsets, attention.koffsets, attention.qwidth, attention.kwidth,
            softmax_scale=scale, causal=False, return_attn_probs=True)
        prefix = prefix.index_select(0, attention.inverse)
        prefix_lse = prefix_lse.index_select(1, attention.inverse).transpose(0, 1)
        child, child_lse, _ = flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), v.squeeze(0),
            suffix.cu_seqlens, suffix.cu_seqlens, suffix.width, suffix.width,
            softmax_scale=scale, causal=True, return_attn_probs=True)
        difference = prefix_lse - child_lse.transpose(0, 1)
        output = (prefix.float() * difference.sigmoid().unsqueeze(-1) +
                  child.float() * (-difference).sigmoid().unsqueeze(-1)).to(q.dtype)
        for node, (start, stop) in zip(nodes, attention.spans, strict=True):
            if node not in self.retained:
                continue
            source = node.parent.layers[index]
            layer = copy(source)
            layer.__dict__ = _metadata(source, ('keys', 'values'))
            layer.keys = torch.cat((source.keys, k[:, start:stop].transpose(1, 2)), dim=2)
            layer.values = torch.cat((source.values, v[:, start:stop].transpose(1, 2)), dim=2)
            node.layers.append(layer)
        return output
