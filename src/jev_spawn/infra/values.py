from collections import defaultdict
import json
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from jev_spawn.infra.attention import SharedPrefixMergedPrefill
from jev_spawn.infra.merged_layers import Segment
from jev_spawn.infra.merged_tail import LayerMergedTail


def token_table(candidates):
    children = defaultdict(set)
    for tokens in candidates:
        for index, token in enumerate(tokens):
            children[tuple(tokens[:index])].add(token)
    forks = sorted(prefix for prefix, values in children.items() if len(values) > 1)
    indices = {prefix: index for index, prefix in enumerate(forks)}
    edges = [[(indices[tuple(tokens[:index])], token) for index, token in enumerate(tokens)
              if tuple(tokens[:index]) in indices] for tokens in candidates]
    return forks, [sorted(children[prefix]) for prefix in forks], edges


class ValueTail(LayerMergedTail):
    def __init__(self, *args, values, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = values
        self.terminator = self.backend.tokenizer.convert_tokens_to_ids(values['terminator'])
        assert self.backend.tokenizer.decode([self.terminator]) == values['terminator']

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        self.active_count = len(requests)
        texts = [[json.dumps(option['values'], **self.values['serialization'])
                  for option in request.field['options']] for request in requests]
        self.tokens = [[tokens + [self.terminator] for tokens in
            self.backend.tokenizer(group, add_special_tokens=False)['input_ids']] for group in texts]
        if physical_batch_size is not None and len(requests) < physical_batch_size:
            self.tokens.extend([self.tokens[-1]] * (physical_batch_size - len(requests)))
        self.tables = [token_table(group) for group in self.tokens]
        result = super().score(requests, base_lengths, base_cache, physical_batch_size)
        result['values'] = self.last_layout
        return result

    def __call__(self, backend, sequences, prefixes, owners, states, suffixes, native_ids):
        began = time.perf_counter()
        work = {'computed_input_tokens': 0, 'padded_input_tokens': 0, 'graph_replays': 0, 'graph_captures': 0}
        prompts = self.defer(backend, [states[owners[row]] for row in range(len(sequences))],
            [sequence[len(self.completed_prefixes[owners[row]]):]
             for row, sequence in enumerate(sequences)], work)
        fork_prefixes = [list(prefix) for forks, _, _ in self.tables for prefix in forks]
        parents = [prompt for prompt, (forks, _, _) in zip(prompts, self.tables, strict=True) for _ in forks]
        nodes = self.defer(backend, parents, fork_prefixes, work)
        executor = SharedPrefixMergedPrefill(self.language, self.nodes, backend.device)
        hidden = executor.all_hidden()
        positions = torch.tensor([executor.ends[node] for node in nodes], device=hidden.device, dtype=torch.long)
        predictors = hidden.index_select(0, positions)
        allowed = [tokens for _, children, _ in self.tables for tokens in children]
        vocabulary = sorted({token for children in allowed for token in children})
        lookup = {token: index for index, token in enumerate(vocabulary)}
        ids = torch.tensor(vocabulary, device=hidden.device, dtype=torch.long)
        head = backend.model.lm_head
        width = head.weight.shape[0]
        weights = head.weight.index_select(0, ids % width).to(getattr(torch, self.values['score_dtype']))
        weights *= (ids // width == dist.get_rank(head.group))[:, None]
        logits = F.linear(predictors.to(weights.dtype), weights)
        dist.all_reduce(logits, op=dist.ReduceOp.SUM, group=head.group)
        mask = torch.zeros(logits.shape, device=hidden.device, dtype=torch.bool)
        fork_rows = torch.tensor([row for row, tokens in enumerate(allowed) for _ in tokens], device=hidden.device, dtype=torch.long)
        token_columns = torch.tensor([lookup[token] for tokens in allowed for token in tokens], device=hidden.device, dtype=torch.long)
        mask[fork_rows, token_columns] = True
        logps = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
        row_indices, column_indices, option_owners, multiplicities = [], [], [], []
        offset, option = 0, 0
        for tokens, (forks, _, edges) in zip(self.tokens, self.tables, strict=True):
            copies = defaultdict(int)
            for path in tokens:
                copies[tuple(path)] += 1
            for path, choices in zip(tokens, edges, strict=True):
                row_indices.extend(offset + row for row, _ in choices)
                column_indices.extend(lookup[token] for _, token in choices)
                option_owners.extend([option] * len(choices))
                multiplicities.append(copies[tuple(path)])
                option += 1
            offset += len(forks)
        rows = torch.tensor(row_indices, device=hidden.device, dtype=torch.long)
        columns = torch.tensor(column_indices, device=hidden.device, dtype=torch.long)
        destinations = torch.tensor(option_owners, device=hidden.device, dtype=torch.long)
        scores = logps.new_zeros(option).scatter_add_(0, destinations, logps[rows, columns])
        scores -= torch.tensor(multiplicities, device=hidden.device, dtype=scores.dtype).log()
        assert torch.isfinite(scores).all()
        self.last_logits = scores.new_full((len(sequences), len(native_ids)), -torch.inf)
        rows = torch.repeat_interleave(torch.arange(len(sequences), device=hidden.device, dtype=torch.long),
            torch.tensor(list(map(len, self.tokens)), device=hidden.device, dtype=torch.long))
        columns = torch.tensor([index for group in self.tokens for index in range(len(group))], device=hidden.device, dtype=torch.long)
        self.last_logits[rows, columns] = scores
        for key, entries in self.history_cache.items():
            self.history_cache[key] = {
                system: (tokens, state.materialize() if isinstance(state, Segment) else state)
                for system, (tokens, state) in entries.items()}
        self.last_layout = {'execution': 'native_value_forks', 'active_rows': self.active_count,
            'physical_rows': len(sequences), 'candidate_rows': option, 'fork_rows': len(nodes),
            'native_vocabulary_rows': len(vocabulary), 'fork_prefix_model_tokens': sum(map(len, fork_prefixes)),
            'complete_value_tokens': sum(len(tokens) for group in self.tokens for tokens in group),
            'merged_valid_tokens': executor.ids.numel(), 'generated_output_tokens': 0,
            'attention': 'shared_parent_prefix_lse', 'score': self.values['score']}
        self.packed_valid_rows = executor.ids.numel()
        self.packed_dense_rows = sum(suffix.batch_size * suffix.width for _, _, suffix, _ in executor.levels)
        self.nodes = []
        return self.last_logits, work, {'tiles_seconds': time.perf_counter() - began,
            'readout_seconds': 0.0, 'capture_seconds': 0.0}
