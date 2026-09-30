from collections import Counter
import time

import torch
import torch.nn.functional as F

from jev_spawn.infra.history_prefix import HistoryPrefixTail
from jev_spawn.infra.merged_layers import LayerMergedPrefill, Segment


class LayerMergedTail(HistoryPrefixTail):
    def __init__(self, *args, packing, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = self.backend.model.get_submodule(packing['language_model_path'])

    def defer(self, backend, states, tails, work):
        result = []
        for parent, tokens in zip(states, tails, strict=True):
            if tokens:
                depth = parent.depth if isinstance(parent, Segment) else 0
                node = Segment(parent, tokens, parent.get_seq_length() + len(tokens), depth + 1, [])
                self.nodes.append(node)
                result.append(node)
            else:
                result.append(parent)
        lengths = [len(tokens) for tokens in tails if tokens]
        work['computed_input_tokens'] += sum(lengths)
        if lengths:
            work['padded_input_tokens'] += max(lengths) * len(lengths)
        return result

    def extend_prefixes(self, backend, states, prefixes, bases, work, extend_states):
        self.nodes = []
        return super().extend_prefixes(backend, states, prefixes, bases, work, self.defer)

    def complete_prefixes(self, backend, histories, prefixes, boundaries, work, extend_states):
        self.completed_prefixes = [prefix if self.owner_counts[owner] > 1 else boundary
            for owner, prefix, boundary in zip(self.owners, prefixes, boundaries, strict=True)]
        return extend_states(backend, histories,
            [prefix[len(boundary):] for prefix, boundary in
             zip(self.completed_prefixes, boundaries, strict=True)], work)

    def __call__(self, backend, sequences, prefixes, owners, states, suffixes, native_ids):
        started = time.perf_counter()
        representatives = {tuple(sequence): row for row, sequence in enumerate(sequences)}
        indices = {tokens: row for row, tokens in enumerate(representatives)}
        rows = list(representatives.values())
        tails = [sequences[row][len(self.completed_prefixes[owners[row]]):] for row in rows]
        work = {'computed_input_tokens': 0, 'padded_input_tokens': 0,
                'graph_replays': 0, 'graph_captures': 0}
        leaves = self.defer(backend, [states[owners[row]] for row in rows], tails, work)
        executor = LayerMergedPrefill(self.language, self.nodes, backend.device)
        hidden = executor(leaves)
        logits = self.project(hidden, native_ids, rows)
        restore = torch.tensor([indices[tuple(sequence)] for sequence in sequences], device=backend.device)
        self.last_logits = logits.index_select(0, restore)
        for key, entries in self.history_cache.items():
            self.history_cache[key] = {
                system: (tokens, state.materialize() if isinstance(state, Segment) else state)
                for system, (tokens, state) in entries.items()}
        self.last_layout = {'execution': 'layer_merged_prefill', 'active_rows': len(sequences),
            'physical_rows': len(rows), 'selected_head_rows': len(native_ids),
            'identical_request_rows_reused': len(sequences) - len(rows),
            'merged_segment_levels': len(executor.levels), 'merged_valid_tokens': executor.ids.numel(),
            'merged_model_passes': 1,
            'tp_output_reductions': len(self.language.layers) * 2}
        self.packed_valid_rows = executor.ids.numel()
        self.packed_dense_rows = sum(suffix.batch_size * suffix.width for _, _, suffix, _ in executor.levels)
        self.nodes = []
        return self.last_logits, work, {'tiles_seconds': time.perf_counter() - started,
            'readout_seconds': 0.0, 'capture_seconds': 0.0}

    def project(self, hidden, native_ids, rows):
        return F.linear(hidden.float(), self.backend.finite_output_weights[:len(native_ids)])

    def score(self, requests, *args, **kwargs):
        self.owner_counts = Counter((request.task_id, request.field['context']) for request in requests)
        result = super().score(requests, *args, **kwargs)
        result.update(packed_valid_rows=self.packed_valid_rows, packed_dense_rows=self.packed_dense_rows)
        return result
