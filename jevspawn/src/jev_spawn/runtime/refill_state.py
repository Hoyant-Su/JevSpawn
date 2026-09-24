import torch

from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RollingDecode


class RefillState:
    """Keep request metadata and native cache rows under the same stable permutation."""

    def __init__(self, arena, max_new_tokens, pad_token_id):
        self.arena, self.max_new_tokens = arena, max_new_tokens
        self.pad_token_id = pad_token_id
        self.requests = []
        self.pending_requests = None
        self.history = torch.full((arena.max_batch_size, max_new_tokens), pad_token_id,
                                  dtype=torch.long, device=arena.device)
        self.counts = torch.zeros(arena.max_batch_size, dtype=torch.long, device=arena.device)
        self.budgets = torch.zeros_like(self.counts)
        self.deadlines = torch.zeros(arena.max_batch_size, dtype=torch.float64, device=arena.device)
        self.key_positions = torch.arange(arena.max_cache_len, device=arena.device)

    def row_tensors(self):
        return [self.history, self.counts, self.budgets, self.deadlines]

    @property
    def nbytes(self):
        return self.arena.nbytes + sum(t.numel() * t.element_size()
                                      for t in [*self.row_tensors(), self.key_positions])

    def reserve(self, requests, deadline_ends):
        assert self.pending_requests is None
        assert len(requests) == len(deadline_ends)
        assert all(0 < request.max_tokens <= self.max_new_tokens for request in requests)
        rows = self.arena.reserve(len(requests))
        selected = slice(rows.start, rows.stop)
        self.history[selected].fill_(self.pad_token_id)
        self.counts[selected].zero_()
        self.budgets[selected].copy_(torch.tensor([request.max_tokens for request in requests],
                                                dtype=self.budgets.dtype, device=self.arena.device))
        self.deadlines[selected].copy_(torch.tensor(deadline_ends, dtype=self.deadlines.dtype,
                                                   device=self.arena.device))
        self.pending_requests = list(requests)
        return rows

    def commit_prefill(self, rows):
        self.arena.commit_prefill(rows)
        self.requests.extend(self.pending_requests)
        self.pending_requests = None

    def prefill(self, backend, requests, deadline_ends, inputs, graph_pool, graph_stream):
        assert inputs['input_ids'].shape[0] == len(requests)
        assert inputs['input_ids'].shape[1] + max(request.max_tokens for request in requests) <= self.arena.max_cache_len
        rows = self.reserve(requests, deadline_ends)
        decoder = CapturedDecode.from_rows(backend, rows, self.key_positions, graph_pool, graph_stream)
        decoder.prefill(inputs)
        self.commit_prefill(rows)
        return decoder, rows

    def decode(self, backend, graph_pool, graph_stream):
        return RollingDecode.from_rows(backend, self.arena.decode_view(), self.key_positions,
                                       graph_pool, graph_stream)

    def emit(self, start, stop):
        assert self.pending_requests is None and 0 <= start < stop <= self.arena.live_count
        counts = self.counts[start:stop]
        self.history[start:stop].scatter_(1, counts[:, None], self.arena.ids[start:stop])
        counts.add_(1)

    def compact(self, survivors):
        assert self.pending_requests is None
        self.arena.compact(survivors)
        for destination, source in enumerate(survivors):
            if destination != source:
                for tensor in self.row_tensors():
                    tensor[destination].copy_(tensor[source])
        self.requests = [self.requests[index] for index in survivors]

    def stop_groups(self):
        stops = [request.stop for request in self.requests]
        return [(pattern, torch.tensor([current == pattern for current in stops],
                                        device=self.arena.device, dtype=torch.bool))
                for pattern in dict.fromkeys(stops) if pattern]
