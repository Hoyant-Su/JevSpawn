import torch
import triton
import triton.language as tl


@triton.jit
def _advance(Logits, Edges, Targets, Valid, Tokens, Terminal, State, Output,
             Done, VOCAB: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    state = tl.load(State + row)
    stopped = tl.load(Terminal + state)
    columns = tl.arange(0, BLOCK)
    valid = tl.load(Valid + state * WIDTH + columns, columns < WIDTH, other=False) & ~stopped
    indices = tl.load(Edges + state * WIDTH + columns, valid, other=0)
    scores = tl.load(Logits + row * VOCAB + indices, valid, other=-float('inf'))
    best = tl.max(scores, axis=0)
    selected = tl.min(tl.where(valid & (scores == best), columns, BLOCK), axis=0)
    index = tl.load(Edges + state * WIDTH + selected, ~stopped, other=0)
    target = tl.load(Targets + state * WIDTH + selected, ~stopped, other=state)
    token = tl.load(Tokens + index, ~stopped, other=0)
    tl.store(Output + row, token, ~stopped)
    tl.store(State + row, target)
    tl.store(Done + row, tl.load(Terminal + target))


class TokenTransitions:
    """Advance explicit native-token paths without host-side token decisions."""

    def __init__(self, table, batch_size, device, settings):
        self.settings = settings
        self.tokens = torch.tensor(table['token_ids'], device=device, dtype=torch.long)
        self.edges = torch.tensor(table['edge_token_indices'], device=device, dtype=torch.long)
        self.targets = torch.tensor(table['edge_next_states'], device=device, dtype=torch.long)
        self.valid = torch.tensor(table['edge_valid'], device=device, dtype=torch.bool)
        self.terminal = torch.tensor(table['terminal'], device=device, dtype=torch.bool)
        self.state = torch.full((batch_size,), table['root'], device=device, dtype=torch.long)
        self.done = self.terminal[self.state].clone()
        self.output = torch.empty_like(self.state)

    def advance(self, logits):
        _advance[(self.state.numel(),)](
            logits, self.edges, self.targets, self.valid, self.tokens, self.terminal,
            self.state, self.output, self.done, self.tokens.numel(), self.edges.shape[-1],
            triton.next_power_of_2(self.edges.shape[-1]), num_warps=self.settings['num_warps'])
        return self.output
