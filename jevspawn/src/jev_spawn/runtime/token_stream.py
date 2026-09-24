import torch
import torch.nn.functional as F

from jev_spawn.runtime.finite_decoding import CapturedFinite
from jev_spawn.runtime.token_transitions import TokenTransitions
from jev_spawn.runtime.native_cache_batch import export_native_rows
from jev_spawn.infra.cached_suffix import ragged_suffix


class CapturedTokenStream(CapturedFinite):
    """Continue a native model cache along explicit finite token paths."""

    def __init__(self, backend, batch_size, capacity, table, graph_pool, graph_stream, settings):
        super().__init__(backend, batch_size, capacity, table['token_ids'], graph_pool,
                         graph_stream, settings['state_copy'])
        self.root = table['root']
        self.start_state = self.root
        count = table['forced_lengths'][self.root]
        count -= int(table['terminal'][table['forced_next_states'][self.root]])
        prefix_indices = table['forced_token_indices'][self.root][:count]
        self.prefix_state = self.root
        for index in prefix_indices:
            edge = table['edge_token_indices'][self.prefix_state].index(index)
            self.prefix_state = table['edge_next_states'][self.prefix_state][edge]
        self.forced_prefix_ids = [table['token_ids'][index] for index in prefix_indices]
        self.forced_prefix = torch.tensor(self.forced_prefix_ids,
                                          device=backend.device, dtype=torch.long)
        self.transitions = TokenTransitions(table, batch_size, backend.device, settings['transition'])
        self.transitions.output = self.ids[:, 0]
        self.completed = {}

    def reset_path(self):
        self.transitions.state.fill_(self.start_state)
        self.transitions.done.zero_()

    @torch.inference_mode()
    def load(self, states, last_tokens):
        super().load(states, last_tokens)
        self.start_state = self.root
        self.reset_path()

    @torch.inference_mode()
    def prefill(self, inputs):
        self.cache.reset()
        self.reset_path()
        width = inputs['input_ids'].shape[1]
        self.key_valid.fill_(True)
        self.key_valid[:, :width].copy_(inputs['attention_mask'])
        positions = (inputs['attention_mask'].cumsum(-1) - 1).clamp_min(0)
        output = self.trunk(**inputs, position_ids=positions, past_key_values=self.cache, use_cache=True)
        self.logits.copy_(F.linear(output.last_hidden_state[:, -1].float(), self.selected_weights))
        self.transitions.advance(self.logits)
        self.positions.copy_(inputs['attention_mask'].sum(-1)[:, None])
        self.validate_attention()
        return self.ids[:, 0].clone()

    @torch.inference_mode()
    def step(self):
        super().step()
        self.transitions.advance(self.logits)
        self.positions.add_(1)

    def capture_snapshot(self):
        state, saved, kv = super().capture_snapshot()
        extra = [self.transitions.state, self.transitions.done]
        return [*state, *extra], [*saved, *[value.clone() for value in extra]], kv

    def decode_path(self, inputs, options, stopping, warmup_steps, on_tokens=None, between_steps=None):
        assert not options['do_sample'], 'The native finite stream currently implements greedy selection.'
        self.completed = {}
        result = super().generate_tokens(inputs, options,
            lambda tokens, scores: self.transitions.done | stopping(tokens, scores),
            warmup_steps, self.completion_callback(on_tokens), between_steps)
        assert bool(self.transitions.done.all()), 'The output budget ended before all finite paths completed.'
        return result

    def completion_callback(self, on_tokens):
        def complete(tokens, rows):
            caches, metadata = export_native_rows(self, rows)
            self.completed.update(zip(rows, zip(caches, metadata, strict=True), strict=True))
            if on_tokens is not None:
                on_tokens(tokens, rows)
        return complete

    @torch.inference_mode()
    def continue_tokens(self, previous, feedback_tokens, options, warmup_steps):
        """Append actual feedback to saved states and return the next action tokens."""
        assert not options['do_sample']
        prefix = self.forced_prefix_ids
        states, metadata = zip(*previous, strict=True)
        tails = [[record['pending_token'], *feedback, *prefix]
                 for record, feedback in zip(metadata, feedback_tokens, strict=True)]
        work = {'computed_input_tokens': 0, 'padded_input_tokens': 0}
        extended = ragged_suffix(self.backend, list(states), [tail[:-1] for tail in tails], work)
        self.load(extended, [tail[-1] for tail in tails])
        self.start_state = self.prefix_state
        self.reset_path()
        self.completed = {}
        self.step() if self.graph is None else self.graph.replay()
        token = self.ids[:, 0].clone()
        prefix_ids = self.forced_prefix.expand(len(tails), -1)
        assert options['max_new_tokens'] > len(prefix)
        result = self.decode_tokens({'input_ids': prefix_ids},
            {**options, 'max_new_tokens': options['max_new_tokens'] - len(prefix)},
            lambda tokens, scores: self.transitions.done, warmup_steps, token,
            self.completion_callback(None))
        assert bool(self.transitions.done.all()), 'The output budget ended before all finite paths completed.'
        self.continuation_work = work
        return result

    def generate_tokens(self, inputs, options, stopping, warmup_steps, on_tokens=None, between_steps=None):
        prefix = self.forced_prefix.expand(inputs['input_ids'].shape[0], -1)
        assert options['max_new_tokens'] > prefix.shape[-1], 'The token budget cannot complete this action path.'
        extended = {'input_ids': torch.cat([inputs['input_ids'], prefix], dim=-1),
                    'attention_mask': torch.cat([inputs['attention_mask'], torch.ones_like(prefix)], dim=-1)}
        self.start_state = self.prefix_state
        deliver = None if on_tokens is None else lambda tokens, rows: on_tokens(
            torch.cat([prefix, tokens], dim=-1), rows)
        return self.decode_path(extended,
            {**options, 'max_new_tokens': options['max_new_tokens'] - prefix.shape[-1]},
            stopping, warmup_steps, deliver, between_steps)
