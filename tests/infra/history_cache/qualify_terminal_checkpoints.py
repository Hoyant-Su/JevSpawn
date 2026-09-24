import argparse
import json
from pathlib import Path
import time

import torch

from jev_spawn.infra.history_prefill import HistoryDecode
from jev_spawn.runtime.native_restore import load_native_prefixes
from tests.infra.history_cache.qualify import snapshot
from tests.infra.history_cache.qualify_warm_checkpoints import ForwardTrace, instrumented_run


class TerminalHistoryDecode(HistoryDecode):
    @torch.inference_mode()
    def prefill(self, inputs):
        self.sequences = self.history.sequences
        policy = self.history.settings
        prefixes = [self.history.cache.prefix(sequence) for sequence in self.sequences]
        if not any(prefixes):
            return super().prefill(inputs)
        assert all(0 < len(prefix) < len(sequence) for prefix, sequence in zip(prefixes, self.sequences, strict=True))
        assert len(set(map(tuple, self.sequences))) == len(self.sequences)
        started = time.perf_counter()
        terminal = []

        def retain_terminal(module, args, kwargs, output):
            descriptor = kwargs['ragged_suffix']
            rows = torch.arange(descriptor.batch_size, device=self.backend.device)
            terminal.append(output.last_hidden_state[rows, descriptor.lengths - policy['position_step']])

        handle = self.backend.model.model.register_forward_hook(retain_terminal, with_kwargs=True)
        states = self.history.compute(self.sequences)
        handle.remove()
        hidden, = terminal
        work = self.history.records.pop()
        self.logits.copy_(self.backend.model.lm_head(hidden))
        chosen = self.logits.argmax(-1)
        for row, (sequence, state) in enumerate(zip(self.sequences, states, strict=True)):
            assert state.get_seq_length() == len(sequence)
            self.history.cache.store(sequence, state, self.logits[row:row + policy['row_step']].clone())
        load_native_prefixes(self, states, chosen.tolist(), policy['state_copy'])
        self.ids.copy_(chosen[:, None])
        torch.cuda.synchronize(self.backend.device)
        self.history.records.append({**work, 'logical_input_tokens': sum(map(len, self.sequences)),
            'requests': len(self.sequences), 'exact_rows': policy['zero'],
            'cache_bytes': self.history.cache.bytes, 'elapsed_seconds': time.perf_counter() - started,
            'execution': 'native_cached_prefill_terminal_hidden'})
        return self.ids[:, policy['first_index']].clone()


class ObservedTerminalDecode(TerminalHistoryDecode):
    def prefill(self, inputs):
        torch.cuda.synchronize(self.backend.device)
        start = time.perf_counter()
        value = super().prefill(inputs)
        torch.cuda.synchronize(self.backend.device)
        self.prefill_seconds = time.perf_counter() - start
        self.first_logits = self.logits.clone()
        self.prefill_states = snapshot(self, inputs['attention_mask'].sum(-1).tolist())
        return value


class TerminalDecode(ForwardTrace, ObservedTerminalDecode):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    instrumented_run(json.loads(parser.parse_args().config.read_text()), TerminalDecode)
