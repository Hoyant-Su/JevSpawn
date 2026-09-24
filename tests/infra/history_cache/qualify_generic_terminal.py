import argparse
import json
from pathlib import Path
import time
from types import FunctionType

import torch

from tests.infra.history_cache import qualify
from tests.infra.history_cache.qualify_warm_checkpoints import ForwardTrace
from tests.infra.history_cache.terminal_service import TerminalCheckpointHistory, TerminalHistoryDecode


class ObservedTerminal(ForwardTrace, TerminalHistoryDecode):
    def prefill(self, inputs):
        torch.cuda.synchronize(self.backend.device)
        started = time.perf_counter()
        value = super().prefill(inputs)
        torch.cuda.synchronize(self.backend.device)
        self.prefill_seconds = time.perf_counter() - started
        self.first_logits = self.logits.clone()
        self.prefill_states = qualify.snapshot(self, inputs['attention_mask'].sum(-1).tolist())
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    original = qualify.run
    run = FunctionType(original.__code__, dict(original.__globals__, HistoryPrefill=TerminalCheckpointHistory,
        ObservedHistoryDecode=ObservedTerminal), original.__name__, original.__defaults__, original.__closure__)
    run(json.loads(parser.parse_args().config.read_text()))
