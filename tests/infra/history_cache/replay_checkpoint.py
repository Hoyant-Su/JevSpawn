import argparse
import json
from pathlib import Path

from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.history_cache.finite_checkpoint import SharedCheckpointTail
from tests.infra.history_cache.replay_finite import reset_history
from tests.infra.native_history_cache import replay


class ObservedScore:
    def score(self, *args, **kwargs):
        shapes = []

        def observe(module, positional, keywords):
            shapes.append(tuple(keywords['input_ids'].shape))

        handle = self.backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
        result = super().score(*args, **kwargs)
        handle.remove()
        result['prefill_forwards'] = sum(width > 1 for batch, width in shapes)
        result['prefill_token_slots'] = sum(batch * width for batch, width in shapes if width > 1)
        return result


class ObservedReference(ObservedScore, StableFiniteGraphTail):
    pass


class ObservedCheckpoint(ObservedScore, SharedCheckpointTail):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    replay.HistoryTail = ObservedCheckpoint
    replay.StableFiniteGraphTail = ObservedReference
    replay.reset_history = reset_history
    replay.run(json.loads(parser.parse_args().config.read_text()))
