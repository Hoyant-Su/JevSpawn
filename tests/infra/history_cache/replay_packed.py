import argparse
import json
from pathlib import Path

from tests.infra.history_cache.packed_mlp import PackedCheckpoint, PackedReference
from tests.infra.history_cache.replay_finite import reset_history
from tests.infra.native_history_cache import replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    replay.HistoryTail = PackedCheckpoint
    replay.StableFiniteGraphTail = PackedReference
    replay.reset_history = reset_history
    replay.run(json.loads(parser.parse_args().config.read_text()))
