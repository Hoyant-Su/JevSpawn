import argparse
import json
from pathlib import Path

from tests.infra.history_cache import qualify
from tests.infra.history_cache.checkpoints import CheckpointHistory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    qualify.HistoryPrefill = CheckpointHistory
    qualify.run(json.loads(parser.parse_args().config.read_text()))
