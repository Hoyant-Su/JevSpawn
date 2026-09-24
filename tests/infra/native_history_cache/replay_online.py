import argparse
import json
from pathlib import Path

from tests.infra.native_history_cache import replay
from tests.infra.native_history_cache.online import OnlineHistoryTail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    replay.HistoryTail = OnlineHistoryTail
    replay.run(json.loads(parser.parse_args().config.read_text()))
