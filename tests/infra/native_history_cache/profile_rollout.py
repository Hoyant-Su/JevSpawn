import argparse
import json
from pathlib import Path

from tests.infra.native_history_cache.rollout import install_history_cache
from tests.infra.native_rollout_profile.profile_rollout import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    install_history_cache(settings)
    run(settings)
