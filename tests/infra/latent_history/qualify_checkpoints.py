import argparse
import json
from pathlib import Path

from tests.infra.latent_history import qualify
from tests.infra.latent_history.checkpoints import CheckpointRoleZeroHistory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    qualify.RoleZeroHistory = CheckpointRoleZeroHistory
    qualify.run(json.loads(Path(parser.parse_args().config).read_text()))
