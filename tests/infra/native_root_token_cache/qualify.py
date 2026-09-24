import argparse
import json
from pathlib import Path

from baselines.common.jevspawn_service import StructuredService
from baselines.common.parallel_run import execute


def disable_cache():
    initialize = StructuredService.__init__

    def initialize_uncached(self, backend, shared, deadlines, **kwargs):
        initialize(self, backend, shared, deadlines, **kwargs)
        self.root_prefix_tokens = self._root_prefix_tokens

    StructuredService.__init__ = initialize_uncached


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    if not settings['root_cache']:
        disable_cache()
    execute(json.loads(Path(settings['specification']).read_text()),
            Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
