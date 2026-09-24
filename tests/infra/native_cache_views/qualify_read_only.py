import argparse
import json
from pathlib import Path
from unittest.mock import patch

from jev_spawn.infra import cached_suffix
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_cache_views.read_only import split_read_only
from tests.infra.native_direct_finite import qualify as paired


class ReadOnlyTail(StableFiniteGraphTail):
    def score(self, *args, **kwargs):
        with patch.object(cached_suffix, 'split_native_cache_at', split_read_only):
            return super().score(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    paired.DirectFiniteTail = ReadOnlyTail
    paired.run(json.loads(parser.parse_args().config.read_text()))
