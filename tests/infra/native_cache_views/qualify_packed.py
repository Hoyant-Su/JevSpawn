import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

from jev_spawn.infra import cached_suffix
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_cache_views.packed import pack_batched
from tests.infra.native_direct_finite import qualify as paired


def run(settings):
    operation = partial(pack_batched,
        settings=json.loads(Path(settings['state_copy_settings']).read_text()))

    class PackedTail(StableFiniteGraphTail):
        def score(self, *args, **kwargs):
            with patch.object(cached_suffix, 'pack_native_caches', operation):
                return super().score(*args, **kwargs)

    paired.DirectFiniteTail = PackedTail
    paired.run(settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
