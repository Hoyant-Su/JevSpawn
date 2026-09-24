import argparse
import json
from pathlib import Path
from unittest.mock import patch

from jev_spawn.infra import cached_suffix
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_direct_finite import qualify as paired
from tests.infra.native_ragged_transport.candidate import FusedRaggedSuffix, fused_gdn_forward


class TransportTail(StableFiniteGraphTail):
    def score(self, *args, **kwargs):
        with patch.object(cached_suffix, 'RaggedSuffix', FusedRaggedSuffix), \
             patch.object(gdn, 'ragged_gdn_forward', fused_gdn_forward()):
            return super().score(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    FusedRaggedSuffix.kernel_settings = json.loads(Path(settings['service']).read_text())['settings']['state_copy']
    paired.DirectFiniteTail = TransportTail
    paired.run(settings)
