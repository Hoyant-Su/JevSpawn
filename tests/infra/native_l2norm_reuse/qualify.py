import argparse
import json
from pathlib import Path

from tests.infra.native_direct_finite import qualify
from tests.infra.native_l2norm_reuse.candidate import CandidateTail, ReferenceTail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    qualify.StableFiniteGraphTail = ReferenceTail
    qualify.DirectFiniteTail = CandidateTail
    qualify.run(settings)
