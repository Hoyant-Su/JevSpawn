import argparse
import faulthandler
import json
from pathlib import Path

from tests.infra.native_direct_finite.candidate import DirectFiniteService
from tests.infra.native_direct_finite.rollout import run


def instrument(settings):
    score = DirectFiniteService._finite_score

    def profiled_score(service, batch, lengths):
        faulthandler.dump_traceback_later(settings['stall_trace_seconds'])
        result = score(service, batch, lengths)
        faulthandler.cancel_dump_traceback_later()
        return result

    DirectFiniteService._finite_score = profiled_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    instrument(settings)
    run(settings)
