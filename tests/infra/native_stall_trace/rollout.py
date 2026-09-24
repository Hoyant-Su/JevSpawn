import argparse
from collections import Counter
from functools import wraps
import json
import os
from pathlib import Path
import time

from triton.runtime.cache import FileCacheManager
from triton.runtime.jit import JITFunction

from baselines.common.parallel_run import execute
from jev_spawn.infra.qwen35 import ragged_gdn
from jev_spawn.infra.qwen35.gates import gdn_gates as native_gates
from tests.infra.native_dynamic_gates.candidate import gdn_gates as runtime_length_gates


def instrument(settings, stream):
    calls, seconds = Counter(), Counter()

    def measured(owner, name):
        original = getattr(owner, name)
        label = owner.__name__ + '.' + name

        @wraps(original)
        def invoke(instance, *args, **kwargs):
            started = time.perf_counter()
            result = original(instance, *args, **kwargs)
            finished = time.perf_counter()
            elapsed = finished - started
            calls[label] += settings['count_increment']
            seconds[label] += elapsed
            if elapsed >= settings['slow_seconds']:
                identity = instance.fn.__module__ + '.' + instance.fn.__name__ if owner is JITFunction else instance.key
                stream.write(json.dumps({'operation': label, 'identity': identity,
                    'started_monotonic': started, 'finished_monotonic': finished,
                    'elapsed_seconds': elapsed}) + '\n')
                stream.flush()
            return result

        setattr(owner, name, invoke)

    for name in settings['cache_methods']:
        measured(FileCacheManager, name)
    for name in settings['jit_methods']:
        measured(JITFunction, name)
    return calls, seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    ragged_gdn.gdn_gates = {'native': native_gates,
                          'runtime_length': runtime_length_gates}[settings['finite_gate_kernel']]
    destination = Path(settings['profile_output'])
    destination.mkdir(parents=True, exist_ok=True)
    rank = os.environ['RANK']
    with (destination / f'rank-{rank}-events.jsonl').open('a') as stream:
        calls, seconds = instrument(settings['profiling'], stream)
        specification = json.loads(Path(settings['specification']).read_text())
        assert specification['shared_config'] == settings['shared_config']
        execute(specification, Path(settings['run_output']),
                json.loads(Path(settings['parallel_settings']).read_text()))
        (destination / f'rank-{rank}-summary.json').write_text(json.dumps({
            'calls': calls, 'inclusive_host_seconds': seconds,
            'scope': 'Diagnostic host call intervals; nested intervals overlap. '
                     'No GPU synchronization or model arithmetic changes. '
                     'These timings are not an uninstrumented speed benchmark.'}, indent=2) + '\n')
