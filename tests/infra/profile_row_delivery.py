import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
from threading import get_native_id
import time
from unittest.mock import patch

from baselines.common import foldagent
from baselines.common.configured_environment_run import run
from baselines.common.parallel_run import execute_with_runner
from baselines.common.service import BatchService


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def profile_run(config):
    specification = json.loads(Path(config['specification']).read_text())
    parallel = json.loads(Path(config['parallel_settings']).read_text())
    output = Path(config['output'])
    if int(os.environ['RANK']) != parallel['leader_rank']:
        return execute_with_runner(specification, output, parallel, run)
    events, profiles = [], []
    original_generate = BatchService._generate_tokens
    original_deliver = BatchService._deliver
    original_copy = foldagent.deepcopy
    original_core = foldagent.original.load_core

    def observe(label, function, *args, **kwargs):
        start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            return function(*args, **kwargs)
        finally:
            events.append({'operation': label, 'thread': get_native_id(),
                           'start': start, 'end': time.perf_counter(),
                           'thread_cpu_seconds': time.thread_time() - cpu_start})

    def deliver(service, tokens, indices):
        return observe('row_delivery', original_deliver, service, tokens, indices)

    def copy_tokenizer(value):
        return observe('foldagent_tokenizer_deepcopy', original_copy, value)

    def load_core(*args, **kwargs):
        return observe('foldagent_core_loading', original_core, *args, **kwargs)

    def generate(service, inputs, options, stopping):
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        try:
            return original_generate(service, inputs, options, stopping)
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).stats
            profiles.append({'start': start, 'end': time.perf_counter(),
                'task_ids': [request.task_id for request in service.current_batch],
                'functions': [{'file': key[0], 'line': key[1], 'function': key[2],
                               'primitive_calls': row[0], 'calls': row[1],
                               'self_seconds': row[2], 'cumulative_seconds': row[3]}
                              for key, row in stats.items()]})
            save(Path(config['profiles']), profiles)
            save(Path(config['events']), events)

    with patch.object(BatchService, '_generate_tokens', generate), \
            patch.object(BatchService, '_deliver', deliver), \
            patch.object(foldagent, 'deepcopy', copy_tokenizer), \
            patch.object(foldagent.original, 'load_core', load_core):
        try:
            execute_with_runner(specification, output, parallel, run)
        finally:
            save(Path(config['events']), events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    profile_run(json.loads(parser.parse_args().config.read_text()))
