import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
from itertools import repeat
import json
from multiprocessing import get_context
from pathlib import Path
import time

import yaml

from baselines.common.evaluate import score_task, summarize
from baselines.common.persistence import save


def run(settings):
    reports = []
    for source in settings['runs']:
        directory = Path(source['run']).resolve()
        contract = json.loads((directory / 'protocol.json').read_text())
        reference = json.loads(Path(source['reference']).read_text())
        reference, = reference['runs']
        tasks = contract['tasks']
        shared = yaml.safe_load(contract['shared_config_text'])
        definition = contract['environment_execution']
        module, name = definition['class'].rsplit('.', maxsplit=1)
        factory = getattr(importlib.import_module(module), name)
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=shared['runtime']['cpu_threads'],
                                 mp_context=get_context(settings['process_start_method'])) as pool:
            scores = list(pool.map(score_task, tasks,
                [directory / f'task-{index:05d}.json' for index in range(len(tasks))],
                repeat(factory), repeat(definition['parameters']), repeat(contract['tools']),
                [directory / settings['scratch_directory'] / f'task-{index:05d}'
                 for index in range(len(tasks))]))
        elapsed = time.perf_counter() - started
        actual = [{key: value for key, value in row.items() if key not in settings['timing_fields']}
                  for row in scores]
        expected = [{key: value for key, value in row.items() if key not in settings['timing_fields']}
                    for row in reference['scores']]
        summary = summarize(scores)
        reports.append({'run': source['run'], 'reference': source['reference'],
            'workers': shared['runtime']['cpu_threads'], 'wall_seconds': elapsed,
            'all_task_records_equal_except_offline_timing': actual == expected,
            'aggregate_metrics_equal': summary == reference['summary'],
            'summary': summary, 'scores': scores})
        save(settings['output'], {'settings': settings, 'reports': reports})
        assert actual == expected and summary == reference['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
