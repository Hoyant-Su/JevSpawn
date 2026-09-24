import argparse
import cProfile
from functools import partial
import json
from pathlib import Path
import time

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.runtime import InferenceRuntime
from jev_spawn.infra import cached_suffix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime import native_cache_batch


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(specification, output, backend):
    output.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in Path(specification['source']).read_text().splitlines()]
    assert len(tasks) == specification['task_count']
    profiler = cProfile.Profile()
    split = native_cache_batch.split_native_cache_at
    measured_split = partial(profiler.runcall, split)
    native_cache_batch.split_native_cache_at = measured_split
    cached_suffix.split_native_cache_at = measured_split
    method = read(specification['method'])
    runtime = InferenceRuntime(specification['shared_config'], partial(StableGraphFiniteService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)
    runtime.service.agent_tokenizers = {task['task_id']: backend.tokenizer for task in tasks}
    save(output / 'protocol.json', {'specification': specification, 'runtime': runtime.metadata()})

    def execute(task):
        options = task['options']
        requests = [{'id': specification['field_id'].format(rotation=rotation),
                     'context': task['context'], 'state': specification['state'],
                     'question': task['question'],
                     'options': options[rotation:] + options[:rotation]}
                    for rotation in range(len(options))]
        started = time.perf_counter()
        decisions = runtime.service.decide(requests, task_id=task['task_id'])
        return {'task_id': task['task_id'], 'requests': requests, 'decisions': decisions,
                'elapsed_seconds': time.perf_counter() - started}

    def commit(index, result):
        save(output / specification['result_file'].format(index=index), result)
        print(json.dumps(result), flush=True)

    results, elapsed = runtime.run(tasks, execute, commit)
    save(output / 'completion.json', {'results': results, 'elapsed_seconds': elapsed})
    save(output / 'batches.json', runtime.service.records)
    save(output / 'input_failures.json', runtime.service.input_failures)
    runtime.close()
    profiler.dump_stats(str(output / specification['profile_file']))
    native_cache_batch.split_native_cache_at = split
    cached_suffix.split_native_cache_at = split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-settings', type=Path, required=True)
    args = parser.parse_args()
    execute_with_runner(read(args.specification), args.output, read(args.parallel_settings), run)
