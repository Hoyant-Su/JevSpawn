import argparse
from functools import partial
import json
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.runtime import InferenceRuntime
from jev_spawn.schema import CONTROLLER


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(specification, output, backend):
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(specification['source_protocol'])
    method = read(specification['method'])
    tasks = []
    for path in specification['source_traces']:
        trace = read(path)
        for node in trace['nodes']:
            tasks.append({'task_id': specification['task_id'].format(source=trace['task_id'], turn=node['turn']),
                'source_task': trace['task_id'], 'turn': node['turn'], 'field': node['consumer'],
                'original_choice': node['selection']['choice']})
    assert len(tasks) == specification['task_count']
    runtime = InferenceRuntime(specification['shared_config'], partial(StableGraphFiniteService,
        settings=method['settings'], prompts=protocol['service_prompts']), backend=backend)
    runtime.service.agent_tokenizers = {task['task_id']: backend.tokenizer for task in tasks}
    save(output / 'protocol.json', {'specification': specification, 'source_protocol': protocol,
        'controller': CONTROLLER, 'tasks': tasks, 'runtime': runtime.metadata(), 'labels_loaded': False})

    def solve(task):
        field = task['field']
        options = field['options']
        fields = [{**field, 'id': specification['field_id'].format(original=field['id'], rotation=rotation),
            'options': options[rotation:] + options[:rotation]} for rotation in range(len(options))]
        decisions = runtime.service.decide(fields, task_id=task['task_id'])
        mappings = [dict(zip(backend.answer_labels[:len(options)],
            [option['id'] for option in item['options']], strict=True)) for item in fields]
        choices = [decision['choice'] for decision in decisions]
        return {**task, 'answer': {'consistent': len(set(choices)) == specification['consistent_choice_count'],
            'choices': choices}, 'submitted_fields': fields, 'label_to_original_id': mappings,
            'decisions': decisions}

    def commit(index, result):
        save(output / f'task-{index:05d}.json', result)
        print(json.dumps({'task_id': result['task_id'], 'status': result['status'],
            'answer': result['answer'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)

    try:
        results, elapsed = runtime.run(tasks, solve, commit)
        save(output / 'completion.json', {'scope': specification['scope'], 'tasks': len(tasks),
            'elapsed_seconds': elapsed, 'results': [{'task_id': result['task_id'],
                'status': result['status'], 'answer': result['answer']} for result in results]})
    finally:
        runtime.close()
        save(output / 'batches.json', runtime.service.records)
        save(output / 'input_failures.json', runtime.service.input_failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-settings', type=Path, required=True)
    args = parser.parse_args()
    execute_with_runner(read(args.specification), args.output, read(args.parallel_settings), run)
