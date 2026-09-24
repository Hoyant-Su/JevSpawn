import argparse
from functools import partial
import json
from pathlib import Path
import re
import time

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.runtime import InferenceRuntime
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prompts


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(specification, output, backend):
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(specification['source_protocol'])
    method = read(specification['method'])
    prompts = load_prompt(specification['prompts'])
    runtime = InferenceRuntime(specification['shared_config'], partial(StableGraphFiniteService,
        settings=method['settings'], prompts=protocol['service_prompts']), backend=backend)
    tasks = []
    for path in specification['source_traces']:
        trace = read(path)
        for node in trace['nodes']:
            tasks.append({'task_id': specification['task_id'].format(source=trace['task_id'], turn=node['turn']),
                'source_task': trace['task_id'], 'turn': node['turn'], 'field': node['consumer'],
                'original_choice': node['selection']['choice']})
    assert len(tasks) == specification['task_count']
    runtime.service.agent_tokenizers = {task['task_id']: backend.tokenizer for task in tasks}
    save(output / 'protocol.json', {'specification': specification, 'source_protocol': protocol,
        'controller': CONTROLLER, 'control_prompts': prompts, 'tasks': tasks,
        'runtime': runtime.metadata(), 'labels_loaded': False})

    def solve(task):
        field = task['field']
        labels = list(backend.answer_labels[:len(field['options'])])
        mapping = dict(zip(labels, [option['id'] for option in field['options']], strict=True))
        state = CONTROLLER['state_template'].format(context=field['context'], state=field['state'])
        finite_user, = controller_prompts([state], field['question'], field['options'], labels,
                                          CONTROLLER['output_instruction'])
        reasoning_user, = controller_prompts([state], field['question'], field['options'], labels,
                                             prompts['output_instruction'])
        finite_messages = [{'role': 'system', 'content': CONTROLLER['system']},
                           {'role': 'user', 'content': finite_user}]
        reasoning_messages = [{'role': 'system', 'content': prompts['system']},
                              {'role': 'user', 'content': reasoning_user}]
        started = time.perf_counter()
        finite, = runtime.service.decide([field], task_id=task['task_id'])
        finite_seconds = time.perf_counter() - started
        started = time.perf_counter()
        reasoning, = runtime.complete(task['task_id'])(reasoning_messages,
            runtime.config.generation.max_new_tokens, runtime.config.generation.temperature)
        reasoning_seconds = time.perf_counter() - started
        answers = re.findall(specification['answer_pattern'], reasoning)
        valid = len(answers) == specification['answer_count'] and answers[0].strip() in mapping
        label = answers[0].strip() if valid else None
        return {**task, 'answer': {'finite_choice': finite['choice'],
            'reasoning_choice': mapping[label] if valid else None, 'reasoning_valid': valid},
            'finite': finite, 'reasoning_text': reasoning, 'reasoning_labels': answers,
            'label_to_option': mapping, 'finite_messages': finite_messages,
            'reasoning_messages': reasoning_messages, 'finite_seconds': finite_seconds,
            'reasoning_seconds': reasoning_seconds}

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
