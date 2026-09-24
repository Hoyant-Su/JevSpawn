import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.run import read, save
from baselines.common.runtime import InferenceRuntime
from jev_spawn.infra.prompts import load_prompt


def requests_for(task, prompts):
    original = read(task['request_path'])
    original['id'] = 'original'
    focused = deepcopy(original)
    focused['id'] = 'focused'
    state = json.loads(original['state'])
    events = {event['id']: event for event in state['execution_events']}
    feedback = [{'event_id': identity, 'action': events[identity]['action'],
                 'observation': state['observations'][events[identity]['observation_id']]}
                for identity in state['current_feedback']]
    focused['question'] += prompts['focus'].format(feedback=json.dumps(feedback))
    return [original, focused]


def run(specification, output, backend):
    tasks = read(specification['tasks'])
    assert len(tasks) == specification['task_count']
    method, inference = read(specification['method']), read(specification['inference'])
    prompts = load_prompt(specification['prompts'])
    output.mkdir(parents=True, exist_ok=True)
    runtime = InferenceRuntime(specification['shared_config'], partial(StableGraphFiniteService,
        settings=inference['settings'], prompts=load_prompt(inference['prompts'])), backend=backend)
    runtime.service.configure_runtime_contract(method['settings'])
    runtime.service.agent_tokenizers = {task['task_id']: runtime.backend.tokenizer for task in tasks}
    save(output / 'protocol.json', {'specification': specification, 'tasks': tasks,
        'prompts': prompts, 'shared_config': Path(specification['shared_config']).read_text()})
    save(output / 'runtime.json', runtime.metadata())

    def execute(task):
        requests = requests_for(task, prompts)
        messages = [[{'role': 'system', 'content': prompts['system']},
                     {'role': 'user', 'content': json.dumps(request)}] for request in requests]
        started = time.perf_counter()
        decisions = runtime.service.decide(requests, task_id=task['task_id'])
        finite_seconds = time.perf_counter() - started
        started = time.perf_counter()
        generated = runtime.service.complete_batch(messages, runtime.config.generation.max_new_tokens,
            runtime.config.generation.temperature, None, task_id=task['task_id'], return_tokens=True)
        return {'task_id': task['task_id'], 'answer': generated, 'decisions': decisions,
                'requests': requests, 'messages': messages, 'finite_seconds': finite_seconds,
                'generation_seconds': time.perf_counter() - started}

    def commit(index, result):
        save(output / (result['task_id'] + '.json'), result)
        print(result['task_id'], result['status'], flush=True)

    try:
        results, elapsed = runtime.run(tasks, execute, commit)
        save(output / 'completion.json', {'results': results, 'elapsed_seconds': elapsed})
    finally:
        runtime.close()
        save(output / 'batches.json', runtime.service.records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = read(args.specification)
    execute_with_runner(specification, args.output, read(specification['parallel_settings']), run)


if __name__ == '__main__':
    main()
