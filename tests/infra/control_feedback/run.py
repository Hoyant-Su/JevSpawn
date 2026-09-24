import argparse
from functools import partial
from pathlib import Path

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.run import read, save
from baselines.common.runtime import InferenceRuntime
from jev_spawn.infra.prompts import load_prompt


def run(specification, output, backend):
    tasks = read(specification['tasks'])
    method = read(specification['method'])
    inference = read(specification['inference'])
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'protocol.json', {'specification': specification, 'tasks': tasks,
        'shared_config': Path(specification['shared_config']).read_text()})
    runtime = InferenceRuntime(specification['shared_config'],
        partial(StableGraphFiniteService, settings=inference['settings'],
                prompts=load_prompt(inference['prompts'])), backend=backend)
    runtime.service.configure_runtime_contract(method['settings'])
    runtime.service.agent_tokenizers = {task['task_id']: runtime.backend.tokenizer for task in tasks}
    save(output / 'runtime.json', runtime.metadata())

    def execute(task):
        decisions = runtime.service.decide(task['requests'], task_id=task['task_id'])
        return {'task_id': task['task_id'],
                'answer': {decision['id']: decision['choice'] for decision in decisions},
                'decisions': decisions}

    def commit(index, result):
        save(output / (result['task_id'] + '.json'), result)
        print(result['task_id'], result['status'], result['answer'], flush=True)

    try:
        results, elapsed = runtime.run(tasks, execute, commit)
        save(output / 'completion.json', {'results': results, 'elapsed_seconds': elapsed,
            'scope': 'Finite readout on recorded states; not task accuracy.'})
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
