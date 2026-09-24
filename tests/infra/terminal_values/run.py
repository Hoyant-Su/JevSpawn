import argparse
from dataclasses import asdict
import json
from pathlib import Path

from baselines.common.parallel_run import execute_with_runner
from baselines.common.run import read, save
from baselines.common.runtime import InferenceRuntime
from baselines.common.service_factory import service_factory
from typed_answer import finalize_answer


def run(specification, output, backend):
    entries = read(specification['tasks'])
    assert len(entries) == specification['task_count']
    method, inference = read(specification['method']), read(specification['inference'])
    settings = read(specification['terminal_settings'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'traces').mkdir(exist_ok=True)
    runtime = InferenceRuntime(specification['shared_config'], service_factory(method, inference), backend=backend)
    runtime.service.configure_runtime_contract(runtime.config.method_settings(method['settings']))
    runtime.service.agent_tokenizers = {entry['task_id']: runtime.backend.tokenizer for entry in entries}
    indexes = {entry['task_id']: index for index, entry in enumerate(entries)}
    save(output / 'protocol.json', {'specification': specification, 'entries': entries,
        'terminal_settings': settings, 'shared_config': asdict(runtime.config),
        'scope': 'Final-value construction from recorded real states; not a new complete rollout.'})
    save(output / 'runtime.json', runtime.metadata())

    def execute(entry):
        fixture = read(entry['fixture_path'])
        assert fixture['task_id'] == entry['task_id']
        trace = {}
        try:
            answer = finalize_answer(fixture['context'], fixture['state'], fixture['answer_schema'],
                runtime.service, fixture['task_id'], asdict(runtime.config.generation), settings, trace)
            return {'task_id': fixture['task_id'], 'answer': answer, 'trace': trace}
        finally:
            save(output / 'traces' / f"task-{indexes[entry['task_id']]:05d}.json", trace)

    def commit(index, result):
        save(output / f'task-{index:05d}.json', result)
        print(json.dumps({'task_id': result['task_id'], 'status': result['status'],
                          'elapsed_seconds': result['elapsed_seconds']}), flush=True)

    try:
        results, elapsed = runtime.run(entries, execute, commit)
        save(output / 'completion.json', {'tasks': len(results), 'elapsed_seconds': elapsed,
            'statuses': [result['status'] for result in results]})
    finally:
        runtime.close()
        save(output / 'batches.json', runtime.service.records)
        save(output / 'input_failures.json', runtime.service.input_failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = read(args.specification)
    execute_with_runner(specification, args.output, read(specification['parallel_settings']), run)


if __name__ == '__main__':
    main()
