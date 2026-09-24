import argparse
import json
from pathlib import Path

from baselines.common.lats import load_core
from baselines.common.runtime import InferenceRuntime
from baselines.common.tasks import read
from jev_spawn.infra.prompts import load_prompt
from tests.baselines.lats.value_role_requests import assessment, source_request


def replay(settings, output):
    method = read(settings['method'])
    prompts = load_prompt(method['prompts'])
    prompt, recorded = source_request(settings, prompts)
    _, original_task = load_core(method['settings']['source_directory'], None, None)
    requests = [{'task_id': settings['identity_template'].format(variant=variant),
                 'variant': variant, 'messages': recorded['messages'] if variant == 'original' else prompt.messages}
                for variant in settings['variants']]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps({'settings': settings, 'method': method,
        'prompts': prompts, 'recorded': recorded, 'requests': requests}, indent=2) + '\n')
    runtime = InferenceRuntime(settings['shared_config'])
    assert recorded['requested_max_new_tokens'] == runtime.config.generation.max_new_tokens

    def execute(request):
        result, = runtime.complete(request['task_id'])(request['messages'],
            recorded['requested_max_new_tokens'], runtime.config.generation.temperature,
            n=settings['expected_output_samples'], stop=recorded['row_stops'], return_tokens=True)
        return {'task_id': request['task_id'], 'variant': request['variant'],
                'answer': None, 'generation': result, 'assessment': assessment(result['text'], original_task)}

    def commit(index, result):
        (output / (requests[index]['variant'] + '.json')).write_text(json.dumps(result, indent=2) + '\n')

    try:
        results, elapsed = runtime.run(requests, execute, commit)
    finally:
        runtime.close()
        (output / 'batches.json').write_text(json.dumps(runtime.service.records, indent=2) + '\n')
        (output / 'runtime.json').write_text(json.dumps(runtime.metadata(), indent=2) + '\n')
    candidate, = [row for row in results if row['task_id'] ==
                  settings['identity_template'].format(variant='evaluator_roles')]
    qualified = candidate['status'] == 'completed'
    summary = {'qualified': qualified, 'elapsed_seconds': elapsed,
               'historical_assessment': assessment(recorded['texts'], original_task),
               'results': results, 'scope': settings['stage']}
    (output / 'completion.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'qualified': qualified, 'elapsed_seconds': elapsed}), flush=True)
    assert qualified, 'Evaluator replay did not complete the original value computation.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(read(args.settings), args.output)


if __name__ == '__main__':
    main()
