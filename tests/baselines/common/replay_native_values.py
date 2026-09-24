import argparse
import json
from pathlib import Path

from baselines.common.config import SharedConfig
from baselines.common.tasks import read
from jev_spawn.infra.backend import Backend


def replay(specification, output):
    settings = read(specification)
    shared = SharedConfig.load(settings['shared_config'])
    source = read(settings['source_batches'])[settings['source_batch_index']]
    assert source['batch_size'] == shared.runtime.batch_size
    assert source['max_new_tokens'] == shared.generation.max_new_tokens
    assert not any(source['truncated']) or any(source['expired'])
    systems = {system['content'] for system, user in source['messages']}
    system, = systems
    prompts = [user['content'] for system_message, user in source['messages']]
    backend = Backend(shared.backend())
    actual = backend.generate(prompts, system, shared.generation.max_new_tokens)
    assert actual['input_tokens'] == source['input_tokens']
    result = {'specification': settings, 'backend': backend.metadata, 'task_ids': source['task_ids'],
              'actual': actual, 'recorded_texts': source['texts'],
              'same_text_by_row': [left == right for left, right in zip(actual['texts'], source['texts'], strict=True)],
              'scope': 'Replay complete recorded scalar requests with native model.generate; no task-quality claim.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'same_text_by_row': result['same_text_by_row'], 'texts': actual['texts']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    replay(arguments.specification, arguments.output)


if __name__ == '__main__':
    main()
