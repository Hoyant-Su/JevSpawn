import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.resources import TEMPLATES
from baselines.common.task_prefix_service import TaskPrefixService
from baselines.common.tasks import read, rows
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped


def compare(reference, candidate, fields):
    return {'choices_equal': [left['choice'] == right['choice']
                             for left, right in zip(reference, candidate, strict=True)],
            'maximum_absolute_difference': {
                field: max((torch.tensor(left[field]) - torch.tensor(right[field])).abs().max().item()
                           for left, right in zip(reference, candidate, strict=True))
                for field in fields}}


def replay(settings, output):
    shared = SharedConfig.load(settings['shared_config'])
    source = Path(settings['source_run'])
    recorded = read(source / settings['source_task'])
    task, = [task for task in rows(settings['tasks']) if task['task_id'] == recorded['task_id']]
    environment = TaskEnvironment(task, read(settings['environment']), output.parent, deadline=lambda: None)
    context = environment.reset()
    calls = {call['node']: call for call in recorded['calls']}
    batches = read(source / 'session-0000/batches.json')
    method = read(settings['method'])
    CONTROLLER['option_template'] = load_prompt(method['prompts'])['option_template']
    backend = Backend(shared.backend())
    service = SimpleNamespace(backend=backend, field_mode=method['settings']['field_mode'],
                              prefix_cache=PrefixCache(shared.runtime.root_batch_size))
    results = []
    for index in settings['batch_indices']:
        batch = batches[index]
        assert set(batch['task_ids']) == {task['task_id']}
        assert batch['batch_size'] == shared.runtime.batch_size
        selected = [calls[node] for node in batch['node_ids']]
        fields = [{'id': call['node'], 'question': call['question'], 'options': call['options'],
                   'context': context,
                   'state': TEMPLATES['worker_state'].format(
                       context=context, state=json.dumps(call['input'], ensure_ascii=False))}
                  for call in selected]
        prompts = [controller_prompts([field['state']], field['question'], field['options'],
                   list(backend.answer_labels[:len(field['options'])]), CONTROLLER['output_instruction'])[0]
                   for field in fields]
        rendered = backend._render(prompts, CONTROLLER['system'])
        sequences = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
        requests = [SimpleNamespace(task_id=task['task_id'], field=field, input_ids=tokens)
                    for field, tokens in zip(fields, sequences, strict=True)]
        independent, = score_grouped(backend, [fields], settings['modes'][0])['groups']
        service.prefix_cache.clear()
        fresh = TaskPrefixService._score(service, requests)
        reused = TaskPrefixService._score(service, requests)
        fresh_rows, = fresh['groups']
        reused_rows, = reused['groups']
        assert [row['input_tokens'] for row in independent] == batch['input_tokens']
        result = {'batch_index': index, 'node_ids': batch['node_ids'],
                  'paths': [call['input']['path'] for call in selected],
                  'independent': independent, 'fresh_prefix': fresh_rows,
                  'reused_prefix': reused_rows, 'recorded': [call['result'] for call in selected],
                  'independent_vs_fresh': compare(independent, fresh_rows, settings['comparison_fields']),
                  'fresh_vs_reused': compare(fresh_rows, reused_rows, settings['comparison_fields']),
                  'recorded_vs_fresh': compare([call['result'] for call in selected], fresh_rows,
                                               settings['comparison_fields']),
                  'fresh_prefix_hit': fresh['persistent_prefix_hit'],
                  'reused_prefix_hit': reused['persistent_prefix_hit']}
        results.append(result)
        output.write_text(json.dumps({'settings': settings, 'backend': backend.metadata,
                                      'batches': results}, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ['independent', 'fresh_prefix', 'reused_prefix', 'recorded']}), flush=True)
    service.prefix_cache.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    replay(read(arguments.specification), arguments.output)


if __name__ == '__main__':
    main()
