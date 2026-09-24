import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.resources import TEMPLATES
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def tensor_state(value, path, seen):
    if id(value) in seen:
        return {}
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (list, tuple)):
        children = enumerate(value)
    elif hasattr(value, '__dict__'):
        children = vars(value).items()
    else:
        return {}
    result = {}
    for key, child in children:
        result.update(tensor_state(child, f'{path}/{key}', seen))
    return result


def snapshots(caches):
    states = {}
    for name, cache in caches.items():
        for index, value in enumerate(cache.entries.values()):
            states.update(tensor_state(value, f'{name}/{index}', set()))
    return {name: tensor.clone() for name, tensor in states.items()}


def unchanged(caches, reference):
    current = snapshots(caches)
    assert current.keys() == reference.keys()
    return all(torch.equal(reference[key], current[key]) for key in reference)


def render(backend, group, context):
    prompts = [controller_prompts([node['state']], node['question'], node['options'],
        list(backend.answer_labels[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
        for node in group]
    texts = backend._render(prompts, CONTROLLER['system'])
    ids = backend.tokenizer(texts, add_special_tokens=False)['input_ids']
    partial = CONTROLLER['user_template'].split('{state}')[0] + context
    task_text = backend.tokenizer.apply_chat_template(
        [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': partial}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    task_ids = backend.tokenizer(task_text, add_special_tokens=False)['input_ids']
    return {'texts': texts, 'input_ids': ids, 'task_prefix_length': common_prefix([task_ids, *ids]),
            'state_prefix_length': common_prefix(ids)}


def workload(settings):
    source = Path(settings['source_run'])
    protocol = json.loads((source / 'protocol.json').read_text())
    task, = protocol['tasks']
    record = json.loads((source / settings['task_file']).read_text())
    assert record['task_id'] == task['task_id'] == settings['task_id']
    assert TEMPLATES == protocol['resources']['templates']
    environment = TaskEnvironment(task, protocol['tools'], Path(settings['output']) / 'unused_tools',
                                  deadline=lambda: None)
    context = environment.reset()
    calls = {call['node']: call for call in record['calls']
             if call['kind'] == 'finite' and all(key in call['input'] for key in settings['worker_fields'])}
    assert len(calls) == settings['worker_count']
    batches = json.loads((source / settings['batches_file']).read_text())
    groups, original_groups = [], []
    visited = []
    for index in settings['batch_indices']:
        batch = batches[index]
        assert batch['batch_size'] == settings['batch_size']
        assert set(batch['task_ids']) == {task['task_id']}
        group, originals = [], []
        for identity in batch['node_ids']:
            call = calls[identity]
            payload = {key: call['input'][key] for key in settings['shared_fields']}
            payload.update(call['input'])
            assert payload == call['input']
            node = {'id': identity, 'context': context, 'question': call['question'], 'options': call['options']}
            for output, value in ((group, payload), (originals, call['input'])):
                state = protocol['prompts']['encoded_state'].format(
                    value=json.dumps(value, **settings['serialization']))
                output.append({**node, 'state': TEMPLATES['worker_state'].format(context=context, state=state)})
            visited.append(identity)
        groups.append(group)
        original_groups.append(originals)
    assert set(visited) == set(calls) and len(visited) == len(calls)
    return protocol, context, groups, original_groups, calls


def compare(reference, candidate):
    rows = []
    for left, right in zip(reference, candidate, strict=True):
        assert left['id'] == right['id'] and left['option_ids'] == right['option_ids']
        a, b = torch.tensor(left['option_logits']), torch.tensor(right['option_logits'])
        rows.append({'id': left['id'], 'reference_choice': left['choice'], 'candidate_choice': right['choice'],
                     'choice_equal': left['choice'] == right['choice'], 'exact_logits': torch.equal(a, b),
                     'maximum_absolute_logit_difference': float((a - b).abs().max())})
    return rows


@torch.inference_mode()
def run(settings):
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    protocol, context, groups, original_groups, calls = workload(settings)
    CONTROLLER.clear()
    CONTROLLER.update(load_prompt(settings['controller_prompt']))
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    shared = SharedConfig.load(settings['shared_config'])
    backend = Backend(shared.backend())
    assert shared.runtime.seed == settings['seed']
    layouts = [render(backend, group, context) for group in groups]
    original_layouts = [render(backend, group, context) for group in original_groups]
    for group, layout in zip(original_groups, original_layouts, strict=True):
        assert [len(ids) for ids in layout['input_ids']] == [calls[n['id']]['result']['input_tokens'] for n in group]
    assert all(max(map(len, layout['input_ids'])) <= shared.model.max_input_tokens for layout in layouts)
    task_length, = set(layout['task_prefix_length'] for layout in layouts)
    assert all(layout['state_prefix_length'] > task_length for layout in layouts)
    save(output / 'protocol.json', {'settings': settings, 'controller': CONTROLLER, 'groups': groups,
        'layouts': layouts, 'original_layouts': original_layouts, 'task_context': context,
        'source_calls': [calls[node['id']] for group in groups for node in group],
        'scope': 'Same reordered full prompts in every measured cache mode. Original layout is verified by recorded token counts and retained separately; no representation speedup is attributed to cache reuse.'})
    forward_shapes = []

    def capture_shape(module, args, kwargs):
        shape = list(kwargs['input_ids'].shape)
        forward_shapes.append(shape)

    hook = backend.model.model.register_forward_pre_hook(capture_shape, with_kwargs=True)
    reports = {}
    for mode in settings['modes']:
        caches = {name: PrefixCache(settings['cache_capacity']) for name in settings['cache_names']}
        iterations = []
        snapshot = None
        for phase in settings['phases']:
            batches = []
            for index, group in enumerate(groups):
                forward_shapes.clear()
                torch.cuda.synchronize(backend.device)
                started = time.perf_counter()
                if mode == 'task_only':
                    result = score_grouped(backend, [group], 'tiled_shared',
                                          prefix_cache=caches['task'], prefix_lengths=[task_length])
                elif mode == 'two_level':
                    result = score_grouped(backend, [group], 'tiled_shared', prefix_cache=caches['state'],
                                          base_prefix_cache=caches['task'], base_prefix_length=task_length)
                else:
                    raise ValueError(mode)
                torch.cuda.synchronize(backend.device)
                wall = time.perf_counter() - started
                shapes = deepcopy(forward_shapes)
                batches.append({'source_batch_index': settings['batch_indices'][index],
                                'wall_seconds': wall, 'actual_forward_shapes': shapes,
                                'actual_forward_token_slots': sum(batch * width for batch, width in shapes),
                                'result': result})
                save(output / f'{mode}-{phase}-batch-{index}.json', batches[-1])
            iterations.append({'phase': phase, 'batches': batches,
                               'wall_seconds': sum(batch['wall_seconds'] for batch in batches),
                               'computed_tokens': sum(batch['result']['computed_input_tokens'] for batch in batches)})
            if snapshot is not None:
                assert unchanged(caches, snapshot), 'Retained task or execution-state cache was modified by a worker fork.'
            snapshot = snapshots(caches)
        reports[mode] = {'iterations': iterations, 'cached_tensor_bytes': sum(t.numel() * t.element_size() for t in snapshot.values()),
                         'cache_immutable_after_warm_wave': True,
                         'memory_measurement_note': 'Warm measurements include live cloned cache tensors used for exact immutability checks; inspection is outside timing.'}
        save(output / f'{mode}.json', reports[mode])
        snapshot = None
        for cache in caches.values():
            cache.clear()
    hook.remove()
    flattened = {mode: {wave['phase']: [row for b in wave['batches'] for row in b['result']['groups'][0]]
                        for wave in record['iterations']} for mode, record in reports.items()}
    comparisons = {}
    for left_mode, left_phase, right_mode, right_phase in settings['comparisons']:
        comparisons[f'{left_mode}_{left_phase}__{right_mode}_{right_phase}'] = compare(
            flattened[left_mode][left_phase], flattened[right_mode][right_phase])
    result = {'reports': reports, 'comparisons': comparisons, 'worker_count': len(calls),
              'all_choices_equal': all(row['choice_equal'] for rows in comparisons.values() for row in rows),
              'all_logits_exact': all(row['exact_logits'] for rows in comparisons.values() for row in rows),
              'quality_scope': 'Cache execution qualification on actual worker requests; no task correctness or overall request speedup claim.'}
    save(output / 'completion.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('reports', 'comparisons')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.specification.read_text()))
