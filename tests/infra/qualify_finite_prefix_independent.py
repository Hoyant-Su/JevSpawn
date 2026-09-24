import argparse
import json
from pathlib import Path

import torch

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.resources import TEMPLATES
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def workload(settings):
    source = Path(settings['source_run'])
    protocol = json.loads((source / 'protocol.json').read_text())
    task, = [task for task in protocol['tasks'] if task['task_id'] == settings['task_id']]
    record = json.loads((source / settings['task_file']).read_text())
    assert record['task_id'] == task['task_id']
    assert all(action['tool'] == 'finish' for action in record['actions'])
    assert TEMPLATES == protocol['resources']['templates'], 'Recorded interface templates differ.'
    environment = TaskEnvironment(task, protocol['tools'], Path(settings['output']) / 'unused_tools', deadline=lambda: None)
    context = environment.reset()
    calls = {call['node']: call for call in record['calls']}
    batches = json.loads((source / settings['batches_file']).read_text())
    # Raw calls carry the immutable task context and independently verify its exact reconstruction.
    raw_messages = [protocol['prompts']['raw_request'].format(context=context, assignment=call['assignment'],
                   input=json.dumps(call['input'], ensure_ascii=False))
                   for call in record['calls'] if call['kind'] == 'raw_value']
    actual_raw = [row[-1]['content'] for batch in batches if 'messages' in batch
                  for identity, row in zip(batch['task_ids'], batch['messages'], strict=True)
                  if identity == task['task_id']]
    assert raw_messages and all(message in actual_raw for message in raw_messages), 'Task context replay differs from recorded requests.'
    groups, originals = [], []
    for index, size in zip(settings['batch_indices'], settings['expected_batch_sizes'], strict=True):
        batch = batches[index]
        assert batch['operation'] == 'finite' and batch['batch_size'] == size
        assert set(batch['task_ids']) == {task['task_id']}
        selected = [calls[identity] for identity in batch['node_ids']]
        groups.append([{'id': call['node'], 'question': call['question'], 'options': call['options'],
                        'context': context, 'state': TEMPLATES['worker_state'].format(
                            context=context, state=json.dumps(call['input'], ensure_ascii=False))}
                       for call in selected])
        originals.append([call['result'] for call in selected])
    return protocol, context, groups, originals


def render(backend, group, context):
    prompts = [controller_prompts([node['state']], node['question'], node['options'],
                list(backend.answer_labels[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
               for node in group]
    rendered = backend._render(prompts, CONTROLLER['system'])
    sequences = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
    partial = CONTROLLER['user_template'].split('{state}')[0] + context
    prefix_text = backend.tokenizer.apply_chat_template(
        [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': partial}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix = backend.tokenizer(prefix_text, add_special_tokens=False)['input_ids']
    length = common_prefix([prefix, *sequences])
    return {'rendered': rendered, 'input_ids': sequences, 'prefix_length': length,
            'suffix_lengths': [len(sequence) - length for sequence in sequences]}


def compare(left, right, settings):
    rows = []
    for expected, actual in zip(left['groups'][0], right['groups'][0], strict=True):
        assert expected['id'] == actual['id'] and expected['option_ids'] == actual['option_ids']
        a, b = torch.tensor(expected['option_logits']), torch.tensor(actual['option_logits'])
        rows.append({'id': expected['id'], 'reference_choice': expected['choice'],
                     'candidate_choice': actual['choice'], 'choice_equal': expected['choice'] == actual['choice'],
                     'exact_logits': torch.equal(a, b), 'maximum_absolute_logit_difference': float((a - b).abs().max()),
                     'within_declared_tolerance': torch.allclose(a, b, atol=settings['absolute_tolerance'],
                                                                rtol=settings['relative_tolerance'])})
    return rows


@torch.inference_mode()
def run(settings):
    protocol, context, groups, originals = workload(settings)
    CONTROLLER.clear()
    CONTROLLER.update(settings['controller_snapshot'])
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', {'settings': settings, 'context': context, 'groups': groups,
        'controller': CONTROLLER, 'scope': 'Actual recorded finite requests; no gold, accuracy claim or altered batch boundaries.'})
    shared = SharedConfig.load(settings['shared_config'])
    backend = Backend(shared.backend())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    source_backend = json.loads((Path(settings['source_run']) / settings['runtime_file']).read_text())['backend']
    assert list(backend.answer_labels) == source_backend['answer_labels']
    assert list(backend.answer_label_ids) == source_backend['answer_label_ids']
    cache = PrefixCache(settings['cache_capacity'])
    reports = []
    for index, (group, recorded) in enumerate(zip(groups, originals, strict=True)):
        rendered = render(backend, group, context)
        assert list(map(len, rendered['input_ids'])) == [row['input_tokens'] for row in recorded], 'Recorded prompt token counts differ.'
        save(output / f'rendered-{index}.json', rendered)
        assert len(group) <= shared.runtime.branch_batch_size
        assert min(rendered['suffix_lengths']) > 1
        length = rendered['prefix_length']
        reference = score_grouped(backend, [group], 'tiled_independent', prefix_lengths=[length])
        cold = score_grouped(backend, [group], 'tiled_shared', prefix_cache=cache, prefix_lengths=[length])
        assert cold['persistent_prefix_hit'] is bool(index)
        warm = score_grouped(backend, [group], 'tiled_shared', prefix_cache=cache, prefix_lengths=[length])
        assert warm['persistent_prefix_hit'] is True
        fresh = PrefixCache(settings['cache_capacity'])
        cold_each = score_grouped(backend, [group], 'tiled_shared', prefix_cache=fresh, prefix_lengths=[length])
        assert cold_each['persistent_prefix_hit'] is False
        fresh.clear()
        report = {'batch_index': settings['batch_indices'][index], 'batch_size': len(group),
                  'recorded': recorded, 'independent': reference, 'persistent_sequence': cold,
                  'warm': warm, 'cold': cold_each,
                  'independent_vs_cold': compare(reference, cold_each, settings),
                  'independent_vs_warm': compare(reference, warm, settings),
                  'cold_vs_warm': compare(cold_each, warm, settings)}
        reports.append(report)
        save(output / f'batch-{index}.json', report)
    cache.clear()
    comparisons = [row for report in reports for key in ('independent_vs_cold', 'independent_vs_warm', 'cold_vs_warm')
                   for row in report[key]]
    result = {'exact_logits': all(row['exact_logits'] for row in comparisons),
              'choices_equal': all(row['choice_equal'] for row in comparisons),
              'within_declared_tolerance': all(row['within_declared_tolerance'] for row in comparisons),
              'batch_sizes': list(map(len, groups)), 'source_run': settings['source_run'],
              'prior_cache_test_limitation': 'Previous qualification compared tiled_shared fresh versus persistent only; it did not use tiled_independent.'}
    save(output / 'completion.json', result)
    print(json.dumps(result), flush=True)
    assert result['within_declared_tolerance'] and result['choices_equal'], 'Finite prefix reuse differs from independent prefill; inspect batch evidence.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.specification.read_text()))
