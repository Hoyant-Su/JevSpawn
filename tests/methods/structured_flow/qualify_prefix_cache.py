import argparse
from copy import deepcopy
import json
from pathlib import Path
import string
import time

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from jev_spawn.algo.structured import common_prefix
from methods.program_execution.grouped import score_grouped


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def workload(settings):
    tasks = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()]
    selected = [task for task in tasks if task['task_id'] == settings['task_id']]
    assert len(selected) == 1
    task = selected[0]
    assert task['kind'] == 'fields'
    fields = task['input']['fields']
    identifiers = [identity for batch in settings['field_batches'] for identity in batch]
    assert identifiers == list(fields), 'Every original finite field must appear exactly once.'
    assert len(settings['field_batches']) >= 2 and all(settings['field_batches'])
    definitions = json.loads(Path(settings['tool_definitions']).read_text())
    problem = {key: deepcopy(task[key]) for key in ['instruction', 'input', 'answer_schema']}
    problem['tools'] = {name: definitions[name] for name in settings['task_tools']}
    context = json.dumps(problem, ensure_ascii=False)
    groups = [[{'id': identity, 'question': fields[identity]['question'],
                'options': deepcopy(fields[identity]['options']), 'context': context,
                'state': context + '\n\nWorker input:\n' + task['input']['state']}
               for identity in batch] for batch in settings['field_batches']]
    return task, groups


def render_workload(backend, groups):
    rows = []
    for group in groups:
        prompts = [controller_prompts([field['state']], field['question'], field['options'],
                   list(string.ascii_uppercase[:len(field['options'])]), CONTROLLER['output_instruction'])[0]
                   for field in group]
        rendered = backend._render(prompts, CONTROLLER['system'])
        sequences = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
        partial = CONTROLLER['user_template'].split('{state}')[0] + group[0]['context']
        prefix_text = backend.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': partial}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prefix_ids = backend.tokenizer(prefix_text, add_special_tokens=False)['input_ids']
        length = common_prefix([prefix_ids, *sequences])
        assert 0 < length <= common_prefix(sequences)
        assert max(map(len, sequences)) <= backend.config['max_input_tokens']
        rows.append({'field_ids': [field['id'] for field in group], 'rendered': rendered,
                     'input_ids': sequences, 'prefix_length': length,
                     'prefix_ids': sequences[0][:length]})
    assert all(row['prefix_ids'] == rows[0]['prefix_ids'] for row in rows)
    return rows


def native_state(state):
    tensors, metadata = {}, []
    for index, layer in enumerate(state.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for name in ['conv_states', 'recurrent_states']:
                for slot, tensor in getattr(layer, name).items():
                    assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
                    tensors[f'{index}/{name}/{slot}'] = tensor
            metadata.append({'kind': type(layer).__name__,
                             'conv_initialized': dict(layer.is_conv_states_initialized),
                             'recurrent_initialized': dict(layer.is_recurrent_states_initialized)})
        else:
            for name in ['keys', 'values']:
                tensor = getattr(layer, name)
                assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
                tensors[f'{index}/{name}'] = tensor
            metadata.append({'kind': type(layer).__name__, 'initialized': layer.is_initialized,
                             'sequence_length': int(layer.get_seq_length())})
    assert any('/keys' in name for name in tensors)
    assert any('/conv_states/' in name for name in tensors)
    assert any('/recurrent_states/' in name for name in tensors)
    return tensors, metadata


class AuditedPrefixCache(PrefixCache):
    def get(self, sequences, compute):
        state, hit = super().get(sequences, compute)
        tensors, self.metadata_before = native_state(state)
        self.before = {name: (tensor, tensor.clone()) for name, tensor in tensors.items()}
        self.native = state
        return state, hit

    def verify(self):
        tensors, metadata = native_state(self.native)
        assert tensors.keys() == self.before.keys() and metadata == self.metadata_before
        for name, (original, snapshot) in self.before.items():
            assert tensors[name] is original, 'A suffix fork replaced a persistent state tensor.'
            assert torch.equal(tensors[name], snapshot), 'A suffix fork mutated persistent state: ' + name
        result = {'tensor_count': len(tensors),
                  'snapshot_bytes': sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
                  'layers': metadata, 'attention_conv_recurrent_exactly_unchanged': True}
        self.before.clear()
        return result


def measure(backend, group, prefix_length, cache):
    shapes = []
    def observe(module, args, kwargs):
        shape = list(kwargs['input_ids'].shape)
        assert len(shape) == 2 and shape[0] <= backend.config['branch_batch_size']
        shapes.append(shape)
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    before = {'allocated_bytes': torch.cuda.memory_allocated(backend.device),
              'reserved_bytes': torch.cuda.memory_reserved(backend.device)}
    started = time.perf_counter()
    hook = backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = score_grouped(backend, [group], 'tiled_shared', prefix_cache=cache,
                               prefix_lengths=[prefix_length])
    finally:
        hook.remove()
    torch.cuda.synchronize(backend.device)
    return {'wall_seconds': time.perf_counter() - started, 'memory_before': before,
            'memory_after': {'allocated_bytes': torch.cuda.memory_allocated(backend.device),
                             'reserved_bytes': torch.cuda.memory_reserved(backend.device)},
            'forward_input_shapes': shapes, 'result': result}


def compare(reference, candidate, group, prefix_length, *, hit):
    assert candidate['persistent_prefix_hit'] is hit
    assert reference['groups'] == candidate['groups'], 'Choices, logits, probabilities or identities changed.'
    for key in ['logical_input_tokens', 'input_tokens', 'suffix_tokens', 'prefix_tokens',
                'logical_field_count', 'option_counts', 'group_sizes']:
        assert reference[key] == candidate[key], 'Logical workload changed: ' + key
    assert candidate['prefix_tokens'] == [prefix_length]
    assert [row['id'] for row in candidate['groups'][0]] == [field['id'] for field in group]
    assert [row['option_ids'] for row in candidate['groups'][0]] == [
        [option['id'] for option in field['options']] for field in group]
    suffix_tokens = sum(candidate['suffix_tokens'][0])
    assert candidate['computed_input_tokens'] == suffix_tokens + (0 if hit else prefix_length)
    assert candidate['prefix_padded_tokens'] == (0 if hit else prefix_length)


def run(settings, output):
    task, groups = workload(settings)
    shared = SharedConfig.load(settings['shared_config'])
    assert all(len(group) <= shared.runtime.branch_batch_size for group in groups)
    prompts = json.loads(Path(settings['method_prompts']).read_text())
    CONTROLLER['option_template'] = prompts['option_template']
    assert '{id}' not in CONTROLLER['option_template']
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', {'settings': settings, 'task': task, 'groups': groups,
                                  'shared_config_text': Path(settings['shared_config']).read_text(),
                                  'scope': 'Paired replay of original finite fields, not generated workers or task-quality evaluation.'})
    started = time.perf_counter()
    backend = Backend(shared.backend())
    torch.cuda.synchronize(backend.device)
    save(output / 'backend.json', {'metadata': backend.metadata, 'load_seconds': time.perf_counter() - started})
    rendered = render_workload(backend, groups)
    save(output / 'rendered.json', rendered)
    prefix_length = rendered[0]['prefix_length']
    rows, baselines = [], {}

    def record(phase, condition, index, cache, expected_hit):
        row = measure(backend, groups[index], prefix_length, cache)
        row.update(phase=phase, condition=condition, batch=index,
                   field_ids=[field['id'] for field in groups[index]])
        save(output / f'{len(rows):02d}-{phase}-{condition}-{index}.json', row)
        rows.append(row)
        result = row['result']
        if condition == 'no_persistent_cache':
            if phase == 'warm_replay':
                compare(baselines[index], result, groups[index], prefix_length, hit=False)
            baselines[index] = result
        compare(baselines[index], result, groups[index], prefix_length, hit=expected_hit)
        expected_forwards = len(result['tiles']) + (0 if expected_hit else 1)
        assert len(row['forward_input_shapes']) == expected_forwards
        return result

    for phase in settings['measurement_phases']:
        for index in range(len(groups)):
            record(phase, 'no_persistent_cache', index, None, False)
        cache = PrefixCache(settings['cache_capacity'])
        record(phase, 'fresh_cache', 0, cache, False)
        for index in range(len(groups)):
            record(phase, 'reused_cache', index, cache, True)
        cache.clear()
        for index in range(1, len(groups)):
            record(phase, 'fresh_cache', index, cache, False)
            cache.clear()

    audit = AuditedPrefixCache(settings['cache_capacity'])
    audits = []
    for index, group in enumerate(groups):
        measured = measure(backend, group, prefix_length, audit)
        check = audit.verify()
        compare(baselines[index], measured['result'], group, prefix_length, hit=index > 0)
        audits.append({'batch': index, 'check': check, 'measurement': measured})
        save(output / f'audit-{index}.json', audits[-1])
    audit.clear()
    save(output / 'completion.json', {'exact_choices_logits_probabilities': True,
         'attention_conv_recurrent_immutable': True, 'shared_prefix_tokens': prefix_length,
         'unique_task_count': 1, 'unique_original_field_count': sum(map(len, groups)),
         'paired_replay_calls': len(rows), 'additional_state_audit_calls': len(audits),
         'measured_field_evaluations_including_replays': sum(row['result']['logical_field_count'] for row in rows),
         'state_audit_field_evaluations': sum(len(group) for group in groups),
         'timing_scope': 'first_pass includes first finite-call cold kernel costs; warm_replay repeats the same sequence after all shapes have executed. Model load is separate. No excluded warmup or allocator reset.',
         'memory_scope': 'Measured passes contain native cache storage only. Additional state audits clone all attention, convolution and recurrent tensors on GPU and are excluded from performance comparisons.',
         'performance': [{'phase': row['phase'], 'condition': row['condition'], 'batch': row['batch'],
                          'wall_seconds': row['wall_seconds'],
                          'peak_allocated_bytes': row['result']['peak_cuda_memory_bytes'],
                          'peak_reserved_bytes': row['result']['peak_cuda_reserved_bytes'],
                          'computed_input_tokens': row['result']['computed_input_tokens'],
                          'logical_input_tokens': row['result']['logical_input_tokens']} for row in rows]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.settings.read_text()), args.output)


if __name__ == '__main__':
    main()
