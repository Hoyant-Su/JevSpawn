import argparse
from importlib import import_module
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.context_window import truncate_prompt
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import common_prefix, padded
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.native_cache_batch import split_native_cache
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import controller_prefix
from tests.infra.action_prefix.replay import request_for


def requests_for(workload, backend, shared, settings):
    service = SimpleNamespace(shared=shared, backend=backend)
    policy = json.loads(Path(workload['protocol']).read_text())['inference']['settings']['input_window']
    notice = load_prompt(policy['prompt'])
    batches = []
    for batch in workload['batches']:
        rows = []
        for packet in batch['requests']:
            request = request_for(packet['field'], packet['messages'], packet['task_id'], service, settings)
            rendered, tokens, _ = truncate_prompt(backend.tokenizer, request.admitted.rendered,
                list(request.admitted.tokens), shared.model.max_input_tokens, policy, notice)
            root = backend.tokenizer.apply_chat_template([packet['messages'][0], {'role': 'user',
                'content': controller_prefix(packet['field']['context'], '')}], **settings['chat_template'])
            root_tokens = backend.tokenizer(root, add_special_tokens=False)['input_ids']
            request.root_tokens = tuple(tokens[:common_prefix([tokens, root_tokens])])
            assert tokens == packet['tokens']
            assert rendered == packet['rendered']
            assert list(request.root_tokens) == packet['root_tokens']
            assert list(backend.answer_label_ids[:len(request.field['options'])]) == packet['candidate_token_ids']
            request.admitted = AdmittedPrompt(packet['rendered'], tuple(packet['tokens']))
            rows.append(request)
        batches.append(rows)
    return batches


def seed_roots(backend, roots, batches):
    sequences = list(dict.fromkeys(tuple(request.root_tokens) for batch in batches for request in batch))

    def compute(missing):
        ids, mask = padded(missing, backend.tokenizer.pad_token_id, backend.device, 'left')
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        return split_native_cache(output.past_key_values, list(map(len, missing)))

    torch.cuda.synchronize(backend.device)
    started = time.perf_counter()
    roots.get_many([list(sequence) for sequence in sequences], compute)
    torch.cuda.synchronize(backend.device)
    return {'wall_seconds': time.perf_counter() - started,
            'unique_roots': len(sequences), 'computed_tokens': sum(map(len, sequences))}


def measure(backend, tail, roots, batches, settings):
    records, logits = [], []
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    turn_started = time.perf_counter()
    for requests in batches:
        shapes = []

        def observe(module, args, kwargs):
            shapes.append(list(kwargs['input_ids'].shape))

        hook = backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        begin.record()
        result = tail.score(requests, [len(request.root_tokens) for request in requests], roots)
        end.record()
        torch.cuda.synchronize(backend.device)
        wall = time.perf_counter() - started
        hook.remove()
        logits.append(tail.last_logits.clone())
        records.append({'wall_seconds': wall, 'cuda_span_ms': begin.elapsed_time(end),
            'eager_forward_shapes': shapes, 'graph_layout': result['graph_layout'],
            'choices': [row['choice'] for row in result['groups'][0]],
            'work': {key: result[key] for key in settings['workload_fields']},
            'action_prefix_reused_by_row': result.get('action_prefix_reused_by_row', []),
            'action_prefix_hit_rows': result.get('action_prefix_hit_rows', 0)})
    torch.cuda.synchronize(backend.device)
    total = time.perf_counter() - turn_started
    return {'wall_seconds': total, 'batches': records}, logits


def compare(left, right, batches):
    rows = []
    for cached, recomputed, requests in zip(left, right, batches, strict=True):
        for cached_row, recomputed_row, request in zip(cached, recomputed, requests, strict=True):
            count = len(request.field['options'])
            a, b = cached_row[:count].float(), recomputed_row[:count].float()
            delta = a - b
            rows.append({'field_id': request.field['id'], 'candidate_count': count,
                'max_absolute_error': delta.abs().max().item(),
                'relative_norm_error': (delta.norm() / b.norm()).item(),
                'argmax_equal': a.argmax().item() == b.argmax().item(),
                'candidate_argmax': a.argmax().item(), 'reference_argmax': b.argmax().item()})
    return rows


@torch.inference_mode()
def run(settings):
    prepared = json.loads(Path(settings['prepared']).read_text())
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    report = {'settings': settings, 'startup': startup, 'pending_tracks': prepared['pending_tracks'],
              'measurement_scope': prepared['measurement_scope'], 'batching': prepared['batching'], 'workloads': []}
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    classes = {name: getattr(import_module(definition['module']), definition['class'])
               for name, definition in settings['implementations'].items()}
    reference_mode, candidate_mode = settings['modes']
    for workload in prepared['workloads']:
        protocol = json.loads(Path(workload['protocol']).read_text())
        inference = protocol['inference']['settings']
        batches = requests_for(workload, backend, shared, settings)
        measurements, output_logits = {}, {}
        for mode in settings['modes']:
            tail = classes[mode](backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                                 inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            warmup, _ = measure(backend, tail, roots, batches, settings)
            repeats, output_logits[mode] = [], []
            for repeat in range(settings['repetitions']):
                roots.clear()
                tail.prefix_cache.clear()
                root_work = seed_roots(backend, roots, batches)
                result, logits = measure(backend, tail, roots, batches, settings)
                assert all(batch['work']['graph_captures'] == 0 for batch in result['batches'])
                result.update(repetition=repeat, root_initialization=root_work)
                repeats.append(result)
                output_logits[mode].append(logits)
            measurements[mode] = {'cold_warmup': warmup, 'repetitions': repeats}
            tail.graphs.clear()
            tail.prefix_cache.clear()
            roots.clear()
        comparisons = [compare(cached, recomputed, batches) for cached, recomputed in
                       zip(output_logits[candidate_mode], output_logits[reference_mode], strict=True)]
        report['workloads'].append({key: value for key, value in workload.items() if key != 'batches'} |
            {'modes': measurements, 'numerical_comparisons': comparisons,
             'batch_shapes': [len(batch) for batch in batches]})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'], 'complete': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    run(settings)
