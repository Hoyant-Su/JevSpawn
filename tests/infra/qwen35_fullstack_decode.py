import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.qwen35 import install_qwen35_execution
from jev_spawn.runtime.decoding import CapturedDecode


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def measure(backend, shared, inputs, identities, steps, reference):
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    capacity = inputs['input_ids'].shape[1] + shared.generation.max_new_tokens
    decoder = CapturedDecode(backend, len(identities), capacity)
    started = time.perf_counter()
    decoder.prefill(inputs)
    torch.cuda.synchronize(backend.device)
    prefill_seconds = time.perf_counter() - started
    initial_ids, initial_logits = decoder.ids.clone(), decoder.logits.clone()
    snapshot = decoder.capture_snapshot()
    decoder.step()
    eager_logits, eager_ids = decoder.logits.clone(), decoder.ids.clone()
    decoder.restore_capture_snapshot(snapshot)
    started = time.perf_counter()
    decoder.capture(shared.runtime.graph_warmup_steps)
    capture_seconds = time.perf_counter() - started
    decoder.graph.replay()
    graph_exact = torch.equal(eager_logits, decoder.logits) and torch.equal(eager_ids, decoder.ids)
    assert graph_exact, 'Eager and captured outputs differ within the declared execution mode.'
    decoder.restore_capture_snapshot(snapshot)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
    tokens, logits = [], []
    torch.cuda.synchronize(backend.device)
    started = time.perf_counter()
    events[0].record()
    for step in range(steps):
        if reference is not None:
            decoder.ids.copy_(reference['initial_ids'] if step == 0 else reference['tokens'][step - 1])
        decoder.graph.replay()
        events[step + 1].record()
        tokens.append(decoder.ids.clone())
        logits.append(decoder.logits.clone())
    torch.cuda.synchronize(backend.device)
    wall_seconds = time.perf_counter() - started
    intervals = [left.elapsed_time(right) for left, right in zip(events, events[1:])]
    record = {'execution': backend.config['execution'], 'decode_attention_splits': backend.config['decode_attention_splits'],
        'task_ids': identities, 'batch_size': len(identities), 'input_tokens': inputs['attention_mask'].sum(-1).tolist(),
        'padded_width': inputs['input_ids'].shape[1], 'cache_capacity': capacity,
        'generation_capacity': shared.generation.max_new_tokens, 'decode_steps': steps,
        'executed_token_slots': steps * len(identities), 'eager_graph_exact': graph_exact,
        'prefill_seconds': prefill_seconds, 'capture_seconds': capture_seconds,
        'decode_wall_seconds': wall_seconds, 'per_step_ms': intervals,
        'median_itl_ms': statistics.median(intervals), 'mean_itl_ms': statistics.mean(intervals),
        'maximum_itl_ms': max(intervals), 'per_row_tokens_per_second': steps / wall_seconds,
        'aggregate_tokens_per_second': steps * len(identities) / wall_seconds,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device),
        'initial_token_ids': initial_ids[:, 0].tolist(),
        'output_token_ids': torch.cat(tokens, dim=-1).tolist(), 'reference_forced': reference is not None}
    trajectory = {'initial_ids': initial_ids, 'initial_logits': initial_logits, 'tokens': tokens, 'logits': logits}
    if reference is not None:
        record['initial_max_logit_difference'] = (initial_logits.float() - reference['initial_logits'].float()).abs().amax(-1).tolist()
        record['comparisons'] = [{'step': step,
            'same_token': (actual_token[:, 0] == expected_token[:, 0]).tolist(),
            'maximum_logit_difference': (actual.float() - expected.float()).abs().amax(-1).tolist(),
            'mean_logit_difference': (actual.float() - expected.float()).abs().mean(-1).tolist()}
            for step, (actual, expected, actual_token, expected_token) in enumerate(zip(
                logits, reference['logits'], tokens, reference['tokens'], strict=True))]
        record['same_token_count'] = sum(sum(row['same_token']) for row in record['comparisons'])
    return record, trajectory


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    native = SharedConfig.load(settings['native_shared_config'])
    optimized = SharedConfig.load(settings['optimized_shared_config'])
    left, right = asdict(native), asdict(optimized)
    for key in ('execution', 'decode_attention_splits'):
        left['model'].pop(key)
        right['model'].pop(key)
    assert left == right
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', {'settings': settings, 'native': asdict(native), 'optimized': asdict(optimized)})
    backend = Backend(native.backend())
    source = json.loads(Path(settings['single_source']).read_text())
    tokens = source['input_ids'][settings['single_row']]
    ids = torch.tensor([tokens], device=backend.device)
    single = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    captured = json.loads(Path(settings['batch_source']).read_text())[settings['batch_index']]
    rendered = backend.tokenizer.apply_chat_template(captured['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                              return_tensors='pt', truncation=False).to(backend.device)
    assert batch['attention_mask'].sum(-1).tolist() == captured['input_tokens']
    assert len(captured['task_ids']) == native.runtime.batch_size
    requests = [(single, [source['fields'][settings['single_row']]['id']]), (batch, captured['task_ids'])]
    references, records = [], []
    for inputs, identities in requests:
        record, reference = measure(backend, native, inputs, identities, settings['decode_steps'], None)
        records.append(record)
        references.append(reference)
        save(output / f"native_b{len(identities)}.json", record)
        print(json.dumps({key: record[key] for key in ('execution', 'batch_size', 'median_itl_ms', 'per_row_tokens_per_second')}), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    install_qwen35_execution(backend.model, load_resource('qwen35_execution'))
    backend.config = optimized.backend()
    torch.cuda.synchronize(backend.device)
    installation_seconds = time.perf_counter() - started
    for (inputs, identities), reference in zip(requests, references, strict=True):
        record, _ = measure(backend, optimized, inputs, identities, settings['decode_steps'], reference)
        records.append(record)
        save(output / f"optimized_b{len(identities)}.json", record)
        print(json.dumps({key: record[key] for key in ('execution', 'batch_size', 'median_itl_ms', 'per_row_tokens_per_second', 'same_token_count')}), flush=True)
    save(output / 'summary.json', {'settings': settings, 'installation_seconds': installation_seconds,
        'records': records, 'production_benchmarks_started': False})


if __name__ == '__main__':
    main()
