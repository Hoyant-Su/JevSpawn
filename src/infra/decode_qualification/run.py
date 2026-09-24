import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
from transformers import CompileConfig, GenerationConfig, StoppingCriteria

from jev_spawn.infra.backend import Backend
from methods.finite_constraints.program import domains, parse_puzzle


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class CompletionEvents(StoppingCriteria):
    def __init__(self, maximum):
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(maximum)]
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.events[self.count].record()
        self.count += 1
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def prompts_from_source(config):
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    assert len(rows) == config['batch_size']
    directory = Path(config['schema_directory'])
    templates = json.loads((directory / 'prompts.json').read_text())
    prompts = []
    for row in rows:
        state = parse_puzzle(row['puzzle'])
        schema = json.loads((directory / 'scopes.json').read_text())
        schema.update(minItems=len(state['clues']), maxItems=len(state['clues']))
        schema['items']['items']['enum'] = list(domains(state))
        prompts.append(templates['scope_user'].format(
            variables=json.dumps({key: state[key] for key in ['positions', 'variables']}),
            clues=json.dumps(list(enumerate(state['clues'], 1))), schema=json.dumps(schema)))
    return rows, templates['scope_system'], prompts


@torch.inference_mode()
def measure(backend, config, arm, prompts, system, trace):
    events = CompletionEvents(config['max_new_tokens'])
    options = dict(do_sample=False, max_new_tokens=config['max_new_tokens'], use_cache=True,
                   eos_token_id=backend.eos_ids, pad_token_id=backend.tokenizer.pad_token_id,
                   bos_token_id=backend.tokenizer.bos_token_id)
    options['disable_compile'] = arm != 'static_compiled'
    if arm.startswith('static'):
        options['cache_implementation'] = 'static'
    if arm == 'static_compiled':
        options['compile_config'] = CompileConfig(mode='reduce-overhead', fullgraph=True)
    synchronize_times = []

    def synchronize_hook(module, inputs, output):
        torch.cuda.synchronize()
        synchronize_times.append(time.perf_counter())

    hook = backend.model.register_forward_hook(synchronize_hook) if arm == 'eager_sync' else None
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA]) if trace else nullcontext()
    torch.manual_seed(config['seed'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    try:
        with profiler as captured:
            inputs, lengths = backend._encode(backend._render(prompts, system))
            sequences = backend.model.generate(**inputs, generation_config=GenerationConfig(**options),
                                                stopping_criteria=[events], logits_to_keep=1)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - started
    finally:
        if hook is not None:
            hook.remove()
    generated = sequences[:, inputs['input_ids'].shape[1]:]
    ids = generated.tolist()
    counts = [next((i + 1 for i, token in enumerate(row) if token in backend.eos_ids), len(row)) for row in ids]
    intervals = [events.events[i - 1].elapsed_time(events.events[i]) / 1000 for i in range(1, events.count)]
    names = Counter(event.name for event in captured.events()) if trace else Counter()
    graph_events = {name: count for name, count in names.items() if 'graphlaunch' in name.lower()}
    return {'arm': arm, 'seconds': inference_seconds, 'batch_size': len(prompts),
            'input_tokens': lengths, 'output_tokens': counts,
            'output_ids': [row[:count] for row, count in zip(ids, counts)],
            'texts': backend.tokenizer.batch_decode(generated, skip_special_tokens=True),
            'truncated': [row[count - 1] not in backend.eos_ids for row, count in zip(ids, counts)],
            'completion_intervals_seconds': intervals,
            'itl_median_seconds': statistics.median(intervals),
            'itl_max_seconds': max(intervals),
            'allocated_before_bytes': allocated_before,
            'allocated_after_bytes': torch.cuda.memory_allocated(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'synchronizing_hook_calls': len(synchronize_times),
            'cuda_graph_launch_events': graph_events,
            'cuda_graph_replay_observed': bool(graph_events),
            'timing': 'Events are recorded after token selection and resolved after generation. Device intervals include host submission gaps. Trace overhead applies only when trace is true.',
            'trace': trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows, system, prompts = prompts_from_source(config)
    native = json.loads(Path(config['native_config']).read_text())
    assert native['batch_size'] == config['batch_size'] == 8
    assert native['dtype'] == 'bfloat16' and native['attention'] == 'sdpa' and native['kernel'] == 'fla'
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'config': config, 'native': native,
         'task_ids': [row['task_id'] for row in rows], 'system': system, 'prompts': prompts})
    backend = Backend(native)
    save(args.output / 'model.json', backend.metadata)
    reference = None
    for arm in config['arms']:
        if hasattr(backend.model, '_cache'):
            del backend.model._cache
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'arm': arm, 'phase': 'warmup', 'status': 'started'}), flush=True)
        try:
            warm = measure(backend, config, arm, prompts, system, False)
            save(args.output / f'{arm}-warmup.json', warm)
            print(json.dumps({'arm': arm, 'phase': 'measured', 'status': 'started'}), flush=True)
            measured = measure(backend, config, arm, prompts, system, arm == 'static_compiled')
            if reference is None:
                reference = measured['output_ids']
            measured['identical_rows_to_eager_sync'] = sum(a == b for a, b in zip(reference, measured['output_ids']))
            measured['identical_rows_to_own_warmup'] = sum(a == b for a, b in zip(warm['output_ids'], measured['output_ids']))
            save(args.output / f'{arm}-measured.json', measured)
            print(json.dumps({key: measured[key] for key in ['arm', 'seconds', 'output_tokens', 'itl_median_seconds',
                             'itl_max_seconds', 'identical_rows_to_eager_sync', 'cuda_graph_replay_observed']}), flush=True)
        except Exception as error:
            save(args.output / f'{arm}-failure.json', {'arm': arm, 'status': 'not_qualified',
                 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
