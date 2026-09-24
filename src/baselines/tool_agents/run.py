import argparse
import json
from pathlib import Path
import time

import torch

from jev_spawn.infra.backend import Backend
from agents import run_batch
from jev_spawn.infra.prompts import load_prompt


def timed_generator(backend):
    def generate(prompts, system, max_new_tokens):
        token_times = []

        def record_token(module, inputs, output):
            torch.cuda.synchronize(backend.device)
            token_times.append(time.perf_counter())

        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        hook = backend.model.register_forward_hook(record_token)
        try:
            result = backend.generate(prompts, system, max_new_tokens)
        finally:
            hook.remove()
        assert len(token_times) >= max(result['output_tokens'])
        result['decode'] = []
        for count in result['output_tokens']:
            times = token_times[:count]
            result['decode'].append({
                'forward_completion_seconds': [stamp - started for stamp in times],
                'ttft_seconds': times[0] - started,
                'inter_token_seconds': [right - left for left, right in zip(times, times[1:])],
            })
        return result
    return generate


def save_batch(path, results, stage, batch_index, repeat):
    for result in results:
        result.update(stage=stage, batch_index=batch_index, repeat=repeat)
    with path.open('a') as stream:
        stream.write(''.join(json.dumps(result) + '\n' for result in results))
    print(json.dumps({'stage': stage, 'repeat': repeat, 'batch_index': batch_index,
                      'batch_tasks': len(results),
                      'failed_tasks': sum(result['status'] == 'failed' for result in results),
                      'batch_seconds': results[0]['batch_elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ['backend-config', 'agent-config', 'prompts', 'tasks', 'output', 'warmup-tasks']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--warmup-task-count', type=int, required=True)
    parser.add_argument('--task-count', type=int, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--method', choices=['react', 'compiler', 'reasoning'], required=True)
    parser.add_argument('--measurement-kind', choices=['timing', 'memory'], default='timing')
    args = parser.parse_args()
    config = json.loads(args.agent_config.read_text())
    prompts = load_prompt(args.prompts)
    tasks = list(map(json.loads, args.tasks.read_text().splitlines()))[:args.task_count]
    assert len(tasks) == args.task_count
    assert args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'predictions.jsonl'
    previous = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    completed = {(row['repeat'], row['task_id']) for row in previous}
    batches = [tasks[start:start + config['batch_size']] for start in range(0, len(tasks), config['batch_size'])]
    pending = [(repeat, index, batch) for repeat in range(args.repeats) for index, batch in enumerate(batches)
               if not all((repeat, task['task_id']) in completed for task in batch)]
    assert all(not any((repeat, task['task_id']) in completed for task in batch) for repeat, _, batch in pending), 'Incomplete saved batch; use a separate run directory.'
    if not pending:
        return
    backend_config = json.loads(args.backend_config.read_text())
    assert config['seed'] == backend_config['seed']
    assert config['batch_size'] == backend_config['batch_size']
    assert config['temperature'] == 0
    warmup = list(map(json.loads, args.warmup_tasks.read_text().splitlines()))[:args.warmup_task_count]
    assert len(warmup) == args.warmup_task_count == config['batch_size']
    assert not {task['task_id'] for task in warmup} & {task['task_id'] for task in tasks}
    backend = Backend(backend_config)
    assert backend.config['seed'] == config['seed']
    generate = timed_generator(backend)
    metadata = {'method': args.method, 'config': config, 'backend': backend.metadata,
                'measurement_kind': args.measurement_kind,
                'memory_scope': 'CUDA allocated peak reset before each complete root batch after synchronization; includes resident model weights and every model call in the workflow. Loading and warmup peaks excluded. Per-call peaks are cumulative within the root batch; reserved memory is reported separately.',
                'tasks': str(args.tasks), 'task_count': len(tasks), 'repeats': args.repeats,
                'sampling': {'do_sample': False, 'temperature': config['temperature'], 'seed': backend.config['seed']},
                'warmup_tasks': str(args.warmup_tasks), 'warmup_task_count': len(warmup),
                'warmup_policy': 'Separate development batch, then one complete same-workload pass before measured repeats. Every pass performs fresh generation; warmup answers are not reused and labels are unavailable to inference.',
                'shared_resource_ceiling': {'model_calls_per_task': config['max_model_calls_per_task'],
                                            'output_tokens_per_task': config['max_output_tokens_per_task'],
                                            'tokens_per_call': config['max_new_tokens'],
                                            'root_batch_size': config['batch_size'],
                                            'tool_concurrency': config['tool_concurrency']},
                'decode_timing_scope': 'CUDA-synchronized outer-model forward completion per output token, including EOS. ITL differences exclude TTFT; intervals include host sampling between forwards. TTFT includes prompt preparation.',
                'compiler_variant': 'Non-streaming planner; dependency-ready concurrent tool execution and bounded replanning.',
                'timing_scope': 'Root-batch wall time includes planning, generation, actual tools and joining; excludes model loading, warmup passes and output file writes.'}
    (args.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_batch(args.output / 'development-warmup.jsonl',
               run_batch(generate, warmup, args.method, config, prompts), 'development_warmup', 0, None)
    for index, batch in enumerate(batches):
        save_batch(args.output / 'workload-warmup.jsonl',
                   run_batch(generate, batch, args.method, config, prompts), 'workload_warmup', index, None)
    for repeat, batch_index, batch in pending:
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        allocated_before = torch.cuda.memory_allocated(backend.device)
        results = run_batch(generate, batch, args.method, config, prompts)
        torch.cuda.synchronize(backend.device)
        for result in results:
            result.update(allocated_before_bytes=allocated_before,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.device))
        stage = 'memory_measurement' if args.measurement_kind == 'memory' else 'measured'
        save_batch(path, results, stage, batch_index, repeat)


if __name__ == '__main__':
    main()
