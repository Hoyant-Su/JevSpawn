import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import torch

from baselines.latentmas.adapter import SOURCE, method, task_messages
from baselines.latentmas.io import read_jsonl
from baselines.latentmas.adapter import task_item
from jev_spawn.infra.backend import Backend


def save(path, value):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def preflight(runner, tasks, config, native):
    records = []
    for start in range(0, len(tasks), config['root_batch_size']):
        batch = tasks[start:start + config['root_batch_size']]
        widths = []
        for agent in runner.agents:
            rendered = [runner.model.render_chat(task_messages(
                role=agent.role, question=task_item(task)['question'], context='',
                method=runner.method_name, args=runner.args)) for task in batch]
            tokens = runner.model.tokenizer(rendered, add_special_tokens=False)['input_ids']
            widths.append(max(map(len, tokens)))
        maximum = sum(widths) + (len(runner.agents) - 1) * config['latent_steps'] + config['final_tokens']
        assert maximum <= native['max_input_tokens'], 'Full role history and final budget exceed the context limit.'
        records.append({'start': start, 'batch_size': len(batch), 'role_padded_widths': widths,
                        'maximum_physical_history_tokens': maximum, 'truncation': False})
    return records


def measure(runner, backend, tasks, config, seed):
    torch.manual_seed(seed)
    runner.model.reset(capture=False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = runner.run_batch([task_item(task) for task in tasks])
    torch.cuda.synchronize()
    records = []
    assert len(predictions) == len(runner.model.output_ids) == len(tasks)
    for task, prediction, tokens in zip(tasks, predictions, runner.model.output_ids):
        boxes = re.findall(r'\\boxed\{([^}]*)\}', prediction['raw_prediction'])
        options = {option['id'] for option in task['fields']['q0']['options']}
        boxed = boxes[-1].strip() if boxes else None
        upstream = (prediction['prediction'] or '').upper()
        answer = boxed if boxed in options and boxed == upstream else None
        records.append({'task_id': task['task_id'], 'answer': answer,
                        'upstream_prediction': prediction['prediction'],
                        'raw_prediction': prediction['raw_prediction'], 'agents': prediction['agents'],
                        'output_tokens': len(tokens),
                        'truncated': len(tokens) == config['final_tokens'] and tokens[-1] not in backend.eos_ids,
                        'first_token_seconds': runner.model.completion_times[0] - started,
                        'final_token_ready_seconds': runner.model.completion_times[len(tokens) - 1] - started})
    elapsed = time.perf_counter() - started
    for record in records:
        record['request_to_answer_seconds'] = elapsed
    intervals = [value for row in runner.model.itl for value in row]
    return {'seconds': elapsed, 'batch_size': len(tasks), 'seed': seed,
            'role_seconds': runner.model.role_times, 'predictions': records,
            'output_ids': runner.model.output_ids, 'per_sample_itl_seconds': runner.model.itl,
            'itl_max_seconds': max(intervals, default=None),
            'itl_over_100ms': sum(value >= .1 for value in intervals),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'restored_padding_rows': runner.model.model.restored_rows,
            'forward_calls': runner.model.model.records}


def evaluate(output, tasks, config):
    blocks = [json.loads(path.read_text()) for path in sorted(output.glob('block-*.json'))]
    records = [record for block in blocks for record in block['predictions']]
    assert [record['task_id'] for record in records] == [task['task_id'] for task in tasks]
    labels = {row['task_id']: row['labels']['q0'] for row in read_jsonl(config['labels'])}
    correct = sum(record['answer'] == labels[record['task_id']] for record in records)
    intervals = [value for block in blocks for row in block['per_sample_itl_seconds'] for value in row]
    seconds = sum(block['seconds'] for block in blocks)
    summary = {'tasks': len(records), 'blocks': len(blocks), 'actual_batch_sizes': [block['batch_size'] for block in blocks],
               'correct': correct, 'accuracy': correct / len(records),
               'valid_answers': sum(record['answer'] is not None for record in records),
               'truncated': sum(record['truncated'] for record in records),
               'generated_tokens': sum(record['output_tokens'] for record in records),
               'seconds': seconds, 'tasks_per_second': len(records) / seconds,
               'mean_request_to_answer_seconds': statistics.mean(record['request_to_answer_seconds'] for record in records),
               'mean_final_token_ready_seconds': statistics.mean(record['final_token_ready_seconds'] for record in records),
               'peak_allocated_bytes': max(block['peak_allocated_bytes'] for block in blocks),
               'peak_reserved_bytes': max(block['peak_reserved_bytes'] for block in blocks),
               'itl_max_seconds': max(intervals), 'itl_median_seconds': statistics.median(intervals),
               'itl_p95_seconds': statistics.quantiles(intervals, n=100)[94],
               'itl_over_100ms': sum(value >= .1 for value in intervals),
               'latent_steps_per_role': config['latent_steps'], 'final_token_budget': config['final_tokens'],
               'scope': 'One formal pass over the complete declared evaluation input. Labels were read only after all block outputs were committed.'}
    save(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    native = json.loads(Path(config['native_config']).read_text())
    tasks = read_jsonl(config['tasks'])
    warm = read_jsonl(config['warmup_tasks'])
    assert len(tasks) == config['task_count']
    assert native['batch_size'] == config['root_batch_size'] == 8
    assert native['seed'] == config['seed'] and native['dtype'] == 'bfloat16'
    assert native['max_input_tokens'] == config['context_tokens'] == 8192
    assert not ({task['task_id'] for task in tasks} & {task['task_id'] for task in warm})
    protocol = {'settings': config, 'native': native, 'task_ids': [task['task_id'] for task in tasks],
                'upstream_revision': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()}
    if args.resume:
        assert json.loads((args.output / 'protocol.json').read_text()) == protocol
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / 'protocol.json', protocol)
    completed = sorted(args.output.glob('block-*.json'))
    assert [path.name for path in completed] == [f'block-{index:03d}.json' for index in range(len(completed))]
    for index, path in enumerate(completed):
        expected = tasks[index * config['root_batch_size']:(index + 1) * config['root_batch_size']]
        assert [row['task_id'] for row in json.loads(path.read_text())['predictions']] == [row['task_id'] for row in expected]
    attempt = args.output / f'attempt-{len(list(args.output.glob("attempt-*"))):03d}'
    attempt.mkdir()
    setup_started = time.perf_counter()
    backend = Backend(native)
    runner = method(backend, config, tasks + warm)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - setup_started
    before = torch.cuda.memory_allocated()
    alignment_started = time.perf_counter()
    runner.model._ensure_latent_realign_matrix(runner.model.model, backend.device, runner.args)
    torch.cuda.synchronize()
    save(attempt / 'setup.json', {'model_load_seconds': load_seconds,
         'alignment_seconds': time.perf_counter() - alignment_started,
         'alignment_resident_bytes': torch.cuda.memory_allocated() - before,
         'model': {key: value for key, value in backend.metadata.items() if key != 'controller'},
         'roles': [agent.name for agent in runner.agents], 'completed_blocks_at_resume': len(completed),
         'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
         'gpus': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)})
    save(attempt / 'preflight.json', preflight(runner, tasks, config, native))
    with torch.inference_mode():
        for size in config['warmup_batch_sizes']:
            save(attempt / f'warmup-{size}.json', measure(runner, backend, warm[:size], config, native['seed']))
            print(json.dumps({'warmup_batch_size': size, 'completed': True}), flush=True)
        for start in range(len(completed) * config['root_batch_size'], len(tasks), config['root_batch_size']):
            index = start // config['root_batch_size']
            block = measure(runner, backend, tasks[start:start + config['root_batch_size']], config, native['seed'])
            block.update(block_index=index, source_offset=start, attempt=attempt.name)
            save(args.output / f'block-{index:03d}.json', block)
            print(json.dumps({'block': index, 'completed_tasks': start + block['batch_size'], 'seconds': block['seconds'],
                              'valid': sum(row['answer'] is not None for row in block['predictions']),
                              'max_itl_seconds': block['itl_max_seconds']}), flush=True)
    evaluate(args.output, tasks, config)


if __name__ == '__main__':
    main()
