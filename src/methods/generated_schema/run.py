import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import jsonschema
import torch

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.generated_schema.dag import compatible_batches, eligible_groups, original_input, outcome_context, prepare_unit, validate_plan

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def measured_call(backend, callback, record_tokens=False):
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    times = []

    def token(module, inputs, output):
        torch.cuda.synchronize(backend.device)
        times.append(time.perf_counter())

    hook = backend.model.register_forward_hook(token) if record_tokens else None
    started = time.perf_counter()
    try:
        result = callback()
    finally:
        if hook is not None:
            hook.remove()
    torch.cuda.synchronize(backend.device)
    record = {'elapsed_seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device), 'result': result}
    if 'peak_cuda_memory_bytes' in result:
        record['peak_allocated_bytes'] = max(record['peak_allocated_bytes'], result['peak_cuda_memory_bytes'])
    if 'peak_cuda_reserved_bytes' in result:
        record['peak_reserved_bytes'] = max(record['peak_reserved_bytes'], result['peak_cuda_reserved_bytes'])
    if record_tokens:
        record['decode'] = [{'ttft_seconds': times[0] - started,
                             'inter_token_seconds': [b - a for a, b in zip(times[:count], times[1:count])]}
                            for count in result['output_tokens']]
    return record


def stage(backend, rows, settings, schema, prompts, directory):
    directory.mkdir()
    started = time.perf_counter()
    user_prompts = [prompts['planner_user'].format(task=json.dumps(original_input(row), ensure_ascii=False),
                                                 schema=json.dumps(schema, separators=(',', ':')),
                                                 budgets=json.dumps(settings['budgets'])) for row in rows]
    planning = measured_call(backend, lambda: backend.generate(user_prompts, prompts['planner_system'],
                                                              settings['budgets']['planner_tokens']), True)
    write(directory / 'planning.json', {'prompts': user_prompts, **planning})
    roots, failures = [], []
    for index, row in enumerate(rows):
        try:
            if planning['result']['truncated'][index]:
                raise ValueError('Planner exhausted its token budget before EOS')
            plan, depths = validate_plan(planning['result']['texts'][index], schema, settings['budgets']['max_depth'])
        except (ValueError, jsonschema.ValidationError) as error:
            failures.append({'task_id': row['task_id'], 'stage': 'planning', 'error': str(error)})
        else:
            roots.append({'task': row, 'plan': plan, 'depths': depths, 'outcomes': {}})
    write(directory / 'plans.json', {'valid': [{'task_id': r['task']['task_id'], 'plan': r['plan'], 'depths': r['depths']} for r in roots],
                                   'failures': failures})
    execution = []
    round_index = 0
    while any(len(root['outcomes']) < len(root['plan']['nodes']) for root in roots):
        units = [prepare_unit(root, parents, nodes) for root in roots for parents, nodes in eligible_groups(root)]
        for batch in compatible_batches(units, settings['root_batch_size']):
            record = measured_call(backend, lambda: backend.score_fields([u['state'] for u in batch],
                                                                         [u['fields'] for u in batch], mode='streamed'))
            record.update(round=round_index, root_ids=[u['root']['task']['task_id'] for u in batch],
                          node_ids=[[node['id'] for node in u['nodes']] for u in batch],
                          parents=[u['parents'] for u in batch], states=[u['state'] for u in batch],
                          canonical_fields=[u['fields'] for u in batch])
            execution.append(record)
            for row_index, unit in enumerate(batch):
                for field_index, node in enumerate(unit['nodes']):
                    answer = record['result']['fields'][f'f{field_index}']
                    slot = answer['option_ids'].index(answer['choices'][row_index])
                    option = node['options'][slot]
                    unit['root']['outcomes'][node['id']] = {'status': 'activated', 'choice': option['id'],
                                                           'description': option['description'],
                                                           'probabilities': answer['probabilities'][row_index]}
            write(directory / 'execution.json', execution)
        round_index += 1
    final_units = [{'root': root, 'state': json.dumps({**original_input(root['task']),
                                                     'decision_graph_results': [outcome_context(root, node['id']) for node in root['plan']['nodes']]}, ensure_ascii=False),
                    'fields': root['task']['fields']} for root in roots]
    final = []
    for batch in compatible_batches(final_units, settings['root_batch_size']):
        record = measured_call(backend, lambda: backend.score_fields([u['state'] for u in batch],
                                                                     [u['fields'] for u in batch], mode='streamed'))
        record.update(root_ids=[u['root']['task']['task_id'] for u in batch], states=[u['state'] for u in batch])
        final.append(record)
        for row_index, unit in enumerate(batch):
            unit['root']['prediction'] = {name: result['choices'][row_index] for name, result in record['result']['fields'].items()}
    write(directory / 'final.json', final)
    outputs = [{'task_id': root['task']['task_id'], 'plan': root['plan'], 'outcomes': root['outcomes'],
                'prediction': root['prediction'], 'spawned': len(root['plan']['nodes']),
                'activated': sum(value['status'] == 'activated' for value in root['outcomes'].values()),
                'skipped': sum(value['status'] == 'skipped' for value in root['outcomes'].values())} for root in roots]
    write(directory / 'outputs.json', outputs)
    intervals = [value for row in planning['decode'] for value in row['inter_token_seconds']]
    summary = {'assigned_roots': len(rows), 'valid_plans': len(roots), 'failures': failures,
               'planning_seconds': planning['elapsed_seconds'],
               'execution_seconds': sum(record['elapsed_seconds'] for record in execution),
               'final_seconds': sum(record['elapsed_seconds'] for record in final),
               'total_wall_seconds': time.perf_counter() - started,
               'spawned_nodes': sum(row['spawned'] for row in outputs),
               'activated_nodes': sum(row['activated'] for row in outputs),
               'skipped_nodes': sum(row['skipped'] for row in outputs),
               'mean_spawned_per_assigned_root': sum(row['spawned'] for row in outputs) / len(rows),
               'mean_activated_per_assigned_root': sum(row['activated'] for row in outputs) / len(rows),
               'execution_batches': [{'root_slots': r['result']['batch_size'], 'fields_per_slot': r['result']['field_count'],
                                      'finite_decisions': r['result']['batch_size'] * r['result']['field_count']} for r in execution],
               'planner_itl_median_ms': statistics.median(intervals) * 1000,
               'planner_itl_max_ms': max(intervals) * 1000,
               'planner_intervals_ge_100ms': sum(value >= .1 for value in intervals),
               'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in [planning, *execution, *final])}
    write(directory / 'summary.json', summary)
    print(json.dumps({'stage': directory.name, **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(json.loads(args.settings.read_text()))
    directory = ROOT / 'configs/methods/generated_schema'
    prompts = load_prompt('configs/methods/generated_schema/schema/prompts.json')
    schema = json.loads((directory / 'schema/plan.json').read_text())
    schema['properties']['nodes']['maxItems'] = settings['budgets']['max_nodes']
    options = schema['properties']['nodes']['items']['properties']['options']
    options.update(minItems=settings['budgets']['min_candidates'], maxItems=settings['budgets']['max_candidates'])
    config = resolve_prompts(json.loads(Path(settings['native_config']).read_text()))
    assert config['batch_size'] == settings['root_batch_size'] == settings['task_count']
    rows = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()][:settings['task_count']]
    assert len(rows) == settings['task_count'] and all(len(row['fields']) == 1 for row in rows)
    args.output.mkdir(parents=True)
    config['run_dir'] = str(args.output)
    CONTROLLER['option_template'] = prompts['option_template']
    write(args.output / 'protocol.json', {'settings': settings, 'native': config, 'prompts': prompts, 'schema': schema,
                                        'task_ids': [row['task_id'] for row in rows],
                                        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                                        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True),
                                        'scope': 'Generated semantic intermediate decision DAG. Original benchmark target options are not generated. One shared frozen model. JSON syntax and graph validation follow generation. No retry or fallback on invalid plans.'})
    backend = Backend(config)
    for phase in ['warmup', 'measured']:
        stage(backend, rows, settings, schema, prompts, args.output / phase)
    labels = {row['task_id']: row['labels'] for row in map(json.loads, Path(settings['labels']).read_text().splitlines())}
    for phase in ['warmup', 'measured']:
        path = args.output / phase
        outputs = json.loads((path / 'outputs.json').read_text())
        summary = json.loads((path / 'summary.json').read_text())
        summary['correct_roots'] = sum(row['prediction'] == labels[row['task_id']] for row in outputs)
        summary['accuracy_all_assigned_roots'] = summary['correct_roots'] / len(rows)
        summary['quality_scope'] = 'Development only. Invalid plans count as failures in the full assigned denominator. Labels first opened after both stages completed.'
        write(path / 'summary.json', summary)


if __name__ == '__main__':
    main()
