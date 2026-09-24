import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
METRICS = ['tasks', 'correct', 'valid', 'accuracy', 'elapsed_seconds',
           'output_tokens', 'peak_allocated_bytes',
           'max_itl_ms', 'intervals_over_100ms', 'budget']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    records = []
    for entry in settings['runs']:
        run = ROOT / entry['run']
        protocol = json.loads((run / 'protocol.json').read_text())
        if entry['format'] in {'formal_choices', 'single_reasoning'}:
            source = run / 'evaluation.json'
            result = json.loads(source.read_text())
            blocks = [json.loads(path.read_text()) for path in sorted(run.glob('block-*/complete.json'))]
            assigned = [task['task_id'] for task in protocol['tasks']]
            completed = [task for block in blocks for task in block['task_ids']]
            assert result['elapsed_seconds'] == sum(block['metrics']['elapsed_seconds'] for block in blocks)
            metrics = {key: result[key] for key in METRICS}
            detail_keys = ['truncated_calls', 'model_calls']
            if entry['format'] == 'formal_choices':
                detail_keys += ['adapter_exceptions', 'model_request_errors']
            else:
                detail_keys += ['failure_reasons']
            details = {key: result[key] for key in detail_keys}
        elif entry['format'] == 'latentmas':
            source = run / 'summary.json'
            result = json.loads(source.read_text())
            blocks = [json.loads(path.read_text()) for path in sorted(run.glob('block-*.json'))]
            assigned = protocol['task_ids']
            completed = [row['task_id'] for block in blocks for row in block['predictions']]
            assert result['seconds'] == sum(block['seconds'] for block in blocks)
            metrics = {key: result[key] for key in ['tasks', 'correct', 'accuracy', 'peak_allocated_bytes']}
            metrics.update(valid=result['valid_answers'], elapsed_seconds=result['seconds'],
                           output_tokens=result['generated_tokens'], max_itl_ms=result['itl_max_seconds'] * 1000,
                           intervals_over_100ms=result['itl_over_100ms'],
                           budget={'latent_roles': len(blocks[0]['role_seconds']) - 1,
                                   'steps_per_role': result['latent_steps_per_role'],
                                   'final_token_budget': result['final_token_budget']})
            details = {'final_generation_truncations': result['truncated'],
                       'logical_role_calls': sum(len(block['role_seconds']) * block['batch_size'] for block in blocks)}
        elif entry['format'] == 'agentprune':
            source = run / 'evaluation.json'
            result = json.loads(source.read_text())
            blocks = [json.loads(path.read_text()) for path in sorted(run.glob('block-*/complete.json'))]
            assigned = [task['task_id'] for task in protocol['tasks']]
            completed = [row['task_id'] for block in blocks for row in block['result']['records']]
            assert result['elapsed_seconds'] == sum(block['metrics']['elapsed_seconds'] for block in blocks)
            metrics = {key: result[key] for key in METRICS if key != 'budget'}
            metrics['budget'] = {'generation': protocol['settings']['generation'],
                                 'communication_rounds': protocol['settings']['num_rounds'],
                                 'training_config': protocol['training_config']}
            details = {key: result[key] for key in ['truncated_calls', 'model_calls', 'training_provenance']}
        else:
            raise ValueError('Unsupported result format: ' + entry['format'])
        assert completed == assigned and len(set(assigned)) == len(assigned)
        assert metrics['tasks'] == entry['tasks'] == len(assigned)
        assert metrics['accuracy'] == metrics['correct'] / metrics['tasks']
        native = protocol['native']
        records.append({**entry, **metrics, 'details': details,
                        'model_path': native['model_path'], 'dtype': native['dtype'],
                        'batch_capacity': native['batch_size'],
                        'input_token_limit': native['max_input_tokens'],
                        'task_scope': {'assigned_tasks': len(assigned),
                                       'tasks_source': protocol['settings']['tasks'],
                                       'selection': entry['selection']},
                        'source': str(source.relative_to(ROOT))})
    output = {'scope': settings['scope'], 'records': records}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    columns = ['method', 'dataset', 'tasks', 'correct', 'valid', 'accuracy',
               'elapsed_seconds', 'peak_allocated_bytes', 'output_tokens',
               'max_itl_ms', 'intervals_over_100ms']
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps([{key: row[key] for key in columns} for row in records]))


if __name__ == '__main__':
    main()
