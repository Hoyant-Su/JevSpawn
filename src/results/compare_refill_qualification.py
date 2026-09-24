import argparse
import json
from pathlib import Path
import statistics

import yaml


def read(path):
    return json.loads(Path(path).read_text())


def describe(run, quantiles):
    root = Path(run)
    completion = read(root / 'completion.json')
    tasks = [read(path) for path in sorted(root.glob('task-*.json'))]
    assert [task['task_id'] for task in tasks] == completion['task_ids']
    batches = [batch for path in sorted(root.glob('session-*/batches.json')) for batch in read(path)]
    intervals = sorted(value for batch in batches for row in batch['decode']
                       for value in row['inter_token_seconds'])
    assert intervals
    return {
        'completion': completion,
        'session_seconds': [read(path)['elapsed_seconds']
                            for path in sorted(root.glob('session-*/completion.json'))],
        'output_tokens': sum(sum(batch['output_tokens']) for batch in batches),
        'peak_allocated_bytes': max(batch['peak_allocated_bytes'] for batch in batches),
        'peak_reserved_bytes': max(batch['peak_reserved_bytes'] for batch in batches),
        'itl_seconds': {'observations': len(intervals), 'mean': statistics.mean(intervals),
                        'maximum': max(intervals),
                        'quantiles': {str(q): intervals[round(q * (len(intervals) - 1))] for q in quantiles}},
        'tasks': {task['task_id']: task for task in tasks},
    }


def compare(settings):
    comparisons = []
    for method in settings['methods']:
        left, right = [read(Path(method[key]) / 'protocol.json')
                       for key in ('reference_run', 'candidate_run')]
        assert left['tasks'] == right['tasks']
        assert left['method'] == right['method'] and left['prompts'] == right['prompts']
        assert left['tools'] == right['tools']
        config = yaml.safe_load(left['shared_config_text'])
        config['runtime'].update(generation_scheduling='continuous', cache_allocation='shared_refill_cache_arena_v1')
        assert config == yaml.safe_load(right['shared_config_text'])
        reference, candidate = [describe(method[key], settings['latency_quantiles'])
                                for key in ('reference_run', 'candidate_run')]
        old, new = [read(method[key]) for key in ('reference_evaluation', 'candidate_evaluation')]
        assert old['tasks'] == new['tasks'] == len(left['tasks'])
        rows = []
        for task in left['tasks']:
            identity = task['task_id']
            before, after = reference['tasks'][identity], candidate['tasks'][identity]
            rows.append({'task_id': identity, 'dataset': task['dataset'],
                         'reference_status': before['status'], 'candidate_status': after['status'],
                         'reference_seconds': before['elapsed_seconds'], 'candidate_seconds': after['elapsed_seconds'],
                         'same_answer': before['answer'] == after['answer'],
                         'reference_metrics': old['datasets'][task['dataset']],
                         'candidate_metrics': new['datasets'][task['dataset']]})
        del reference['tasks'], candidate['tasks']
        comparisons.append({'method': method['method'], 'sources': method,
                            'reference': reference, 'candidate': candidate, 'tasks': rows})
    Path(settings['output']).write_text(json.dumps({'scope': settings['scope'], 'methods': comparisons}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    compare(read(parser.parse_args().specification))
