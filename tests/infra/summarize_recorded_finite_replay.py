import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main(config, destination):
    source = json.loads(Path(config['result']).read_text())
    result = {'source': config['result'], 'verified_preparation_seconds': source['verified_preparation_seconds'],
              'timing_scope': config['timing_scope'], 'phases': {}, 'comparisons': source['comparisons']}
    for report in source['reports']:
        phases, caches = defaultdict(float), defaultdict(Counter)
        details = [record['result'] for record in report['records']]
        for record in report['records']:
            for name, seconds in record['result']['timings'].items():
                phases[name] += seconds
            for name, plans in record['cache_plans'].items():
                for plan in plans:
                    for hit, tokens in zip(plan['hits'], plan['token_counts'], strict=True):
                        caches[name]['hits' if hit else 'misses'] += 1
                        caches[name]['hit_tokens' if hit else 'miss_tokens'] += tokens
        choices = [[answer['choice'] for answer in record['result']['groups'][0]] for record in report['records']]
        ranks = []
        for path in Path(config['output']).glob(f"rank*-{report['phase']}-{report['variant']}.json"):
            rank = json.loads(path.read_text())
            assert [[answer['choice'] for answer in record['result']['groups'][0]] for record in rank['records']] == choices
            ranks.append({'file': str(path), 'elapsed_seconds': rank['elapsed_seconds'],
                          'initial_allocated_bytes': rank['initial_allocated_bytes'],
                          'peak_allocated_bytes': rank['peak_allocated_bytes'], 'peak_reserved_bytes': rank['peak_reserved_bytes']})
        result['phases'][report['phase'] + '/' + report['variant']] = {
            'cohorts': len(details), 'decisions': sum(value['logical_field_count'] for value in details),
            'elapsed_seconds': report['elapsed_seconds'],
            'rankmax_cohort_seconds_sum': sum(record['rankmax_seconds'] for record in report['records']),
            'phases_seconds': dict(phases), 'cache_unique_prefix_lookups': dict(caches),
            'work': {name: sum(value[name] for value in details) for name in
                     ('computed_input_tokens', 'logical_input_tokens', 'padded_input_tokens', 'graph_captures', 'graph_replays')},
            'physical_graph_rows': sum(value['graph_layout']['physical_rows'] for value in details),
            'batch_sizes': dict(Counter(value['batch_size'] for value in details)),
            'all_rank_choices_equal': True, 'ranks': ranks}
    Path(destination).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: values[key] for key in ('cohorts', 'decisions', 'elapsed_seconds', 'work')}
                      for name, values in result['phases'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(json.loads(args.config.read_text()), args.output)
