import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    return {'minimum': min(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'maximum': max(values)}


def summarize(run):
    protocol = read(run / 'protocol.json')
    loaded = read(run / 'model-load-memory.json')
    evaluation = read(run / 'evaluation.json')
    by_id = {row['task_id']: row['metrics'] for row in evaluation['queries']}
    roots = []
    for row in read(run / 'measured/summary.json'):
        directory = Path(row['directory'])
        compilation = read(directory / 'compilation.json')['generation']
        intervals = [value * 1000 for sample in compilation['decode'] for value in sample['inter_token_seconds']]
        methods = {}
        for prefix in ('', 'direct-'):
            for mode in protocol['settings']['modes']:
                name = prefix + mode
                call, = read(directory / f'{name}-calls.json')
                result = call['result']
                timing = row['summary']['direct'][mode] if prefix else row['summary'][mode]
                tiles = result['tiles']
                peak = max(call['peak_allocated_bytes'], compilation['peak_allocated_bytes']) if not prefix else call['peak_allocated_bytes']
                methods[name] = {
                    'logical_workers': result['logical_field_count'],
                    'root_batch_size': result['root_batch_size'],
                    'peak_live_workers': result['peak_field_concurrency'],
                    'physical_tiles': len(tiles), 'execution_seconds': timing['execution_seconds'],
                    'compiler_seconds': 0 if prefix else timing['compiler_seconds'],
                    'cold_program_inclusive_seconds': timing['execution_seconds'] if prefix else timing['total_seconds'],
                    'ndcg_at_10': by_id[row['task_id']][name]['macro_ndcg'],
                    'peak_allocated_bytes': peak, 'peak_reserved_bytes': call['peak_reserved_bytes'],
                    'peak_extra_allocated_over_loaded_model_bytes': peak - loaded['allocated_bytes'],
                    'peak_extra_allocated_amortized_per_live_worker_bytes': (peak - loaded['allocated_bytes']) / result['peak_field_concurrency'],
                    'tile_transient_extra_per_live_worker_bytes': distribution([
                        (tile['peak_allocated_bytes'] - tile['allocated_before_bytes']) / tile['field_batch_size'] for tile in tiles]),
                    'tile_retained_increment_per_completed_worker_bytes': distribution([
                        (tile['allocated_after_release_bytes'] - tile['allocated_before_bytes']) / tile['field_batch_size'] for tile in tiles]),
                    'before_execution': result['memory_checkpoints']['before_execution'],
                    'memory_checkpoints': result['memory_checkpoints'],
                    'logical_input_tokens': sum(node['input_tokens'] for group in result['groups'] for node in group),
                    'warm_measured': by_id[row['task_id']][name]['warm_measured']}
        roots.append({'task_id': row['task_id'], 'compiler_max_itl_ms': max(intervals),
                      'compiler_intervals_over_100ms': sum(value >= 100 for value in intervals),
                      'methods': methods})
    method_names = list(roots[0]['methods'])
    return {'run': str(run), 'scope': 'Eight development queries executed separately, without test-set inference.',
            'model_load_memory': loaded, 'roots': roots,
            'definitions': {
                'cold_program_inclusive_seconds': 'Fresh program generation plus finite execution and ranking with resident model weights. No schema reuse between queries. Model loading is outside latency.',
                'live_workers': 'At most 128 actual document branches simultaneously evaluated by the shared model. Total logical workers count all real distinct candidate documents for one query.',
                'memory': 'CUDA tensor allocation and allocator reservation are reported separately. Reservation can retain earlier allocations. Per-live-worker quantities amortize an observed simultaneous tile and do not estimate the causal marginal cost of adding one worker.',
                'retained': 'Post-tile incremental retained allocation stores final hidden vectors, not full worker execution state.'},
            'aggregate': {name: {key: distribution([row['methods'][name][key] for row in roots])
                                for key in ['execution_seconds', 'compiler_seconds', 'cold_program_inclusive_seconds',
                                            'ndcg_at_10', 'peak_allocated_bytes', 'peak_extra_allocated_over_loaded_model_bytes',
                                            'peak_extra_allocated_amortized_per_live_worker_bytes']}
                          for name in method_names},
            'paired_speed_ratios': {
                'same_program_execution_independent_over_shared': distribution([
                    row['methods']['tiled_independent']['execution_seconds'] / row['methods']['tiled_shared']['execution_seconds'] for row in roots]),
                'direct_shared_over_generated_shared_cold_program_inclusive': distribution([
                    row['methods']['direct-tiled_shared']['execution_seconds'] / row['methods']['tiled_shared']['cold_program_inclusive_seconds'] for row in roots])}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run)
    (args.run / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'model_load_memory': report['model_load_memory'], 'aggregate': report['aggregate'],
                      'paired_speed_ratios': report['paired_speed_ratios']}, indent=2))


if __name__ == '__main__':
    main()
