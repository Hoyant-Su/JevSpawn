import argparse
import json
from pathlib import Path
from statistics import mean, median


def summarize(settings):
    output = Path(settings['output'])
    reference_mode, candidate_mode = settings['modes']
    ranks = [json.loads((output / settings['rank_file'].format(rank=rank)).read_text())
             for rank in settings['rank_ids']]
    rows = []
    for index, workload in enumerate(ranks[0]['workloads']):
        modes = {}
        for mode in settings['modes']:
            repeats = []
            for repetition in range(settings['repetitions']):
                measurements = [rank['workloads'][index]['modes'][mode]['repetitions'][repetition] for rank in ranks]
                batches = measurements[0]['batches']
                repeats.append({'wall_seconds': max(record['wall_seconds'] for record in measurements),
                    'mean_field_batch_wall_seconds': mean(max(record['batches'][position]['wall_seconds']
                         for record in measurements) for position in range(len(batches))),
                    'computed_input_tokens': sum(batch['work']['computed_input_tokens'] for batch in batches),
                    'padded_input_tokens': sum(batch['work']['padded_input_tokens'] for batch in batches),
                    'eager_forward_count': sum(len(batch['eager_forward_shapes']) for batch in batches),
                    'graph_replays': sum(batch['work']['graph_replays'] for batch in batches),
                    'graph_captures': sum(batch['work']['graph_captures'] for batch in batches),
                    'prefix_hit_rows': sum(batch['action_prefix_hit_rows'] for batch in batches),
                    'reused_action_prefix_tokens': sum(sum(batch['action_prefix_reused_by_row']) for batch in batches)})
            wall = [repeat['wall_seconds'] for repeat in repeats]
            modes[mode] = {'mean_wall_seconds': mean(wall), 'median_wall_seconds': median(wall),
                'min_wall_seconds': min(wall), 'max_wall_seconds': max(wall), 'repetitions': repeats}
        comparisons = [comparison for rank in ranks for comparison in
                       rank['workloads'][index]['numerical_comparisons']]
        rows.append({'track': workload['track'], 'task_id': workload['task_id'],
            'turn': workload['turn'], 'finite_request_count': workload['finite_request_count'],
            'batch_shapes': workload['batch_shapes'], 'modes': modes,
            'speedup_ratio': modes[reference_mode]['mean_wall_seconds'] / modes[candidate_mode]['mean_wall_seconds'],
            'max_absolute_logit_error': max(row['max_absolute_error'] for comparison in comparisons for row in comparison),
            'argmax_changed_rows_per_rank_repeat': [sum(not row['argmax_equal'] for row in comparison) for comparison in comparisons],
            'all_argmax_equal': all(row['argmax_equal'] for comparison in comparisons for row in comparison)})
    report = {'measurement_scope': ranks[0]['measurement_scope'], 'batching': ranks[0]['batching'],
        'reference_mode': reference_mode, 'candidate_mode': candidate_mode,
        'timing': 'Mean of repeated synchronized wall measurements; each repetition uses the maximum across tensor-parallel ranks. '
            'Graph capture and root initialization are excluded and stored separately in rank reports. '
            'Each repetition clears cached states then prepares the same real root prefix. '
            'The field-phase wall measures all readouts but excludes beam selection, tool execution and controller calls. '
            'Field-batch latency is not divided by row count and is not autoregressive ITL.',
        'pending_tracks': ranks[0]['pending_tracks'], 'rows': rows}
    (output / settings['summary_file']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'tasks': len(rows), 'argmax_changed_tasks': [row['track'] for row in rows if not row['all_argmax_equal']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    summarize(json.loads(parser.parse_args().config.read_text()))
