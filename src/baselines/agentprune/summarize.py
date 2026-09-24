import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    completion = json.loads((args.run / 'completion.json').read_text())
    rows = [json.loads((args.run / f'iteration-{i:03d}.json').read_text())
            for i in range(config['training']['num_iters'])]
    batches = [batch for i in range(len(rows)) for batch in
               json.loads((args.run / f'iteration-{i:03d}-batches.json').read_text())]
    intervals = [value for batch in batches for decoded in batch['decode']
                 for value in decoded['inter_token_seconds']]
    summary = {
        'scope': 'Development topology optimization. Rewards are repeated training observations, not heldout accuracy.',
        'completed_iterations': completion['completed_iterations'],
        'unique_development_tasks': config['task_count'],
        'training_episodes': sum(len(row['episodes']) for row in rows),
        'model_calls': sum(batch['batch_size'] for batch in batches),
        'actual_batch_sizes': [batch['batch_size'] for batch in batches],
        'iteration_wall_seconds': [row['wall_seconds'] for row in rows],
        'iteration_mean_reward': [statistics.mean(row['rewards']) for row in rows],
        'maximum_objective_reconstruction_error': max(abs(row['loss'] - statistics.mean(
            -probability * reward for probability, reward in zip(row['log_probabilities'], row['rewards'])))
            for row in rows),
        'iteration_spatial_active_edges': [sum(row['spatial_masks']) for row in rows],
        'iteration_temporal_active_edges': [sum(row['temporal_masks']) for row in rows],
        'iteration_spatial_gradient_l1': [sum(abs(value) for value in row['spatial_gradients']) for row in rows],
        'temporal_gradient_present': [row['temporal_gradients'] is not None for row in rows],
        'input_tokens': sum(sum(batch['input_tokens']) for batch in batches),
        'output_tokens': sum(sum(batch['output_tokens']) for batch in batches),
        'length_limited_calls': sum(sum(batch['truncated']) for batch in batches),
        'peak_allocated_gib': max(batch['peak_allocated_bytes'] for batch in batches) / 2**30,
        'all_training_itl_median_ms': statistics.median(intervals) * 1000,
        'all_training_itl_max_ms': max(intervals) * 1000,
        'all_training_intervals_at_least_100ms': sum(value >= 0.1 for value in intervals),
        'latency_scope': 'All training intervals include cold initialization. No warmed inference benchmark is claimed.'}
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
