import argparse
import json
from pathlib import Path

from tests.infra.action_prefix.profile_events import kernel_report


def summarize(settings):
    output = Path(settings['output'])
    ranks = [json.loads((output / settings['rank_file'].format(rank=rank)).read_text())
             for rank in settings['rank_ids']]
    rows = []
    for index, source in enumerate(ranks[settings['summary_rank']]['profiles']):
        kernels = kernel_report(Path(source['trace']), settings)
        row = {'track': source['track'], 'mode': source['mode'],
            'kernel_profile': kernels, 'phase_cuda_spans': source['phase_cuda_events'],
            'eager_forward_shapes': source['eager_forward_shapes'],
            'graph_replays': sum(batch['graph_replays'] for batch in source['batches']),
            'computed_input_tokens': sum(batch['computed_input_tokens'] for batch in source['batches']),
            'profiled_wall_max_rank_seconds': max(rank['profiles'][index]['profiled_wall_seconds'] for rank in ranks),
            'unprofiled_wall_max_rank_seconds': max(rank['profiles'][index]['unprofiled']['wall_seconds'] for rank in ranks),
            'cpu_self_time': sorted(source['operators'], key=lambda event: event['self_cpu_time_total_us'], reverse=True)}
        rows.append(row)
    report = {'scope': ranks[0]['scope'], 'rows': rows,
        'interpretation': 'Kernel times and device unions describe the profiled rank, while wall maxima cover all tensor-parallel ranks. '
            'NCCL kernel duration includes cross-rank waiting and is not useful arithmetic time. '
            'Non-NCCL kernels include computation and cache operations. CUDA event phase spans overlap and include CPU launch gaps. '
            'Event waits are separated from CPU-blocking CUDA synchronization calls. '
            'Device inactivity is span minus the union of kernel, memcpy and memset intervals, not wall minus kernel duration sum. '
            'CPU self time is exclusive; parent phase spans are inclusive. The profiler and phase events perturb timing.'}
    (output / settings['summary_file']).write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        kernel = row['kernel_profile']
        print(json.dumps({'track': row['track'], 'mode': row['mode'], 'eager_forwards': len(row['eager_forward_shapes']),
            'graph_replays': row['graph_replays'], 'kernels': kernel['kernel_count'],
            'non_nccl_ms': (kernel['kernel_duration_sum_us'] - kernel['collective_kernel_duration_sum_us']) / 1000,
            'nccl_ms': kernel['collective_kernel_duration_sum_us'] / 1000,
            'kernel_busy_ms': kernel['kernel_busy_union_us'] / 1000,
            'device_idle_ms': kernel['device_idle_within_span_us'] / 1000,
            'launches': kernel['launch_calls'], 'syncs': kernel['synchronization_calls']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    summarize(json.loads(parser.parse_args().config.read_text()))
