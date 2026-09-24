from collections import defaultdict
import json


def union(intervals):
    end, busy = float('-inf'), 0.0
    for start, stop in sorted(intervals):
        busy += max(stop - max(end, start), 0.0)
        end = max(end, stop)
    return busy


def kernel_report(path, settings):
    trace = json.loads(path.read_text())
    kernels = [event for event in trace['traceEvents'] if event.get('cat') == settings['kernel_category']]
    activity = [event for event in trace['traceEvents'] if event.get('cat') in settings['device_categories']]
    launches = [event for event in trace['traceEvents'] if event.get('cat') in settings['runtime_categories']
                and any(text in event['name'] for text in settings['launch_names'])]
    syncs = [event for event in trace['traceEvents'] if event.get('cat') in settings['runtime_categories']
             and any(text in event['name'] for text in settings['sync_names'])]
    waits = [event for event in trace['traceEvents'] if event.get('cat') in settings['runtime_categories']
             and any(text in event['name'] for text in settings['event_wait_names'])]
    intervals = [(event['ts'], event['ts'] + event['dur']) for event in activity]
    first, last = min(start for start, stop in intervals), max(stop for start, stop in intervals)
    by_name = defaultdict(list)
    for event in kernels:
        by_name[event['name']].append(event['dur'])
    return {'kernel_count': len(kernels), 'kernel_duration_sum_us': sum(event['dur'] for event in kernels),
        'kernel_busy_union_us': union([(event['ts'], event['ts'] + event['dur']) for event in kernels]),
        'device_busy_union_us': union(intervals), 'device_span_us': last - first,
        'device_idle_within_span_us': last - first - union(intervals),
        'launch_calls': len(launches), 'launch_cpu_duration_us': sum(event['dur'] for event in launches),
        'event_wait_calls': len(waits), 'event_wait_cpu_duration_us': sum(event['dur'] for event in waits),
        'synchronization_calls': len(syncs), 'synchronization_cpu_duration_us': sum(event['dur'] for event in syncs),
        'collective_kernel_duration_sum_us': sum(event['dur'] for event in kernels
            if any(marker in event['name'] for marker in settings['collective_markers'])),
        'kernels': sorted([{'name': name, 'count': len(times), 'total_us': sum(times), 'mean_us': sum(times) / len(times)}
                           for name, times in by_name.items()], key=lambda row: row['total_us'], reverse=True)}


