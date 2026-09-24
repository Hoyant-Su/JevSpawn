import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def union_length(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def summarize(config):
    rows = []
    for source in config['sources']:
        for path in sorted(Path(source['directory']).glob(config['trace_glob'])):
            events = json.loads(path.read_text())['traceEvents']
            scope, = [event for event in events if event.get('cat') == 'user_annotation'
                      and event['name'] == config['scope']]
            start, stop = scope['ts'], scope['ts'] + scope['dur']
            device = [event for event in events if event.get('cat') in config['device_categories']]
            intervals = [(max(start, event['ts']), min(stop, event['ts'] + event['dur']))
                         for event in device if event['ts'] < stop and event['ts'] + event['dur'] > start]
            busy = union_length(intervals)
            runtime = [event for event in events if event.get('cat') == 'cuda_runtime']
            durations, counts = defaultdict(float), Counter()
            for event in runtime:
                durations[event['name']] += event['dur']
                counts[event['name']] += 1
            summary_path = Path(str(path).replace(config['trace_suffix'], config['summary_suffix']))
            summary = json.loads(summary_path.read_text())
            units = config['microseconds_per_ms']
            rows.append({'source': str(path), 'root_count': source['roots'],
                'finite_batch_size': len(summary['requests']),
                'input_tokens': [len(request['input_ids']) for request in summary['requests']],
                'service_scope_ms': scope['dur'] / units,
                'gpu_activity_union_ms': busy / units,
                'no_recorded_gpu_activity_ms': (scope['dur'] - busy) / units,
                'device_event_counts': dict(Counter(event['cat'] for event in device)),
                'cuda_runtime': {name: {'calls': counts[name], 'total_ms': duration / units}
                                 for name, duration in sorted(durations.items(), key=lambda item: -item[1])}})
    Path(config['output']).write_text(json.dumps({'scope': config['measurement_scope'], 'rows': rows}, indent=2) + '\n')
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key != 'cuda_runtime'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    summarize(json.loads(parser.parse_args().config.read_text()))
