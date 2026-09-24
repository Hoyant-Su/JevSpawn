import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text())


def build(settings):
    study = read(settings['study'])
    fixed = study['fixed']
    groups = defaultdict(list)
    for method in settings['methods']:
        groups[method['dataset']].append(method)
    assert set(groups) == {item['dataset'] for item in study['datasets']}
    trials = []
    rng = random.Random(fixed['schedule_seed'])
    for dataset in study['datasets']:
        methods = groups[dataset['dataset']]
        assert len({method['method'] for method in methods}) == len(methods)
        sources = []
        for method in methods:
            config = read(method['settings'])
            native = read(config['native_config'])
            assert config['task_count'] == dataset['task_count']
            assert native['model_path'] == fixed['model_path'] and native['dtype'] == fixed['dtype']
            assert native['batch_size'] == fixed['batch_capacity']
            assert native['max_input_tokens'] == fixed['input_token_limit']
            sources.append([json.loads(line) for line in Path(config['tasks']).read_text().splitlines()])
        assert len(sources[0]) == dataset['task_count'] and all(source == sources[0] for source in sources)
        for memory in study['memory_budget']['gib']:
            order = list(methods)
            rng.shuffle(order)
            for repetition in range(fixed['replications']):
                rotated = order[repetition:] + order[:repetition]
                for position, method in enumerate(rotated):
                    config = {'settings': method['settings'], 'study': settings['study'],
                              'policy': method['policy'], 'memory_gib': memory,
                              'deadline_seconds': max(study['time_budget']['seconds']),
                              **{key: settings[key] for key in ['startup_timeout_seconds', 'poll_seconds', 'join_timeout_seconds']}}
                    trials.append({'dataset': dataset['dataset'], 'method': method['method'],
                                   'memory_gib': memory, 'repetition': repetition,
                                   'position': position, 'gpu_slot': (order.index(method) + repetition) % settings['gpu_slots'],
                                   'config': config})
    by_condition = defaultdict(list)
    for trial in trials:
        by_condition[(trial['dataset'], trial['method'], trial['memory_gib'])].append(trial)
    for group in by_condition.values():
        assert len(group) == fixed['replications']
        assert len({row['gpu_slot'] for row in group}) == min(settings['gpu_slots'], fixed['replications'])
        assert len({row['position'] for row in group}) == fixed['replications']
    return {'status': settings['status'], 'gpu_slots': settings['gpu_slots'],
            'physical_assignment': 'Bind abstract slots to verified idle authorized GPUs immediately before execution.',
            'trial_count': len(trials), 'trials': trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(read(args.settings))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'trial_count': result['trial_count'], 'status': result['status']}))


if __name__ == '__main__':
    main()
