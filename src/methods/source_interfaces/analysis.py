from collections import Counter, defaultdict
import json
import statistics

import numpy as np

from methods.source_interfaces.schema import NONE


def hierarchy_statistics(result, calls):
    sources = [unit['id'] for unit in result['sources']]
    assert len(sources) == len(set(sources))
    active = {value['id']: sources for value in result['references']}
    widths = []
    for level, recorded in enumerate(result['levels']):
        assert recorded['level'] == level
        expected = active if level == 0 else {key: values for key, values in active.items() if len(values) > 1}
        inputs, outputs = defaultdict(list), defaultdict(list)
        for call in calls:
            if call['kind'] != 'worker' or call['context']['level'] != level:
                continue
            groups = call['context']['groups']
            if 'texts' in call['result']:
                choices = [[{'id': group[0]['id'],
                             'choice': group[0]['options'][ord(json.loads(text)['choice']) - ord('A')]['id']}]
                           for group, text in zip(groups, call['result']['texts'])]
            else:
                choices = call['result']['groups']
            assert len(groups) == len(choices)
            for group, answers in zip(groups, choices):
                assert [field['id'] for field in group] == [answer['id'] for answer in answers]
                for field, answer in zip(group, answers):
                    candidates = [option['id'] for option in field['options']]
                    assert answer['choice'] in candidates
                    inputs[field['id']].extend(value for value in candidates if value != NONE)
                    outputs[field['id']].append(answer['choice'])
        assert set(inputs) == set(expected)
        assert all(Counter(inputs[key]) == Counter(values) for key, values in expected.items())
        observed = Counter((key, value) for key, values in outputs.items() for value in values)
        saved = Counter((value['id'], value['choice']) for group in recorded['outputs'] for value in group)
        assert observed == saved
        width = sum(map(len, outputs.values()))
        assert width == recorded['workers']
        widths.append(width)
        for key, values in outputs.items():
            active[key] = [value for value in values if value != NONE]
    for reference in result['references']:
        final = active[reference['id']]
        assert len(final) <= 1
        assert final == ([] if reference['source_id'] is None else [reference['source_id']])
    assert sum(widths) == result['worker_invocations']
    return {'workers_per_level': widths, 'maximum_frontier': max(widths),
            'source_units': len(sources), 'fields': len(active)}


def categories(queries):
    grouped = defaultdict(list)
    for query in queries:
        grouped[query['question_type']].append(query)
    return {name: {'tasks': len(rows), 'completed': sum(row['status'] == 'completed' for row in rows),
                   'exact_match': statistics.mean(row['scores']['primary']['exact_match'] for row in rows),
                   'token_f1': statistics.mean(row['scores']['primary']['token_f1'] for row in rows),
                   'elapsed_seconds': sum(row['elapsed_seconds'] for row in rows)}
            for name, rows in grouped.items()}


def paired_comparisons(methods, indices):
    ours = methods['streamed']['queries']
    times = np.array([row['elapsed_seconds'] for row in ours])
    comparisons = {}
    for arm in ['direct', 'json', 'tiled_independent']:
        other = methods[arm]['queries']
        assert [row['task_id'] for row in other] == [row['task_id'] for row in ours]
        quality = np.array([a['scores']['primary']['exact_match'] - b['scores']['primary']['exact_match']
                            for a, b in zip(ours, other)])
        other_times = np.array([row['elapsed_seconds'] for row in other])
        ratios = other_times[indices].sum(axis=1) / times[indices].sum(axis=1)
        comparisons[arm] = {
            'exact_match_difference': float(quality.mean()),
            'exact_match_interval95': np.quantile(quality[indices].mean(axis=1), [.025, .975]).tolist(),
            'wins': int(np.sum(quality > 0)), 'losses': int(np.sum(quality < 0)),
            'speedup': float(other_times.sum() / times.sum()),
            'task_bootstrap_speedup_interval95': np.quantile(ratios, [.025, .975]).tolist(),
            'timing_scope': 'Paired question resampling with one measured realization per method. This interval describes variation across sampled questions, not repeat-run hardware uncertainty.'}
    return comparisons
