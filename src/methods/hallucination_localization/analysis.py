import json

from methods.hallucination_localization.inputs import field, partition, refine, words


def positions(spans, length):
    result = set()
    for start, end in spans:
        assert 0 <= start < end <= length
        result.update(range(start, end))
    return result


def overlap(result, labels, length):
    gold = positions([[label['start'], label['end']] for label in labels], length)
    valid = result['status'] == 'completed'
    predicted = positions(result['spans'], length) if valid else set()
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else float(valid)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1 if valid else 0.0,
            'gold_present': bool(gold), 'predicted_present': bool(predicted) if valid else None,
            'response_correct': int(valid and bool(gold) == bool(predicted))}


def validate_tree(row, result, calls, prompts, operators):
    units = words(row['response'])
    planner, = [call for call in calls if call['kind'] == 'planner']
    plan = json.loads(planner['result']['texts'][0])
    roots = partition(plan['ends'], len(units))
    expected = {node['id']: node for node in roots}
    actual = {node['id']: node for node in result['nodes']}
    assert len(actual) == len(result['nodes'])
    for node in result['nodes']:
        assert {key: node[key] for key in expected[node['id']]} == expected[node['id']]
        if node['choice'] == 'refine':
            expected.update({child['id']: child for child in refine(node)})
    assert set(actual) == set(expected)
    returned = {}
    for call in calls:
        if call['kind'] != 'worker':
            continue
        fields = call['context']['fields']
        for item in fields:
            assert item == field(actual[item['id']], row, units, prompts, operators)
        if 'texts' in call['result']:
            choices = [item['options'][ord(json.loads(text)['choice']) - ord('A')]['id']
                       for item, text in zip(fields, call['result']['texts'])]
        else:
            choices = [answer['choice'] for answer in call['result']['groups'][0]]
        assert len(choices) == len(fields)
        for item, choice in zip(fields, choices):
            assert item['id'] not in returned
            returned[item['id']] = choice
    assert returned == {node['id']: node['choice'] for node in result['nodes']}
    leaves = sorted((node for node in result['nodes'] if node['choice'] != 'refine'), key=lambda node: node['start'])
    assert leaves[0]['start'] == 0 and leaves[-1]['end'] == len(units)
    assert all(a['end'] == b['start'] for a, b in zip(leaves, leaves[1:]))
    assert {node['id'] for node in leaves} == {node['id'] for node in result['leaves']}
    spans = [[units[node['start']][0], units[node['end'] - 1][1]] for node in leaves if node['choice'] == 'hallucinated']
    assert result['spans'] == spans
    assert len(actual) == result['worker_invocations']
    return {'initial_spans': len(roots), 'workers': len(actual), 'terminal_spans': len(leaves),
            'refinements': sum(node['choice'] == 'refine' for node in actual.values()),
            'maximum_depth': max(node['depth'] for node in actual.values())}
