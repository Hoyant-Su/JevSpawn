from collections import Counter, defaultdict
import json


def source_ids(result):
    return {identity for reference in result['references'] for identity in reference['source_ids']}


def validate_hierarchy(result, calls, active):
    sources = [unit['id'] for unit in result['sources']]
    assert len(sources) == len(set(sources))
    widths = []
    for level, recorded in enumerate(result['levels']):
        inputs, outputs = defaultdict(list), defaultdict(list)
        selections = []
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
                    options = {option['id']: option['source_ids'] for option in field['options']}
                    candidates = field['options'][-1]['source_ids']
                    assert len(options) == len(field['options']) == 2 ** len(candidates)
                    subsets = {frozenset(values) for values in options.values()}
                    assert len(subsets) == len(options) and all(values <= set(candidates) for values in subsets)
                    selected = options[answer['choice']]
                    inputs[field['id']].extend(candidates)
                    outputs[field['id']].extend(selected)
                    selections.append((field['id'], answer['choice'], tuple(selected)))
        expected = {identity: values for identity, values in active.items() if values}
        assert set(inputs) == set(expected)
        assert all(Counter(inputs[identity]) == Counter(values) for identity, values in expected.items())
        saved = [(value['id'], value['choice'], tuple(value['source_ids']))
                 for group in recorded['outputs'] for value in group]
        assert Counter(selections) == Counter(saved)
        assert len(saved) == recorded['workers'] and recorded['level'] == level
        widths.append(len(saved))
        active = {identity: outputs[identity] for identity in active}
    for reference in result['references']:
        assert Counter(active[reference['id']]) == Counter(reference['source_ids'])
    assert sum(widths) == result['worker_invocations']
    return {'workers_per_level': widths, 'maximum_frontier': max(widths, default=0),
            'source_units': len(sources), 'fields': len(active), 'returned_source_units': len(source_ids(result))}


def hierarchy_statistics(result, calls):
    sources = [unit['id'] for unit in result['sources']]
    return validate_hierarchy(result, calls, {reference['id']: sources for reference in result['references']})
