import re

from methods.evidence_interfaces.interfaces import InvalidResponse, compact


def words(text):
    units = [(match.start(), match.end()) for match in re.finditer(r'\s*\S+\s*', text)]
    if not units or units[0][0] != 0 or units[-1][1] != len(text):
        raise InvalidResponse('The response cannot be fully indexed into nonempty word intervals.')
    assert all(left[1] == right[0] for left, right in zip(units, units[1:]))
    return units


def context(row):
    return compact({'reference': row['source_info'], 'response': row['response']})


def partition(ends, count):
    if not ends or ends != sorted(set(ends)) or ends[-1] != count or ends[0] < 1:
        raise InvalidResponse('Generated endpoints do not partition the complete response.')
    return [{'id': f'span{i}', 'start': start, 'end': end, 'depth': 1, 'parent': None}
            for i, (start, end) in enumerate(zip([0, *ends[:-1]], ends))]


def field(node, row, units, prompts, operators):
    start, end = units[node['start']][0], units[node['end'] - 1][1]
    options = [option for option in operators['options']
               if node['end'] - node['start'] > 1 or option['id'] in operators['terminal_ids']]
    return {'id': node['id'], 'state': context(row), 'options': options,
            'question': prompts['worker_question'].format(start=node['start'], end=node['end'],
                                                         text=row['response'][start:end])}


def refine(node):
    middle = (node['start'] + node['end']) // 2
    assert node['start'] < middle < node['end']
    return [{'id': node['id'] + f'.{index}', 'start': start, 'end': end,
             'depth': node['depth'] + 1, 'parent': node['id']}
            for index, (start, end) in enumerate([(node['start'], middle), (middle, node['end'])])]


def locate(response, values):
    intervals = []
    for value in values:
        matches = list(re.finditer('(?=' + re.escape(value['text']) + ')', response))
        if value['occurrence'] > len(matches):
            raise InvalidResponse('A generated span occurrence does not exist in the response.')
        start = matches[value['occurrence'] - 1].start()
        intervals.append([start, start + len(value['text'])])
    if len({tuple(interval) for interval in intervals}) != len(intervals):
        raise InvalidResponse('The final response contains duplicate spans.')
    return intervals
