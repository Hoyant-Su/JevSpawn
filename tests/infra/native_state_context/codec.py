from copy import deepcopy
from itertools import takewhile
import json

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import string_leaves


def encode_events(events, preceding):
    protocol = load_prompt('jevspawn.state')
    serialization = protocol['history_serialization']
    sources = [(event['id'], dict(string_leaves(event['value'], ()))) for event in preceding]
    records = []
    for original in events:
        event = deepcopy(original)
        references = []
        for path, text in string_leaves(original['value'], ()):
            lines = text.splitlines(keepends=True)
            best, best_size = event, len(json.dumps(event, **serialization))
            for identity, leaves in sources:
                if path not in leaves:
                    continue
                previous = leaves[path].splitlines(keepends=True)
                prefix = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(lines, previous)))
                suffix = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1],
                    zip(reversed(lines[prefix:]), reversed(previous[prefix:]))))
                candidate = deepcopy(event)
                destination = ('value', *path)
                target = candidate
                for key in destination[:-1]:
                    target = target[key]
                target[destination[-1]] = ''.join(lines[prefix:len(lines) - suffix])
                candidate['text_prefixes'] = [*references, {
                    'path': list(path), 'event': identity, 'lines': prefix, 'trailing_lines': suffix,
                }]
                size = len(json.dumps(candidate, **serialization))
                if size < best_size:
                    best, best_size = candidate, size
            event = best
            references = event.get('text_prefixes', [])
        records.append(protocol['history_record'].format(event=json.dumps(event, **serialization)))
        sources.append((original['id'], dict(string_leaves(original['value'], ()))))
    return ''.join(records)


def decode_events(text):
    events = {}
    for line in text.splitlines():
        event = json.loads(line)
        for reference in event.pop('text_prefixes', []):
            destination = ('value', *reference['path'])
            previous = events[reference['event']]
            target = event
            for key in destination[:-1]:
                previous, target = previous[key], target[key]
            previous = previous[destination[-1]].splitlines(keepends=True)
            target[destination[-1]] = ''.join(previous[:reference['lines']]) + target[destination[-1]] + ''.join(
                previous[len(previous) - reference['trailing_lines']:])
        events[event['id']] = event
    return list(events.values())
