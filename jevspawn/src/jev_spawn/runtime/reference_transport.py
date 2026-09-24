from collections import Counter
from copy import deepcopy
import json


def pack_references(value, settings):
    """Intern repeated JSON values without interpreting original user keys."""
    counts = Counter()
    identities = {}
    entries = []
    ref = settings['reference_key']
    escaped = settings['object_key']

    def identity(item):
        return json.dumps(item, **settings['serialization'])

    def visit(item):
        counts.update([identity(item)])
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    def encode(item):
        key = identity(item)
        repeated = isinstance(item, (str, dict, list)) and counts[key] >= settings['minimum_occurrences']
        if repeated and key in identities:
            return {ref: identities[key]}
        if isinstance(item, dict):
            pairs = [[name, encode(child)] for name, child in item.items()]
            encoded = {escaped: pairs} if ref in item or escaped in item else dict(pairs)
        elif isinstance(item, list):
            encoded = [encode(child) for child in item]
        else:
            encoded = item
        if repeated:
            identities[key] = len(entries)
            entries.append(encoded)
            return {ref: identities[key]}
        return encoded

    visit(value)
    root = encode(value)
    return {settings['dictionary_key']: entries, settings['root_key']: root}


def unpack_references(packed, settings):
    entries = []
    ref = settings['reference_key']
    escaped = settings['object_key']

    def decode(item):
        if isinstance(item, dict):
            if ref in item:
                return deepcopy(entries[item[ref]])
            if escaped in item:
                return {name: decode(child) for name, child in item[escaped]}
            return {name: decode(child) for name, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    for entry in packed[settings['dictionary_key']]:
        entries.append(decode(entry))
    return decode(packed[settings['root_key']])
