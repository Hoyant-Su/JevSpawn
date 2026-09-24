import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

import yaml


def analyze(source, capacity):
    records = json.loads(Path(source).read_text())
    finite = sorted((row for row in records if row.get('operation') == 'finite'),
                    key=lambda row: row['started_monotonic'])
    signatures, counts, lru = [], Counter(), OrderedDict()
    prefix_shapes, padded_shapes = Counter(), Counter()
    zero_groups, zero_batches, hits = 0, 0, 0
    for index, row in enumerate(finite):
        structured = row['structured']
        roots, ends = structured['root_prefix_tokens'], structured['prefix_tokens']
        assert len(roots) == len(ends) == len(structured['group_sizes']) == structured['root_batch_size']
        assert sum(structured['group_sizes']) == structured['physical_batch_size']
        extensions = [end - root for root, end in zip(roots, ends, strict=True)]
        assert all(length >= 0 for length in extensions)
        selected = [(root, extension) for root, extension in zip(roots, extensions, strict=True) if extension]
        zero_groups += len(roots) - len(selected)
        if not selected:
            zero_batches += 1
            continue
        prefix_lengths, suffix_lengths = tuple(zip(*selected, strict=True))
        signature = (prefix_lengths, suffix_lengths)
        counts[signature] += 1
        prefix_shapes[(len(selected), max(prefix_lengths))] += 1
        padded_shapes[(len(selected), max(prefix_lengths), max(suffix_lengths))] += 1
        hit = signature in lru
        hits += hit
        lru[signature] = None
        lru.move_to_end(signature)
        if len(lru) > capacity:
            lru.popitem(last=False)
        signatures.append({'finite_batch_index': index, 'root_lengths': prefix_lengths,
            'extension_lengths': suffix_lengths, 'group_sizes_before_zero_filter': structured['group_sizes'],
            'lru_hit': hit, 'state_extension_seconds': structured['timings']['state_extension_seconds']})
    calls, unique = len(signatures), len(counts)
    return {'source': source, 'finite_batches': len(finite), 'state_extension_calls': calls,
        'zero_extension_groups_excluded': zero_groups, 'all_zero_batches_excluded': zero_batches,
        'unique_exact_signatures': unique, 'unlimited_cache': {'captures': unique, 'replays': calls - unique},
        'dedicated_lru_cache': {'capacity': capacity, 'captures': calls - hits, 'replays': hits},
        'signature_use_counts': sorted(counts.values(), reverse=True),
        'prefix_shape_histogram': [{'batch_size': shape[0], 'prefix_width': shape[1], 'calls': count}
                                   for shape, count in sorted(prefix_shapes.items())],
        'padded_shape_histogram': [{'batch_size': shape[0], 'prefix_width': shape[1],
            'suffix_width': shape[2], 'calls': count} for shape, count in sorted(padded_shapes.items())],
        'calls': signatures}


def main(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    capacity = shared['runtime']['graph_cache_size']
    runs = [analyze(source, capacity) for source in settings['sources']]
    result = {'settings': settings,
        'signature': 'Ordered nonzero owner-level root lengths and prefix-minus-root extension lengths; these determine RaggedSuffix and RaggedCacheAttention metadata. Token values can change within the same signature.',
        'scope': 'Exact recorded state-extension cohorts, one invocation per finite batch with nonzero extensions. Runs have separate caches. Dedicated LRU simulation excludes competition from other graph types; it is an optimistic opportunity count, not measured CUDA graph reuse. Zero-length extensions execute no model work.',
        'runs': runs,
        'totals': {'calls': sum(row['state_extension_calls'] for row in runs),
            'unique_signatures_across_separate_runs': sum(row['unique_exact_signatures'] for row in runs),
            'unlimited_replays': sum(row['unlimited_cache']['replays'] for row in runs),
            'lru_replays': sum(row['dedicated_lru_cache']['replays'] for row in runs)}}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
