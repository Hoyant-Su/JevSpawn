import argparse
from collections import Counter
import json
from pathlib import Path
import time

from test_declaration_builder import builder, ROOT, SETTINGS
from test_singleton_folding import action_family, previous_builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    task = json.loads((ROOT / args.source).read_text())
    rows = []
    started = time.perf_counter()
    for turn in task['trace']['rounds']:
        revision = turn.get('revision', {})
        if revision.get('observation', {}).get('error') != 'Declaration field capacity exhausted.':
            continue
        signature, = [output['text'] for call in revision['calls'] if call['kind'] == 'signature'
                      for output in call['outputs']]
        old = previous_builder(max_fields=SETTINGS['max_variants'])
        new = builder()
        before = old.compile_signature(signature, ['action', 'arguments', 'action'])
        after = new.compile_signature(signature, ['action', 'arguments', 'action'])
        old_family, new_family = action_family(old, before), action_family(new, after)
        assert old_family == new_family, {'turn': turn['turn'], 'missing': sorted(old_family - new_family),
                                         'extra': sorted(new_family - old_family)}
        rows.append({'turn': turn['turn'], 'old_fields': len(old.program['fields']),
                     'new_fields': len(new.program['fields']), 'native_action_count': len(old_family),
                     'exact_native_action_family_equal': True})
    report = {'stage': '053', 'source': args.source, 'task_id': task['task_id'],
              'reference_builder': 'tests/runtime/schema/declaration_builder_stage052.py',
              'reference_field_limit': SETTINGS['max_variants'], 'production_field_limit': SETTINGS['max_fields'],
              'method': 'Enumerate every reachable conditional field assignment and compare complete runtime-materialized native action sets using type-preserving JSON encoding.',
              'replayed_rejections': len(rows), 'all_exact': all(r['exact_native_action_family_equal'] for r in rows),
              'field_count_pairs': {str(k): v for k, v in Counter((r['old_fields'], r['new_fields']) for r in rows).items()},
              'elapsed_seconds': time.perf_counter() - started, 'rows': rows,
              'scope': 'Compiler action-set preservation, not model choice/probability preservation or native success.'}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
