import argparse
from collections import Counter
import json
from pathlib import Path
import re


def presentation_answer(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r'\s*(?P<fence>```json[ \t]*\n)?(?P<object>\{.*\})(?(fence)\s*\n```)\s*',
                         text, re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group('object'))['answer']
    except (ValueError, TypeError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run / 'protocol.json').read_text())
    phase = args.run / 'measured'
    results = [json.loads((phase / (task_id.replace('/', '_') + '.json')).read_text())
               for task_id in protocol['task_ids']]
    batches = json.loads((phase / 'batches.json').read_text())
    metrics = json.loads((phase / 'summary.json').read_text())
    quality = json.loads((args.run / 'development_quality.json').read_text())
    labels = {row['task_id']: row['labels']['q0'] for row in map(
        json.loads, Path(protocol['settings']['labels']).read_text().splitlines())}
    rows = {row['task_id']: row for row in map(
        json.loads, Path(protocol['settings']['tasks']).read_text().splitlines())}
    presented = {r['task_id']: presentation_answer(r.get('final_output')) for r in results}
    calls = [call for result in results for call in result.get('calls', [])]
    operators = Counter(action['instruction_type'] for result in results
                        for stage in result.get('state', {}).get('stages', {}).values()
                        for action in stage.get('history', []))
    report = {
        'scope': 'Eight AQuA development roots after full workload warmup. No formal evaluation.',
        'quality': quality, 'measured': metrics,
        'presentation_grammar_quality': {
            'grammar': 'A JSON object, optionally enclosed in one Markdown json fence. No surrounding explanation.',
            'correct': sum(presented[r['task_id']] == labels[r['task_id']] for r in results),
            'valid': sum(presented[r['task_id']] in [o['id'] for o in rows[r['task_id']]['fields']['q0']['options']]
                         for r in results),
            'total': len(results),
            'strict_parser_results_preserved': True,
        },
        'warmup': json.loads((args.run / 'warmup/summary.json').read_text()),
        'calls_by_role': dict(Counter(call['role'] for call in calls)),
        'executed_operators': dict(operators),
        'generation_token_limits': dict(Counter(str(call['max_tokens']) for call in calls)),
        'generation_temperatures': dict(Counter(str(call['temperature']) for call in calls)),
        'truncated_calls': sum(sum(batch['truncated']) for batch in batches),
        'roots': [{'task_id': r['task_id'], 'answer': r.get('answer'),
                   'presentation_grammar_answer': presented[r['task_id']],
                   'model_calls': r.get('model_calls'),
                   'stages': len(r.get('state', {}).get('stages', {})),
                   'parse_error': r.get('parse_error'),
                   'workflow_errors': r.get('state', {}).get('error_log', [])}
                  for r in results],
        'promotion': {
            'actual_method_trace_present': all(r.get('calls') and r.get('state') for r in results),
            'model_call_accounting_matches': len(calls) == metrics['model_calls'],
            'all_measured_itl_under_100ms': metrics['max_itl_ms'] < 100,
            'all_final_answers_valid': quality['valid'] == quality['total'],
            'formal_evaluation_authorized': False,
        },
    }
    (args.run / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['quality', 'measured', 'calls_by_role',
                                                'truncated_calls', 'promotion']}, indent=2))


if __name__ == '__main__':
    main()
