import argparse
import json
from pathlib import Path
import re

import jsonschema

from baselines.common.evaluate import evaluate_run, save


def declaration_check(rounds, schema):
    revisions = [{'turn': record['turn'], **record['revision']['observation']}
                 for record in rounds if 'revision' in record]
    accepted = [record['declaration'] for record in revisions if record['accepted']]
    for declaration in accepted:
        jsonschema.validate(declaration, schema)
    return {'valid': bool(accepted), 'revisions': revisions,
            'accepted': len(accepted), 'rejected': len(revisions) - len(accepted)}


def first_failure(log):
    if not log.exists():
        return {'source': str(log), 'available': False}
    for number, line in enumerate(log.read_text().splitlines(), start=1):
        if re.search(r'\b\w*(?:Error|Exception):', line):
            return {'source': str(log), 'line': number, 'text': line,
                    'scope': 'Run-level failure; task attribution requires the task record or trace.'}
    return {'source': str(log), 'available': True, 'error_line': None}


def report(run, log):
    protocol_path = run / 'protocol.json'
    if not protocol_path.exists():
        return {'run': str(run), 'available': False, 'missing_stage': 'protocol',
                'first_logged_failure': first_failure(log)}
    protocol = json.loads(protocol_path.read_text())
    evaluation = evaluate_run(run)
    schema = protocol['method']['settings']['rollout']['declaration_schema']
    batches = [(path, index, batch) for path in sorted(run.glob('session-*/batches.json'))
               for index, batch in enumerate(json.loads(path.read_text()))]
    scores = {score['task_id']: score for score in evaluation['scores']}
    tasks = []
    for index, task in enumerate(protocol['tasks']):
        identity = task['task_id']
        contexts = list(sorted(run.glob(f'session-*/context-{index:05d}.json')))
        context = json.loads(contexts[0].read_text())['context'] if contexts else None
        first = next(((path, batch_index, row, batch) for path, batch_index, batch in batches
                      for row, task_id in enumerate(batch['task_ids'])
                      if task_id == identity and 'output_texts' in batch), None)
        completion = None
        if first is not None:
            path, batch_index, row, batch = first
            completion = {'source': str(path), 'batch_index': batch_index, 'row': row,
                          'messages': batch['messages'][row],
                          'output': batch['output_texts'][row],
                          'input_tokens': batch['input_tokens'][row],
                          'output_tokens': batch['output_tokens'][row]}
        result_path = run / f'task-{index:05d}.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        trace = result.get('trace', {})
        rounds = trace.get('rounds', [])
        structural = declaration_check(rounds, schema)
        executions = []
        for round_record in rounds:
            for child, records in round_record.get('children', {}).items():
                for record_index, record in enumerate(records):
                    if 'actions' in record:
                        current = [event for event in record['observations'] if event['id'] == child]
                        if not current:
                            continue
                        executions.append({'turn': round_record['turn'], 'child': child,
                            'record_index': record_index, 'actions': record['actions'],
                            'observations': current,
                            'tool_batch_seconds': record['tool_batch_seconds']})
        observed = [event for execution in executions for event in execution['observations']]
        score = scores[identity]
        missing_stage = ('context' if context is None else
                         'declaration_trace' if not structural['revisions'] else
                         'declaration_validation' if not structural['valid'] else
                         'action_execution_trace' if not executions else
                         'native_observation' if not observed else
                         'final_answer' if score['answer'] is None else
                         'final_answer_validation' if not score['valid_answer'] else None)
        tasks.append({'task_id': identity,
            'context': {'source': str(contexts[0]) if contexts else None, 'text': context},
            'first_model_prefill_and_completion': completion,
            'initial_context_present_verbatim_in_first_prefill': None if completion is None or context is None
                else any(context in message['content'] for message in completion['messages']),
            'declaration': {'structure': structural,
                'task_semantic_validity': 'Not inferred from JSON validity or tool acceptance.'},
            'control_rounds': len(rounds),
            'spawn_rounds': sum(bool(record.get('children')) for record in rounds),
            'revision_and_spawn_turns': [record['turn'] for record in rounds
                if 'revision' in record and record.get('children')],
            'action_execution_observed': bool(executions), 'native_observation_recorded': bool(observed),
            'recorded_branch_executions': len(executions),
            'first_execution': executions[0] if executions else None,
            'execution_source': str(result_path) if result else None,
            'native_evaluation': score, 'first_missing_stage': missing_stage,
            'evidence_limit': 'Absent trace is missing evidence, not proof that no computation occurred.'})
    return {'run': str(run), 'protocol_source': str(protocol_path),
            'run_completed': (run / 'completion.json').exists(),
            'summary': evaluation['summary'], 'tasks': tasks,
            'first_logged_failure': first_failure(log)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--log', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [report(run, log) for run, log in zip(args.run, args.log, strict=True)]
    save(args.output, {'runs': reports})
    print(json.dumps([{'run': row['run'], 'tasks': [
        {'task_id': task['task_id'], 'missing_stage': task['first_missing_stage'],
         'structure_valid': task['declaration']['structure']['valid'],
         'execution_observed': task['action_execution_observed'],
         'score': task['native_evaluation']['score']} for task in row.get('tasks', [])]}
        for row in reports], indent=2))


if __name__ == '__main__':
    main()
