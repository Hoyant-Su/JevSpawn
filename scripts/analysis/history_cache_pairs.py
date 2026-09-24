import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text())


def generated_sequences(directory):
    sequences = defaultdict(list)
    for file in sorted(Path(directory).glob('session-*/batches.json')):
        for batch in read(file):
            if 'messages' not in batch:
                continue
            for identity, text, tokens in zip(batch['task_ids'], batch['output_texts'],
                                              batch['output_token_ids'], strict=True):
                sequences[identity].append((text, tokens))
    return sequences


def execution_sequences(directory):
    sequences = {}
    for file in sorted(Path(directory).glob('task-*.json')):
        record = read(file)
        sequences[record['task_id']] = [{
            'turn': row['turn'],
            'selected': row.get('selected'),
            'selected_operation': row.get('selected_operation'),
            'fields': {parent: [step['fields'] for step in steps if 'fields' in step]
                       for parent, steps in row.get('parent_computations', {}).items()},
            'children': {identity: [{key: step[key] for key in
                                    ('actions', 'observations', 'selected_values') if key in step}
                                   for step in steps]
                         for identity, steps in row.get('children', {}).items()},
        } for row in record['trace']['rounds']]
    return sequences


def collect_arm(path, evaluations, history_metrics, finite_metrics):
    settings = read(path)
    directory = Path(settings['run_output'])
    specification = read(settings['specification'])
    evaluation = evaluations.get(str(directory.resolve()))
    result = {'configuration': path, 'run': str(directory),
              'declared_tasks': specification['task_count'],
              'saved_tasks': len(list(directory.glob('task-*.json'))),
              'evaluated': evaluation is not None}
    if evaluation is None:
        return result
    result['evaluation'] = evaluation['summary']
    result['samples'] = evaluation['scores']
    trajectories = {}
    for file in sorted(directory.glob('task-*.json')):
        record = read(file)
        rounds = record['trace']['rounds']
        decision_times = [row['frontier_decision']['ready_monotonic'] for row in rounds
                          if 'frontier_decision' in row]
        trajectories[record['task_id']] = {
            'recorded_rounds': len(rounds),
            'before_first_decision_seconds': min(decision_times) - record['admitted_monotonic']
                                             if decision_times else None,
            'operations': dict(Counter(row['operation_decision']['choice'] for row in rounds
                                       if 'operation_decision' in row)),
        }
    result['trajectories'] = trajectories
    batches = [batch for file in sorted(directory.glob('session-*/batches.json'))
               for batch in read(file)]
    finite = [batch for batch in batches if batch.get('operation') == 'finite']
    result['finite_calls'] = len(finite)
    result['finite_rows'] = sum(batch['batch_size'] for batch in finite)
    result['finite_wall_seconds'] = sum(batch['elapsed_seconds'] for batch in finite)
    result['additional_finite_metrics'] = {
        key: sum(batch['structured'][key] for batch in finite) for key in finite_metrics}
    result['generation_wall_seconds'] = sum(batch['elapsed_seconds'] for batch in batches
                                             if 'messages' in batch)
    result['generation_output_tokens'] = sum(sum(batch['output_tokens']) for batch in batches
                                              if 'messages' in batch)
    result['tokens'] = {key: sum(batch['structured'][key] for batch in finite)
                       for key in ('computed_input_tokens', 'padded_input_tokens', 'logical_input_tokens')}
    phases = Counter()
    for batch in finite:
        phases.update(batch['structured']['unfenced_host_timings'])
    result['unfenced_host_phases'] = dict(phases)
    if history_metrics:
        result['history_hit_rows'] = sum(batch['structured']['history_hit_rows'] for batch in finite)
        result['history_eligible_rows'] = sum(batch['structured']['history_eligible_rows'] for batch in finite)
    return result


def compare(reference, candidate):
    protocols = [read(Path(arm['run']) / 'protocol.json') for arm in (reference, candidate)]
    assert all(protocol == protocols[0] for protocol in protocols)
    contexts = [{record['task_id']: record['context']
                 for file in Path(arm['run']).glob('session-*/context-*.json')
                 for record in [read(file)]} for arm in (reference, candidate)]
    assert all(context == contexts[0] for context in contexts)
    left = {row['task_id']: row for row in reference['samples']}
    right = {row['task_id']: row for row in candidate['samples']}
    assert left.keys() == right.keys()
    assert len(left) == reference['declared_tasks'] == candidate['declared_tasks']
    assert left.keys() == contexts[0].keys()
    generations = [generated_sequences(arm['run']) for arm in (reference, candidate)]
    executions = [execution_sequences(arm['run']) for arm in (reference, candidate)]
    samples = [{'task_id': key, 'reference_correct': left[key]['correct'],
                'candidate_correct': right[key]['correct'],
                'reference_status': left[key]['status'], 'candidate_status': right[key]['status'],
                'reference_seconds': left[key]['elapsed_seconds'],
                'candidate_seconds': right[key]['elapsed_seconds'],
                'generated_sequence_equal': generations[0][key] == generations[1][key],
                'reference_generation_calls': len(generations[0][key]),
                'candidate_generation_calls': len(generations[1][key]),
                'execution_sequence_equal': executions[0][key] == executions[1][key],
                'first_execution_difference': next(
                    (index for index, (a, b) in enumerate(zip(executions[0][key], executions[1][key]))
                     if a != b), min(len(executions[0][key]), len(executions[1][key]))
                    if len(executions[0][key]) != len(executions[1][key]) else None),
                'reference_trajectory': reference['trajectories'][key],
                'candidate_trajectory': candidate['trajectories'][key]}
               for key in left]
    correct = [row for row in samples if row['reference_correct'] and row['candidate_correct']]
    accuracy = [arm['evaluation']['accuracy'] for arm in (reference, candidate)]
    return {'recorded_protocol_equal': True, 'recorded_contexts_equal': True,
            'generated_sequence_equal_count': sum(row['generated_sequence_equal'] for row in samples),
            'execution_sequence_equal_count': sum(row['execution_sequence_equal'] for row in samples),
            'accuracy_delta': candidate['evaluation']['accuracy'] - reference['evaluation']['accuracy']
            if all(value is not None for value in accuracy) else None,
            'mean_native_score_delta': candidate['evaluation']['mean_native_score']
                                       - reference['evaluation']['mean_native_score'],
            'all_sample_wall_ratio': mean(row['reference_seconds'] for row in samples)
                                     / mean(row['candidate_seconds'] for row in samples),
            'both_correct_count': len(correct),
            'both_correct_wall_ratio': (mean(row['reference_seconds'] for row in correct)
                                       / mean(row['candidate_seconds'] for row in correct)) if correct else None,
            'paired_samples': samples}


def summarize(configuration):
    evaluations = {}
    for pattern in configuration['evaluation_globs']:
        for path in sorted(Path().glob(pattern)):
            for run in read(path)['runs']:
                evaluations[str(Path(run['run']).resolve())] = run
    pairs = list(configuration['completed_stage_pairs'])
    pairs.extend({'track': job['track'], 'configs': job['configs']}
                 for job in read(configuration['active_stage'])['jobs'])
    tracks = {}
    for pair in pairs:
        configs = {read(path)[configuration['arm_selector']]: path for path in pair['configs']}
        reference = read(configs[False])
        candidate = read(configs[True])
        assert {key: value for key, value in reference.items() if key not in configuration['arm_fields']} == {
            key: value for key, value in candidate.items() if key not in configuration['arm_fields']}
        arms = {'reference': collect_arm(configs[False], evaluations, False,
                                        configuration['finite_metrics']['reference']),
                'candidate': collect_arm(configs[True], evaluations, configuration['history_metrics'],
                                        configuration['finite_metrics']['candidate'])}
        result = {'arms': arms}
        if all(arm['evaluated'] and arm['saved_tasks'] == arm['declared_tasks'] for arm in arms.values()):
            result['paired'] = compare(arms['reference'], arms['candidate'])
        tracks[pair['track']] = result
    output = {'notes': configuration['notes'], 'tracks': tracks}
    destination = Path(configuration['output'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({track: result['paired'] | {'paired_samples': 'saved in output'}
                      for track, result in tracks.items() if 'paired' in result}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    summarize(read(parser.parse_args().config))
