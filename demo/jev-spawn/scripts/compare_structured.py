"""Compare matched completed worker runs using their recorded timings and evaluations."""

import argparse
import collections
import json
import statistics
from pathlib import Path


CONTROLS = ('model_path', 'dtype', 'attention', 'kernel', 'seed', 'cpu_threads', 'max_input_tokens',
            'batch_size', 'world_size', 'task_offset', 'task_count', 'warmup_batch', 'worker_role')


def load_run(directory):
    metadata = json.loads((directory / 'run.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    quality = json.loads((directory / 'quality.json').read_text())
    assert quality['format'] == 'workers'
    assert summary['method'] == metadata['method'] and set(quality['methods']) == {metadata['method']}
    rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    predictions = {row['task_id']: row for row in rows}
    assert len(rows) == len(predictions) == len(metadata['task_ids']) == summary['worker_count']
    assert set(predictions) == set(metadata['task_ids'])
    schemas = metadata['task_fields']
    assert schemas is not None and set(schemas) == set(metadata['task_ids'])
    assert all(list(predictions[task]['fields']) == list(schemas[task]) for task in metadata['task_ids'])
    decisions = sum(len(schema) for schema in schemas.values())
    metrics = quality['methods'][metadata['method']]
    assert decisions == summary['field_decisions'] == metrics['questions']
    assert quality['paragraphs'] == metrics['paragraphs'] == summary['worker_count']
    assert Path(metadata['config']['model_path']).name == 'Qwen3.5-4B'
    summary['model'] = 'Qwen/Qwen3.5-4B'
    summary['evaluation_status'] = 'evaluated'
    summary['median_worker_latency_seconds'] = statistics.median(row['execution_seconds'] for row in rows)
    summary['maximum_per_gpu_peak_allocated_bytes'] = max(gpu['peak_allocated_bytes'] for gpu in summary['cuda_memory']['per_gpu'])
    evaluation = {**metrics, 'scope': quality['scope'], 'metric_conventions': quality['metric_conventions'],
                  'baselines': quality['baselines']}
    return {'summary': summary, 'evaluation': evaluation, 'metadata': metadata, 'predictions': predictions}


def compare(reference, other):
    left, right = reference['summary'], other['summary']
    agreement = collections.Counter()
    transitions = collections.Counter()
    workers_changed = 0
    for task_id in reference['metadata']['task_ids']:
        baseline = reference['predictions'][task_id]['fields']
        candidate = other['predictions'][task_id]['fields']
        changed = False
        for name in baseline:
            a, b = baseline[name]['choice'], candidate[name]['choice']
            agreement['agree' if a == b else 'disagree'] += 1
            transitions[f'{a} -> {b}'] += 1
            changed |= a != b
        workers_changed += changed
    return {
        'reference_run_id': left['run_id'], 'other_run_id': right['run_id'],
        'dispatch_time_ratio_other_over_reference': right['elapsed_seconds'] / left['elapsed_seconds'],
        'current_process_time_ratio_other_over_reference': right['current_process_end_to_end_seconds'] / left['current_process_end_to_end_seconds'],
        'max_per_gpu_peak_ratio_other_over_reference': right['maximum_per_gpu_peak_allocated_bytes'] / left['maximum_per_gpu_peak_allocated_bytes'],
        'question_accuracy_difference_other_minus_reference': other['evaluation']['question_accuracy'] - reference['evaluation']['question_accuracy'],
        'balanced_accuracy_difference_other_minus_reference': other['evaluation']['balanced_accuracy'] - reference['evaluation']['balanced_accuracy'],
        'paired_predictions': {'decisions': sum(agreement.values()), 'agreements': agreement['agree'],
                               'disagreements': agreement['disagree'],
                               'disagreement_rate': agreement['disagree'] / sum(agreement.values()),
                               'workers_with_any_disagreement': workers_changed,
                               'choice_transitions_reference_to_other': dict(transitions)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.runs) >= 2
    runs = [load_run(directory) for directory in args.runs]
    reference = runs[0]
    assert len({run['summary']['run_id'] for run in runs}) == len(runs)
    for run in runs[1:]:
        left, right = reference['metadata'], run['metadata']
        assert left['task_ids'] == right['task_ids'], 'Task order or selection differs.'
        assert left['task_fields'] == right['task_fields'], 'Question text, descriptions, or options differ.'
        assert all(list(left['task_fields'][task]) == list(right['task_fields'][task]) for task in left['task_ids'])
        assert all(left['config'][key] == right['config'][key] for key in CONTROLS), 'Model or runtime controls differ.'
    controls = {key: reference['metadata']['config'][key] for key in CONTROLS if key != 'model_path'}
    controls['model'] = 'Qwen/Qwen3.5-4B'
    result = {
        'runs': [{'summary': run['summary'], 'evaluation': run['evaluation']} for run in runs],
        'reference_run_id': reference['summary']['run_id'],
        'comparisons': [compare(reference, run) for run in runs[1:]],
        'matched_controls': controls,
        'interpretation': 'Matched task IDs, question schemas, model and runtime controls; one measured pass per run. '
                          'Time ratios are other/reference and are descriptive, without repeated-run uncertainty estimates. '
                          'Dispatch-to-completion time includes queue and host work; ranks load and warm before their own dispatch, '
                          'so loading on other ranks can overlap. Current-process timing includes loading and warmup separately '
                          'from the dispatch interval and excludes Python import time and earlier preempted processes. '
                          'Memory lists each GPU allocator peak and the maximum per-GPU peak; peaks are not summed into an instantaneous system value. '
                          'Quality is binary answerability accuracy and balanced accuracy, not answer-span QA EM/F1. '
                          'Joint autoregressive JSON and isolated field readouts can choose different answers. '
                          'Observed speed differences do not establish equal quality; paired disagreements are not themselves errors.',
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(json.dumps({'runs': len(runs), 'reference_run_id': result['reference_run_id'],
                      'comparisons': result['comparisons']}, indent=2))


if __name__ == '__main__':
    main()
