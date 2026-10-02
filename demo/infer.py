import argparse
import json
import os
from pathlib import Path
from threading import Lock

import yaml

from demo.environments.factory import create, evaluate
from demo.trace import compact_round
from jev_spawn.api import run


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--samples', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    shared = yaml.safe_load((ROOT / settings['shared_config']).read_text())
    identities = settings['samples'] if args.samples == ['all'] else args.samples
    samples = [json.loads((ROOT / settings['data'] / (identity + '.json')).read_text()) for identity in identities]
    by_task = {sample['task_id']: sample for sample in samples}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = Lock()

    def emit(task_id, event):
        with lock, (args.output / (by_task[task_id]['id'] + '.jsonl')).open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def on_turn(task_id, turn):
        emit(task_id, {'type': 'round', 'round': compact_round(turn)})

    def environment(task):
        sample = by_task[task['task_id']]
        return create(sample, args.output / sample['id'], shared['runtime']['sample_timeout_seconds'])

    def commit(index, result):
        sample = samples[index]
        (args.output / (sample['id'] + '.json')).write_text(json.dumps(result, ensure_ascii=False) + '\n')
        if result['status'] != 'completed' or result['answer'] is None:
            emit(sample['task_id'], {'type': 'error', 'message': result.get('error', result.get('termination', result['status']))})
            return
        score = evaluate(sample, result['answer'])
        emit(sample['task_id'], {'type': 'result', 'answer': result['answer'], 'score': score,
             'success': None if sample['success'] is None else score == 1,
             'selected_terminal': result['selected_terminal']})

    if int(os.environ['RANK']) == 0:
        for sample in samples:
            emit(sample['task_id'], {'type': 'start', 'mode': 'live', 'sample': {key: sample[key] for key in (
                'id', 'title', 'dataset', 'task_id', 'context', 'score_label')}})
    run([{'task_id': sample['task_id'], 'context': sample['context']} for sample in samples],
        environment, commit, shared_config=ROOT / settings['shared_config'],
        method_config=ROOT / settings['method_config'], inference_config=ROOT / settings['inference_config'],
        parallel_config=ROOT / settings['parallel_config'], on_turn=on_turn)


if __name__ == '__main__':
    main()
