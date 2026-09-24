import importlib.util
import json
from pathlib import Path

import torch

from jev_spawn.infra.backend import Backend
from jev_spawn.algo.structured import score_fields
from baselines.common.tasks import read, rows


def main():
    config = read('configs/native.json')
    output = Path('runs/batched-readout-parity-001')
    output.mkdir(exist_ok=False)
    backend = Backend(config)
    source = Path('../../runtime/research_v1/profile_20260920/structured_before.py')
    spec = importlib.util.spec_from_file_location('previous_structured', source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    tasks = rows('../../data/research_v1/squad2_answerability/evaluation/tasks.jsonl')
    task = max(tasks, key=lambda row: len(row['fields']))
    args = (backend, [task['state']], task['fields'], 'streamed')
    previous.score_fields(*args)
    score_fields(*args)
    measurements = []
    for repeat in range(3):
        before, after = previous.score_fields(*args), score_fields(*args)
        for key in task['fields']:
            assert before['fields'][key]['choices'] == after['fields'][key]['choices']
            for metric in ['probabilities', 'option_logits']:
                torch.testing.assert_close(torch.tensor(before['fields'][key][metric]),
                                           torch.tensor(after['fields'][key][metric]), rtol=0, atol=0)
        row = dict(repeat=repeat, before_seconds=before['elapsed_seconds'], after_seconds=after['elapsed_seconds'],
                   before_readout_seconds=before['timings']['readout_seconds'],
                   after_readout_seconds=after['timings']['readout_seconds'])
        measurements.append(row)
        print(json.dumps(row), flush=True)
    (output / 'result.json').write_text(json.dumps(dict(task_id=task['task_id'], fields=len(task['fields']),
        exact_parity=True, config=config, measurements=measurements), indent=2) + '\n')


if __name__ == '__main__':
    main()
