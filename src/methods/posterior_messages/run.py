import argparse
import json
from pathlib import Path
import time

import torch

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.evidence_interfaces.inputs import read, write
from methods.generated_schema.run import measured_call
from methods.posterior_messages.representation import groups
from methods.program_execution.grouped import score_grouped
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def query(backend, row, arm, fixed, prompts):
    started = time.perf_counter()
    calls, scores = [], []
    documents = row['documents']
    size = fixed['batch_size']
    for start in range(0, len(documents), size):
        batch = groups(row, documents[start:start + size], arm, prompts, fixed['identity_mode'])
        record = measured_call(backend, lambda: score_grouped(backend, batch, 'tiled_independent'))
        result = record['result']
        assert len(result['groups']) == len(batch)
        for original, answer in zip(batch, result['groups']):
            assert len(answer) == 1 and answer[0]['id'] == original[0]['id']
            value = answer[0]
            assert value['input_tokens'] <= fixed['max_input_tokens']
            scores.append(value['probabilities'][value['option_ids'].index('yes')])
        calls.append(record)
    assert len(scores) == len(documents)
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    result = {'task_id': row['task_id'], 'arm': arm, 'elapsed_seconds': time.perf_counter() - started,
              'document_ids': [document['document_id'] for document in documents],
              'probabilities': scores, 'ranking': [documents[index]['document_id'] for index in order],
              'receiver_calls': len(scores), 'batch_sizes': [call['result']['root_batch_size'] for call in calls],
              'input_tokens': [group[0]['input_tokens'] for call in calls for group in call['result']['groups']],
              'peak_allocated_bytes': max(call['peak_allocated_bytes'] for call in calls),
              'peak_reserved_bytes': max(call['peak_reserved_bytes'] for call in calls),
              'generated_tokens': 0}
    return result, calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    fixed = stage['fixed']
    rows = [json.loads(line) for line in Path(stage['inputs']).read_text().splitlines()]
    assert len(rows) == fixed['queries'] and fixed['repetitions'] == 1
    assert all(len(row['documents']) == fixed['documents_per_query'] for row in rows)
    snapshot = {'fixed': fixed, 'arms': stage['arms'], 'inputs': stage['inputs'],
                'native_config': resolve_prompts(read(stage['native_config'])), 'prompts': load_prompt(stage['prompts']),
                'task_ids': [row['task_id'] for row in rows]}
    if args.resume:
        assert read(args.output / 'protocol.json') == snapshot
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'protocol.json', snapshot)
    session = args.output / f'session-{len(list(args.output.glob("session-*"))):04d}'
    session.mkdir()
    native = snapshot['native_config']
    native.update({key: fixed[key] for key in ['model_path', 'dtype', 'seed', 'batch_size',
                                              'branch_batch_size', 'max_input_tokens']})
    native.update(run_id='posterior-messages-development', run_dir=str(session), tasks_path=stage['inputs'])
    CONTROLLER['option_template'] = snapshot['prompts']['option_template']
    start = time.perf_counter()
    backend = Backend(native)
    torch.cuda.synchronize(backend.device)
    write(session / 'backend.json', {'load_seconds': time.perf_counter() - start,
                                   'metadata': backend.metadata, 'config': native})
    for phase in ['warmup', 'measured']:
        for index, row in enumerate(rows):
            offset = index % len(stage['arms'])
            order = stage['arms'][offset:] + stage['arms'][:offset]
            for arm in order:
                base = session if phase == 'warmup' else args.output
                directory = base / phase / arm / f'{index:03d}'
                directory.mkdir(parents=True, exist_ok=True)
                if phase == 'measured' and (directory / 'outcome.json').exists():
                    assert read(directory / 'outcome.json')['task_id'] == row['task_id']
                    continue
                outcome, calls = query(backend, row, arm, fixed, snapshot['prompts'])
                outcome['session'] = str(session)
                write(directory / 'calls.json', calls)
                write(directory / 'outcome.json', outcome)
                print(json.dumps({'phase': phase, 'index': index, 'arm': arm,
                                  'seconds': outcome['elapsed_seconds'],
                                  'receivers': outcome['receiver_calls']}), flush=True)
    write(args.output / 'completed.json', {'task_ids': snapshot['task_ids'], 'arms': stage['arms']})


if __name__ == '__main__':
    main()
