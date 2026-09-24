import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import time

from baselines.agentprune.adapter import REQUESTS, choice_ids, configure
from baselines.agentprune.evaluation import EvaluationDataset, evaluate_checkpoint, load_checkpoint
from baselines.agentprune.upstream import load_core
from baselines.formal_choices.run import metrics, save
from baselines.official.model_service import GenerationService
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def block_run(graph, service, tasks, source_path, settings, directory, seed, offset, split):
    identifiers = [row['task_id'] for row in tasks]
    completed = directory / 'complete.json'
    if completed.exists():
        result = read(completed)
        assert result['task_ids'] == identifiers and result['seed'] == seed
        return result
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f'attempt-{len(list(directory.glob("attempt-*"))):04d}'
    attempt.mkdir()
    dataset = EvaluationDataset(Path(source_path), settings['batch_size'], split)
    dataset.rows = tasks
    dataset.position = offset
    result = asyncio.run(evaluate_checkpoint(
        graph, dataset, service, settings['source'], attempt,
        num_rounds=settings['num_rounds'], batch_size=settings['batch_size'], seed=seed))
    assert [row['task_id'] for row in result['records']] == identifiers
    record = {'task_ids': identifiers, 'seed': seed, 'attempt': attempt.name,
              'result': result, 'metrics': metrics(service.records, result['wall_seconds'])}
    save(attempt / 'provider-requests.json', REQUESTS)
    save(attempt / 'input-failures.json', service.input_failures)
    save(completed, record)
    service.records.clear()
    service.input_failures.clear()
    REQUESTS.clear()
    print(json.dumps({'directory': str(directory), 'tasks': len(tasks), **record['metrics']}), flush=True)
    return record


def run(settings, output):
    tasks = rows(settings['tasks'])
    development = {row['task_id']: row for row in rows(settings['warmup_tasks'])}
    warmup = [development[task_id] for task_id in settings['warmup_task_ids']]
    assert len(tasks) == settings['task_count'] and len(warmup) == settings['warmup_task_count']
    assert len({row['task_id'] for row in tasks}) == len(tasks)
    assert not {row['task_id'] for row in tasks} & {row['task_id'] for row in warmup}
    options = {row['task_id']: choice_ids(row) for row in tasks + warmup}
    native, training = read(settings['native_config']), read(settings['training_config'])
    assert native['batch_size'] == training['batch_size'] == settings['batch_size'] == 8
    assert native['max_input_tokens'] == settings['max_input_tokens'] == 8192
    assert training['training']['num_rounds'] == settings['num_rounds'] == 1
    assert training['generation'] == settings['generation']
    prompts = load_prompt(settings['prompts'])
    protocol = {'settings': settings, 'native': native, 'training_config': training,
                'prompts': prompts, 'tasks': tasks, 'warmup_tasks': warmup}
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / 'protocol.json'
        if protocol_path.exists():
            assert read(protocol_path) == protocol
        else:
            save(protocol_path, protocol)
        blocks = [tasks[start:start + settings['batch_size']]
                  for start in range(0, len(tasks), settings['batch_size'])]
        pending = []
        for index, block in enumerate(blocks):
            path = output / f'block-{index:04d}' / 'complete.json'
            if path.exists():
                saved = read(path)
                assert saved['task_ids'] == [row['task_id'] for row in block]
                assert saved['seed'] == native['seed'] + index + 1
            else:
                pending.append((index, block))
        if pending:
            load_core(settings['source'])
            graph = load_checkpoint(Path(settings['checkpoint']), training)
            frozen = {name: getattr(graph, name).detach().tolist()
                      for name in ['spatial_logits', 'temporal_logits', 'spatial_masks', 'temporal_masks']}
            session = output / f'session-{len(list(output.glob("session-*"))):04d}'
            session.mkdir()
            started = time.perf_counter()
            backend = Backend({**native, 'run_dir': str(output)})
            service = GenerationService(backend, settings['batch_size'], settings['batch_wait_seconds'])
            configure(service, settings['generation'], prompts)
            save(session / 'backend.json', {'metadata': backend.metadata,
                 'load_seconds': time.perf_counter() - started, 'frozen_graph': frozen,
                 'node_ids': list(graph.nodes), 'pending_blocks': [index for index, _ in pending]})
            try:
                block_run(graph, service, warmup, settings['warmup_tasks'], settings,
                          session / 'warmup', native['seed'], 0, 'development')
                for index, block in pending:
                    block_run(graph, service, block, settings['tasks'], settings,
                              output / f'block-{index:04d}', native['seed'] + index + 1,
                              index * settings['batch_size'], 'test')
                    assert all(getattr(graph, name).detach().tolist() == value for name, value in frozen.items())
            finally:
                save(session / 'remaining-provider-requests.json', REQUESTS)
                save(session / 'remaining-input-failures.json', service.input_failures)
                service.close()
        completed = [read(output / f'block-{index:04d}' / 'complete.json') for index in range(len(blocks))]
        results = [row for block in completed for row in block['result']['records']]
        assert [row['task_id'] for row in results] == [row['task_id'] for row in tasks]
        save(output / 'completion.json', {'tasks': len(results), 'blocks': len(completed),
             'model_calls': sum(block['metrics']['model_calls'] for block in completed),
             'elapsed_seconds': sum(block['metrics']['elapsed_seconds'] for block in completed)})
        labels = {row['task_id']: row['labels']['q0'] for row in rows(settings['labels'])}
        assert set(labels) == {row['task_id'] for row in tasks}
        summary = {'tasks': len(results), 'correct': sum(row['answer'] == labels[row['task_id']] for row in results),
                   'valid': sum(row['answer'] in options[row['task_id']] for row in results),
                   'block_metrics': [block['metrics'] for block in completed],
                   'training_provenance': settings['training_provenance'],
                   'scope': settings['stage'] + '. Frozen development topology, original task choices, and offline held-out scoring.'}
        summary['accuracy'] = summary['correct'] / len(results)
        for key in ['elapsed_seconds', 'model_calls', 'output_tokens', 'truncated_calls', 'intervals_over_100ms']:
            summary[key] = sum(block['metrics'][key] for block in completed)
        for key in ['peak_allocated_bytes', 'max_itl_ms']:
            summary[key] = max(block['metrics'][key] for block in completed)
        save(output / 'evaluation.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(read(args.settings), args.output)


if __name__ == '__main__':
    main()
