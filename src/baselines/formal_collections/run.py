import argparse
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from baselines.formal_choices.run import save
from baselines.official.phases import run_phase
from data.evaluate_bright import ranking_metrics
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def execute_block(service, collections, settings, prompts, directory, seed):
    task_ids = [row['task_id'] for row in collections]
    completed = directory / 'complete.json'
    if completed.exists():
        result = read(completed)
        assert result['task_ids'] == task_ids and result['seed'] == seed
        return result
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f'attempt-{len(list(directory.glob("attempt-*"))):04d}'
    attempt.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.save({'seed': seed, 'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, attempt / 'rng-start.pt')
    started = time.perf_counter()
    service.agent_tokenizers = {row['task_id']: copy.deepcopy(service.backend.tokenizer) for row in collections}
    preparation_seconds = time.perf_counter() - started
    results = run_phase(service, collections, settings, prompts, attempt, 'execution')
    whole_seconds = time.perf_counter() - started
    assert [row['task_id'] for row in results] == task_ids
    summary = read(attempt / 'execution/summary.json')
    summary['truncated_calls'] = sum(sum(row['truncated']) for row in service.records)
    summary['input_rejections'] = len(service.input_failures)
    assert all(set(row['task_ids']) <= set(task_ids) and row['batch_size'] <= settings['batch_size']
               for row in service.records)
    save(attempt / 'input-failures.json', service.input_failures)
    result = {'task_ids': task_ids, 'seed': seed, 'attempt': attempt.name, 'results': results,
              'metrics': summary, 'preparation_seconds': preparation_seconds,
              'whole_block_seconds': whole_seconds}
    save(completed, result)
    service.records.clear()
    service.input_failures.clear()
    print(json.dumps({'directory': str(directory), 'tasks': len(task_ids), 'metrics': summary}), flush=True)
    return result


def run(settings, output):
    collections, warmup = rows(settings['collections']), rows(settings['warmup_collections'])
    assert len(collections) == settings['task_count'] == 104
    assert len(warmup) == settings['warmup_task_count'] == 8
    assert all(len(row['candidates']) == settings['candidate_count'] == 128 for row in collections)
    assert all(len(row['candidates']) == settings['warmup_candidate_count'] == 32 for row in warmup)
    assert len({row['task_id'] for row in collections}) == len(collections)
    assert not {row['task_id'] for row in collections} & {row['task_id'] for row in warmup}
    original = read(settings['qualified_source_protocol'])['settings']
    workload = {'stage', 'collections', 'formal_collections', 'relevance', 'task_count', 'candidate_count'}
    assert all(settings[key] == value for key, value in original.items() if key not in workload)
    native, prompts = read(settings['native_config']), load_prompt(settings['prompts'])
    assert native['batch_size'] == settings['batch_size'] == 8
    assert native['max_input_tokens'] == settings['context_length'] == 8192
    protocol = {'settings': settings, 'native': native, 'prompts': prompts,
                'collections': collections, 'warmup_collections': warmup,
                'source_interface': read(settings['source_interface'])}
    output.mkdir(parents=True, exist_ok=True)
    os.environ['EVALTASK'] = 'bright_pony'
    blocks = [collections[start:start + settings['batch_size']]
              for start in range(0, len(collections), settings['batch_size'])]
    with (output / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / 'protocol.json'
        if protocol_path.exists():
            assert read(protocol_path) == protocol
        else:
            save(protocol_path, protocol)
        pending = []
        for index, block in enumerate(blocks):
            path = output / f'block-{index:04d}' / 'complete.json'
            if path.exists():
                result = read(path)
                assert result['task_ids'] == [row['task_id'] for row in block]
                assert result['seed'] == native['seed'] + index + 1
            else:
                pending.append((index, block))
        if pending:
            session = output / f'session-{len(list(output.glob("session-*"))):04d}'
            session.mkdir()
            started = time.perf_counter()
            backend = Backend({**native, 'run_dir': str(output)})
            service_class = getattr(importlib.import_module(settings['service_module']), settings['service_class'])
            service = service_class(backend, settings['batch_size'], settings['batch_wait_seconds'])
            save(session / 'backend.json', {'metadata': backend.metadata,
                 'load_seconds': time.perf_counter() - started,
                 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                 'pending_blocks': [index for index, _ in pending]})
            try:
                execute_block(service, warmup, settings, prompts, session / 'warmup', native['seed'])
                for index, block in pending:
                    execute_block(service, block, settings, prompts, output / f'block-{index:04d}',
                                  native['seed'] + index + 1)
            finally:
                save(session / 'remaining-input-failures.json', service.input_failures)
                service.close()
        completed = [read(output / f'block-{index:04d}' / 'complete.json') for index in range(len(blocks))]
        results = [row for block in completed for row in block['results']]
        assert [row['task_id'] for row in results] == [row['task_id'] for row in collections]
        save(output / 'completion.json', {'tasks': len(results), 'blocks': len(completed)})
        relevance = {row['task_id']: row['relevant_document_ids'] for row in rows(settings['relevance'])}
        assert set(relevance) == {row['task_id'] for row in collections}
        scores = []
        for collection, result in zip(collections, results):
            ranking = result.get('ranked_document_ids')
            candidates = {row['document_id'] for row in collection['candidates']}
            valid = (result['status'] == 'completed' and isinstance(ranking, list)
                     and len(ranking) == len(set(ranking)) == settings['ranking_cutoff']
                     and set(ranking) <= candidates)
            quality = ranking_metrics(ranking, relevance[result['task_id']], settings['ranking_cutoff']) if valid else {'ndcg': 0, 'recall': 0}
            scores.append({'task_id': result['task_id'], 'valid': valid, **quality})
        summary = {'queries': len(scores), 'valid': sum(row['valid'] for row in scores),
                   'ndcg': sum(row['ndcg'] for row in scores) / len(scores),
                   'recall': sum(row['recall'] for row in scores) / len(scores),
                   'per_query': scores, 'block_metrics': [block['metrics'] for block in completed],
                   'whole_block_seconds': sum(block['whole_block_seconds'] for block in completed),
                   'scope': 'All 104 heldout queries, including failed, capped and absent rankings. Original qualified algorithm budgets and top 10 contract are unchanged.'}
        for key in ['elapsed_seconds', 'model_calls', 'output_tokens', 'truncated_calls', 'input_rejections', 'intervals_over_100ms']:
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
