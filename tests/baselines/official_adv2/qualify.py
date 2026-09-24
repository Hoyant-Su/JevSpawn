import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
from threading import Lock
import time

import numpy as np
import torch

from baselines.adv2.component import IndicatorIndex, load_indicators
from baselines.official_adv2.adapter import configure, make_task, run_tasks
from baselines.official_adv2.embedding_service import EmbeddingService, SharedGenerationService
from jev_spawn.infra.backend import Backend


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ['native-config', 'config', 'tasks', 'output', 'pool']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task-count', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native, config = json.loads(args.native_config.read_text()), json.loads(args.config.read_text())
    native['run_dir'] = str(args.output)
    rows = [json.loads(line) for line in args.tasks.read_text().splitlines()][:args.task_count]
    assert len(rows) == args.task_count == native['batch_size']
    for key, value in config['environment'].items():
        assert os.environ[key] == value, f'{key} must be configured before importing upstream prompts.'
    configure(config)
    metrics = json.loads(args.pool.read_text())
    started = time.perf_counter()
    index = IndicatorIndex(load_indicators(args.pool), config['embedding'], 'cuda:0')
    assert index.vectors.shape[0] == len(metrics) and np.isfinite(index.vectors.numpy()).all()
    assert torch.all(torch.linalg.vector_norm(index.vectors, dim=1) > 0)
    index_seconds = time.perf_counter() - started
    backend = Backend(native)
    gpu_lock = Lock()
    embeddings = EmbeddingService(index, native['batch_size'], config['batch_wait_seconds'], gpu_lock)
    service = SharedGenerationService(backend, native['batch_size'], config['batch_wait_seconds'], gpu_lock)
    save(args.output / 'protocol.json', {'native': native, 'config': config,
        'task_ids': [row['task_id'] for row in rows], 'pool_count': len(metrics),
        'embedding_load_and_index_seconds': index_seconds, 'embedding_index_seconds': index.build_seconds})
    try:
        for stage in ['warmup', 'measured']:
            generation_start, embedding_start = len(service.records), len(embeddings.records)
            started = time.perf_counter()
            tasks = [make_task(row, service, embeddings, config, metrics) for row in rows]
            results = asyncio.run(run_tasks(tasks))
            elapsed = time.perf_counter() - started
            batches, retrieval = service.records[generation_start:], embeddings.records[embedding_start:]
            save(args.output / f'{stage}-results.json', results)
            save(args.output / f'{stage}-batches.json', batches)
            save(args.output / f'{stage}-embeddings.json', retrieval)
            intervals = sorted(value for batch in batches for row in batch['decode'] for value in row['inter_token_seconds'])
            summary = {'tasks': len(rows), 'valid_answers': sum(row['status'] == 'valid' for row in results),
                'errors': [dict(task_id=row['task_id'], error=row['error']) for row in results if row['status'] == 'error'],
                'elapsed_seconds': elapsed, 'actual_generation_batch_sizes': [batch['batch_size'] for batch in batches],
                'actual_embedding_batch_sizes': [batch['batch_size'] for batch in retrieval],
                'itl_median_ms': statistics.median(intervals) * 1000,
                'itl_max_ms': max(intervals) * 1000, 'intervals_over_100ms': sum(value >= .1 for value in intervals),
                'peak_allocated_gib': max(batch['peak_allocated_bytes'] for batch in batches) / 2**30}
            save(args.output / f'{stage}-summary.json', summary)
            print(json.dumps({'stage': stage, **summary}), flush=True)
    finally:
        embeddings.close()
        service.close()


if __name__ == '__main__':
    main()
