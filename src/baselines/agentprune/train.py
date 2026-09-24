import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from baselines.agentprune.adapter import Dataset, REQUESTS, configure
from baselines.agentprune.checkpoint import Observer, save
from baselines.agentprune.upstream import graph_kwargs, load_core, training_function
from baselines.official.model_service import GenerationService
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt


def main():
    parser = argparse.ArgumentParser()
    for name in ['source', 'native-config', 'config', 'prompts', 'tasks', 'labels', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    attempt = str(time.time_ns())
    config, native = json.loads(args.config.read_text()), json.loads(args.native_config.read_text())
    assert native['batch_size'] == config['batch_size'] == config['training']['batch_size']
    assert native['seed'] == config['seed']
    native['run_dir'] = str(args.output)
    graph_class = load_core(args.source)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    started = time.perf_counter()
    backend = Backend(native)
    service = GenerationService(backend, config['batch_size'], config['batch_wait_seconds'])
    configure(service, config['generation'], load_prompt(args.prompts))
    dataset = Dataset(args.tasks, args.labels, config['batch_size'])
    assert len(dataset) == config['task_count']
    resume = torch.load(args.resume, weights_only=False) if args.resume is not None else None
    configuration = {'config': config, 'native': native}
    if resume is None:
        graph = graph_class(domain=config['domain'], llm_name=native['model_path'],
                            agent_names=config['agent_names'], decision_method=config['decision_method'],
                            optimized_spatial=config['optimized_spatial'],
                            optimized_temporal=config['optimized_temporal'],
                            **graph_kwargs(args.source)(config['mode'], len(config['agent_names'])))
    else:
        assert resume['task_rows'] == dataset.rows and resume['labels'] == dataset.labels
        assert resume['configuration'] == configuration
        graph = resume['graph']
    setup_seconds = time.perf_counter() - started
    observer = Observer(args.output, dataset, service, config['batch_size'], resume, configuration)
    train, transformed_source = training_function(args.source, observer)
    (args.output / 'executed_train.py').write_text(transformed_source + '\n')
    save(args.output / f'protocol-{attempt}.json', {'config': config, 'native': native,
         'source': str(args.source), 'tasks': str(args.tasks), 'labels': str(args.labels),
         'task_ids': [r['task_id'] for r in dataset.rows], 'setup_seconds': setup_seconds,
         'resume': str(args.resume) if args.resume is not None else None,
         'roles': [node.role for node in graph.nodes.values()],
         'node_ids': list(graph.nodes), 'start_iteration': observer.start_iteration})
    started = time.perf_counter()
    try:
        asyncio.run(train(graph=graph, dataset=dataset, **config['training']))
        save(args.output / 'completion.json', {'training_wall_seconds': time.perf_counter() - started,
             'completed_iterations': config['training']['num_iters'],
             'start_iteration': observer.start_iteration,
             'model_calls': sum(row['batch_size'] for row in service.records),
             'actual_batch_sizes': [row['batch_size'] for row in service.records]})
    finally:
        service.close()
        save(args.output / f'generation-batches-{attempt}.json', service.records)
        save(args.output / f'provider-requests-{attempt}.json', REQUESTS)


if __name__ == '__main__':
    main()
