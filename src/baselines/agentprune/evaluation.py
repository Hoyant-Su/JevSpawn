import copy
import importlib
import json
import random
import time

import numpy as np
import torch

from baselines.agentprune.adapter import Dataset
from baselines.agentprune.checkpoint import Observer, save
from baselines.agentprune.upstream import evaluation_function


class EvaluationObserver(Observer):
    def __init__(self, output, dataset, service):
        self.output, self.dataset, self.service = output, dataset, service
        self.results = []

    def end_evaluation(self, state):
        records, answers = state['record_batch'], state['raw_answers']
        rows = [{'task_id': record['task_id'], 'raw_answer': answer,
                 'answer': self.dataset.postprocess_answer(answer),
                 'spatial_edges': graph.spatial_adj_matrix.tolist()}
                for record, answer, graph in zip(records, answers, self.graphs)]
        self.results.extend(rows)
        save(self.output / f"batch-{state['i_batch']:03d}.json", {
             'wall_seconds': time.perf_counter() - self.started,
             'records': rows, 'generation_batches': self.service.records[self.batch_start:]})


class EvaluationDataset(Dataset):
    def __init__(self, tasks, batch_size, split):
        self.rows = [json.loads(line) for line in tasks.read_text().splitlines()]
        self.batch_size, self.position, self.split = batch_size, 0, split
        self.episodes = []
        self.original = importlib.import_module('dataset.mmlu_dataset').MMLUDataset


def load_checkpoint(path, configuration):
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['configuration']['config'] == configuration
    graph = checkpoint['graph']
    manifest = json.loads((path.parent / f"iteration-{checkpoint['next_iteration'] - 1:03d}.json").read_text())
    for name in ['spatial_logits', 'temporal_logits', 'spatial_masks', 'temporal_masks']:
        assert getattr(graph, name).detach().tolist() == manifest[name]
    node_ids = list(graph.nodes)
    protocols = [json.loads(p.read_text()) for p in path.parent.glob('protocol-*.json')]
    assert protocols and all(row['node_ids'] == node_ids for row in protocols)
    expected_edges = [[a, b] for a in node_ids for b in node_ids]
    assert graph.potential_spatial_edges == graph.potential_temporal_edges == expected_edges
    return graph


async def evaluate_checkpoint(graph, dataset, service, source, output, num_rounds, batch_size, seed):
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.save({'seed': seed, 'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, output / 'rng-start.pt')
    observer = EvaluationObserver(output, dataset, service)
    evaluate, executed = evaluation_function(source, observer)
    (output / 'executed_evaluate.py').write_text(executed + '\n')
    graph_copy = copy.deepcopy(graph)
    started = time.perf_counter()
    batch_start = len(service.records)
    try:
        await evaluate(graph_copy, dataset, num_rounds=num_rounds,
                       limit_questions=len(dataset), eval_batch_size=batch_size)
    except Exception as error:
        save(output / 'failure.json', {'error_type': type(error).__name__, 'error': str(error),
             'wall_seconds': time.perf_counter() - started, 'completed_records': observer.results})
        raise
    finally:
        save(output / 'generation-batches.json', service.records[batch_start:])
    result = {'wall_seconds': time.perf_counter() - started,
              'task_count': len(dataset), 'records': observer.results}
    save(output / 'result.json', result)
    return result


def score_result(result, labels_path):
    labels = {row['task_id']: row['labels']['q0'] for row in
              map(json.loads, labels_path.read_text().splitlines())}
    accuracy = importlib.import_module('experiments.accuracy').Accuracy()
    for row in result['records']:
        accuracy.update(row['answer'], labels[row['task_id']])
    return {'task_count': result['task_count'], 'accuracy': accuracy.get()}
