import itertools
import json
import random
import time

import numpy as np
import torch


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class Observer:
    def __init__(self, output, dataset, service, batch_size, resume, configuration):
        self.output, self.dataset, self.service = output, dataset, service
        self.batch_size, self.resume = batch_size, resume
        self.configuration = configuration
        self.start_iteration = resume['next_iteration'] if resume is not None else 0
        self.initial_numpy = resume['initial_numpy'] if resume is not None else np.random.get_state()
        self.dataset.position = self.start_iteration * batch_size

    def loader(self, factory):
        np.random.set_state(self.initial_numpy)
        loader = factory()
        first = next(loader)
        loader = itertools.chain([first], loader)
        for _ in range(self.start_iteration * self.batch_size):
            next(loader)
        return loader

    def restore(self, optimizer):
        if self.resume is not None:
            optimizer.load_state_dict(self.resume['optimizer'])
            torch.set_rng_state(self.resume['torch_rng'])
            torch.cuda.set_rng_state_all(self.resume['cuda_rng'])
            np.random.set_state(self.resume['numpy_rng'])
            random.setstate(self.resume['python_rng'])

    def begin(self, iteration):
        self.graphs = []
        self.started = time.perf_counter()
        self.batch_start = len(self.service.records)
        self.episode_start = len(self.dataset.episodes)

    def check_execution(self):
        for graph in self.graphs:
            assert all(node.outputs for node in graph.nodes.values()), 'A required agent exhausted upstream retries.'
            assert graph.decision_node.outputs, 'The original final decision produced no output.'

    def end(self, state):
        graph, optimizer, iteration = state['graph'], state['optimizer'], state['i_iter']
        batches = self.service.records[self.batch_start:]
        row = {'iteration': iteration, 'wall_seconds': time.perf_counter() - self.started,
               'episodes': self.dataset.episodes[self.episode_start:],
               'raw_answers': state['raw_answers'], 'answers': state['answers'],
               'correct_answers': state['correct_answers'], 'rewards': state['utilities'],
               'loss': state['total_loss'].item(), 'log_probabilities': [x.item() for x in state['log_probs']],
               'spatial_logits': graph.spatial_logits.detach().tolist(),
               'temporal_logits': graph.temporal_logits.detach().tolist(),
               'spatial_gradients': graph.spatial_logits.grad.tolist(),
               'temporal_gradients': None if graph.temporal_logits.grad is None else graph.temporal_logits.grad.tolist(),
               'spatial_masks': graph.spatial_masks.tolist(), 'temporal_masks': graph.temporal_masks.tolist(),
               'realized_spatial_edges': [g.spatial_adj_matrix.tolist() for g in self.graphs]}
        save(self.output / f'iteration-{iteration:03d}.json', row)
        save(self.output / f'iteration-{iteration:03d}-batches.json', batches)
        checkpoint = {'next_iteration': iteration + 1, 'graph': graph,
                      'optimizer': optimizer.state_dict(), 'initial_numpy': self.initial_numpy,
                      'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                      'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                      'task_rows': self.dataset.rows, 'labels': self.dataset.labels,
                      'configuration': self.configuration}
        temporary = self.output / 'checkpoint.pending.pt'
        torch.save(checkpoint, temporary)
        temporary.replace(self.output / f'checkpoint-{iteration + 1:03d}.pt')
        print(json.dumps({'iteration': iteration + 1, 'reward': np.mean(state['utilities']),
                          'seconds': row['wall_seconds'], 'batches': len(batches)}), flush=True)
