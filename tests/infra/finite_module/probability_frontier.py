import argparse
import json
from pathlib import Path

import torch

from jev_spawn.algo.probability_frontier import expand_frontier


@torch.inference_mode()
def run(settings):
    rows = []
    for path in sorted(Path(settings['source']).glob(settings['trace_glob'])):
        trace = json.loads(path.read_text())
        for batch in trace['fields']:
            if 'decisions' in batch:
                rows.extend({'trace': str(path), 'decision': decision} for decision in batch['decisions']
                            if decision['id'] in settings['field_ids'])
    assert rows
    logits = torch.tensor([row['decision']['option_logits'] for row in rows], device=settings['device']).unsqueeze(1)
    legal = torch.ones_like(logits, dtype=torch.bool)
    prior = torch.zeros_like(logits[..., 0])
    for _ in range(settings['warmup_steps']):
        expand_frontier(logits, legal, prior, settings['branch_budget'])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = expand_frontier(logits, legal, prior, settings['branch_budget'])
    graph.replay()
    torch.cuda.synchronize()
    actions = selected['actions'].cpu()
    expected = torch.tensor([row['decision']['probabilities'] for row in rows])
    torch.testing.assert_close(selected['log_probabilities'].exp().cpu(), expected.gather(-1, actions),
                               rtol=0, atol=settings['probability_atol'])
    assert selected['valid'].all().item()
    assert selected['parents'].eq(0).all().item()
    assert all(len(set(row)) == settings['branch_budget'] for row in actions.tolist())
    timings = []
    for _ in range(settings['timing_repetitions']):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'results.json').write_text(json.dumps({'config': settings, 'states': len(rows),
        'selected_paths_per_state': settings['branch_budget'], 'probabilities_match_recorded_model': True,
        'graph_replay_milliseconds': timings,
        'scope': 'GPU branch selection replay from real recorded model logits. No new model calls, branch executions or task-success claims.',
        'actions': actions.tolist(), 'parents': selected['parents'].tolist()}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
