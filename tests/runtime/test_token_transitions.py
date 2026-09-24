import argparse
import json
from pathlib import Path

import torch

from jev_spawn.algo.token_paths import compile_token_paths
from jev_spawn.runtime.token_transitions import TokenTransitions


def successors(table, state):
    return [(column, target) for column, target, valid in zip(
        table['edge_token_indices'][state], table['edge_next_states'][state],
        table['edge_valid'][state], strict=True) if valid]


def initial_states(table, prefixes):
    result = []
    for prefix in prefixes:
        state = table['root']
        for token in prefix:
            state = {table['token_ids'][column]: target
                     for column, target in successors(table, state)}[token]
        result.append(state)
    return result


def reference_step(table, states, outputs, scores, emissions):
    for lane, state in enumerate(states):
        if table['terminal'][state]:
            continue
        column, target = max(successors(table, state), key=lambda edge: scores[lane][edge[0]])
        outputs[lane] = table['token_ids'][column]
        states[lane] = target
        emissions[lane].append(outputs[lane])
    return [table['terminal'][state] for state in states]


def main(config):
    torch.manual_seed(config['seed'])
    table = compile_token_paths(config['paths'])
    states = initial_states(table, config['initial_prefixes'])
    batch_size = len(states)
    transition = TokenTransitions(table, batch_size, config['device'], config)
    initial = torch.tensor(states, device=config['device'], dtype=torch.long)
    logits = torch.full((batch_size, len(table['token_ids'])), config['base_score'],
                        device=config['device'], dtype=torch.float32)
    scores = []
    for step in config['steps']:
        assert len(step) == batch_size
        scores.append([[row.get(str(token), config['base_score']) for token in table['token_ids']]
                       for row in step])
    device_scores = [torch.tensor(step, device=config['device'], dtype=logits.dtype) for step in scores]

    stream = torch.cuda.Stream(device=config['device'])
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(config['warmup_steps']):
            transition.state.copy_(initial)
            transition.output.fill_(config['initial_output'])
            transition.advance(logits)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transition.advance(logits)

    def verify(advance):
        transition.state.copy_(initial)
        transition.done.copy_(transition.terminal[initial])
        transition.output.fill_(config['initial_output'])
        expected_states = list(states)
        expected_outputs = [config['initial_output']] * batch_size
        emissions = [[] for _ in states]
        snapshots = []
        for score, device_score in zip(scores, device_scores, strict=True):
            logits.copy_(device_score)
            advance()
            done = reference_step(table, expected_states, expected_outputs, score, emissions)
            assert transition.state.tolist() == expected_states
            assert transition.output.tolist() == expected_outputs
            assert transition.done.tolist() == done
            snapshots.append({'states': list(expected_states), 'outputs': list(expected_outputs), 'done': done})
        assert emissions == config['expected_emissions']
        assert all(done)
        return snapshots

    eager = verify(lambda: transition.advance(logits))
    captured = verify(graph.replay)
    assert eager == captured
    report = {'scope': config['scope'], 'seed': config['seed'], 'batch_size': batch_size,
              'steps': len(scores), 'eager_passed': True, 'cuda_graph_replay_passed': True,
              'expected_emissions': config['expected_emissions'], 'snapshots': captured}
    output = Path(config['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
