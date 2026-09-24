import argparse
import json
from pathlib import Path

import torch

from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    torch.manual_seed(settings['seed'])
    tail = object.__new__(StableFiniteGraphTail)
    tail.buckets = tuple(settings['buckets'])
    tail.capacity = settings['capacity']
    tail.graph_ids = tuple(settings['native_ids'])
    records = []
    for active in range(min(tail.buckets), max(tail.buckets) + 1):
        sequences = [list(range(length)) for length in settings['lengths'][:active]]
        states = [object() for sequence in sequences]
        tokens = [sequence[-1] for sequence in sequences]
        for count in settings['candidate_counts']:
            ids = tail.graph_ids[:count]
            key, physical, capacity, rows = tail.graph_layout(sequences, ids)
            assert key == physical and capacity == tail.capacity and rows == tail.graph_ids
            loaded, last = tail.graph_inputs(states, tokens, physical)
            assert len(loaded) == len(last) == physical
            assert all(value is original for value, original in zip(loaded[:active], states, strict=True))
            assert last[:active] == tokens
            assert all(value is states[-1] for value in loaded[active:])
            assert all(value == tokens[-1] for value in last[active:])
            logits = torch.randn(physical, len(rows))
            result = tail.graph_outputs(logits, active, ids)
            assert result.shape == (active, count) and torch.equal(result, logits[:active, :count])
            records.append({'active_rows':active,'physical_rows':physical,'capacity':capacity,'candidate_rows':count,'key':key})
    assert set(record['key'] for record in records) == set(tail.buckets)
    Path(settings['output']).write_text(json.dumps({'scope':'CPU layout, genuine-row padding identity and candidate slicing only; no model numerical qualification.', 'cases':records,'passed':True},indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(records),'output':settings['output']}))


if __name__ == '__main__':
    main()
