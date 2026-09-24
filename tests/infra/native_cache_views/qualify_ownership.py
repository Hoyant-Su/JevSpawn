import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from tests.infra.native_cache_views.qualify_read_only import ReadOnlyTail
from tests.infra.native_direct_finite import qualify as paired
from tests.infra.native_suffix_graph.qualify import tensors


OBSERVATIONS = []


def retained_bytes(states):
    storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
               for cache in states for tensor in tensors(cache)}
    return sum(storage.values())


class OwnershipObservation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = None
        OBSERVATIONS.append(self)

    def score(self, *args, **kwargs):
        result = super().score(*args, **kwargs)
        if self.saved is None:
            self.rank = dist.get_rank()
            self.saved = list(self.prefix_cache.entries.values())
            self.snapshot = [tensor.clone() for cache in self.saved for tensor in tensors(cache)]
        return result

    def report(self):
        values = [tensor for cache in self.saved for tensor in tensors(cache)]
        return {'implementation': type(self).__name__, 'rank': self.rank,
                'retained_rows': len(self.saved),
                'all_rows_storage_bytes': retained_bytes(self.saved),
                'per_row_storage_bytes': [retained_bytes([cache]) for cache in self.saved],
                'logical_tensor_bytes': sum(t.numel() * t.element_size() for t in values),
                'original_rows_unchanged_after_repeated_forwards': all(
                    torch.equal(a, b) for a, b in zip(self.snapshot, values, strict=True))}


class ObservedClone(OwnershipObservation, StableFiniteGraphTail):
    pass


class ObservedViews(OwnershipObservation, ReadOnlyTail):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    paired.StableFiniteGraphTail = ObservedClone
    paired.DirectFiniteTail = ObservedViews
    paired.run(settings)
    reports = [item.report() for item in OBSERVATIONS]
    Path(settings['output'], f'ownership-rank-{reports[0]["rank"]}.json').write_text(
        json.dumps(reports, indent=2) + '\n')
    assert all(row['original_rows_unchanged_after_repeated_forwards'] for row in reports)
