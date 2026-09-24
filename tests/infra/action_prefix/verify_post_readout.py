import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers.cache_utils import StaticLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix import post_readout
from tests.infra.action_prefix.matched_spawn import measure, requests_for, seed_roots
from tests.infra.action_prefix.runtime import ActionPrefixTail


def tensors(state):
    result = {}
    for index, layer in enumerate(state.layers):
        for name in ('keys', 'values'):
            if hasattr(layer, name):
                result[index, name] = getattr(layer, name)
        for name in ('conv_states', 'recurrent_states'):
            if hasattr(layer, name):
                result.update({(index, name, key): value for key, value in getattr(layer, name).items()})
    return result


class SnapshotAudit:
    def __init__(self):
        self.original = post_readout.snapshot_prefixes
        self.snapshots = []
        self.replay = 0

    def capture(self, decoder, sequences, indices, settings):
        rows = self.original(decoder, sequences, indices, settings)
        stop = max(map(len, sequences))
        assert int(decoder.cache.get_seq_length()) == stop
        for row, index in zip(rows, indices, strict=True):
            assert row.get_seq_length() == len(sequences[index])
            expected = {}
            for layer_index, source in enumerate(decoder.cache.layers):
                if type(source) is StaticLayer:
                    start = stop - len(sequences[index])
                    expected.update({(layer_index, name): getattr(source, name)[index:index + 1, :, start:stop, :]
                                     for name in ('keys', 'values')})
                else:
                    expected.update({(layer_index, name, key): value[index:index + 1]
                                     for name in ('conv_states', 'recurrent_states')
                                     for key, value in getattr(source, name).items()})
                    assert row.layers[layer_index].has_previous_state == source.has_previous_state
            saved = tensors(row)
            assert saved.keys() == expected.keys()
            source_storage = {tensor.untyped_storage().data_ptr() for tensor in expected.values()}
            assert all(tensor.untyped_storage().data_ptr() not in source_storage for tensor in saved.values())
            assert len({tensor.untyped_storage().data_ptr() for tensor in saved.values()}) == len(saved)
            assert all(torch.equal(saved[key], value) for key, value in expected.items())
            previous_storage = {tensor.untyped_storage().data_ptr() for record in self.snapshots
                                for tensor in record['tensors'].values()}
            assert all(tensor.untyped_storage().data_ptr() not in previous_storage for tensor in saved.values())
            self.snapshots.append({'decoder': decoder, 'created_at_replay': self.replay, 'tensors': saved,
                'expected': {key: tensor.clone() for key, tensor in saved.items()}, 'later_replays_checked': 0,
                'token_length': len(sequences[index]), 'tensor_count': len(saved),
                'layer_count': len(row.layers), 'source_row': index})
        return rows

    def check(self, decoder):
        for record in self.snapshots:
            assert all(torch.equal(tensor, record['expected'][key]) for key, tensor in record['tensors'].items())
            record['later_replays_checked'] += int(record['decoder'] is decoder and self.replay > record['created_at_replay'])

    def report(self, count):
        records = self.snapshots[:count]
        assert all(record['later_replays_checked'] > 0 for record in records)
        return [{key: value for key, value in record.items() if key not in ('decoder', 'tensors', 'expected')}
                for record in records]


def rankings(logits, batches, width):
    rows = []
    for values, requests in zip(logits, batches, strict=True):
        for value, request in zip(values, requests, strict=True):
            options = [option['id'] for option in request.field['options']]
            scores = value[:len(options)].float()
            order = scores.argsort(dim=-1, descending=True, stable=True).tolist()
            ranked = [options[index] for index in order]
            rows.append({'task_id': request.task_id, 'field_id': request.field['id'],
                'option_ids': options, 'logits': scores.tolist(), 'ranked_option_ids': ranked,
                'topk_option_ids': ranked[:width], 'branch_width': width})
    return rows


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    prepared = json.loads(Path(settings['prepared']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'workloads': [],
        'scope': 'Correctness-only checks, excluded from all speed measurements. Identical recorded field requests in both modes. '
            'Additional already-recorded batches replay each used graph solely to verify snapshot immutability.'}
    classes = {'reuse': ActionPrefixTail, 'post_readout': post_readout.PostReadoutTail}
    for workload in prepared['workloads']:
        protocol = json.loads(Path(workload['protocol']).read_text())
        inference = protocol['inference']['settings']
        width = protocol['method']['settings']['rollout']['branch_width']
        batches = requests_for(workload, backend, shared, settings)
        mode_rankings, snapshots = {}, []
        for mode in settings['modes']:
            tail = classes[mode](backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                                 inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            measure(backend, tail, roots, batches, settings)
            roots.clear()
            tail.prefix_cache.clear()
            seed_roots(backend, roots, batches)
            logits, representatives = [], {}
            audit = SnapshotAudit()
            with patch.object(post_readout, 'snapshot_prefixes', audit.capture):
                for requests in batches:
                    audit.replay += 1
                    tail.score(requests, [len(request.root_tokens) for request in requests], roots)
                    logits.append(tail.last_logits.clone())
                    key = tail.last_layout['key']
                    representatives[key] = requests
                    audit.check(tail.graphs[key])
                count = len(audit.snapshots)
                for probe in range(settings['snapshot_probe_replays']):
                    for key, requests in representatives.items():
                        audit.replay += 1
                        tail.score(requests, [len(request.root_tokens) for request in requests], roots)
                        audit.check(tail.graphs[key])
            snapshots.extend(audit.report(count))
            mode_rankings[mode] = rankings(logits, batches, width)
            tail.graphs.clear()
            tail.prefix_cache.clear()
            roots.clear()
        comparisons = []
        for reference, candidate in zip(mode_rankings['reuse'], mode_rankings['post_readout'], strict=True):
            assert reference['option_ids'] == candidate['option_ids']
            comparisons.append({'field_id': reference['field_id'],
                'argmax_equal': reference['ranked_option_ids'][0] == candidate['ranked_option_ids'][0],
                'full_ranking_equal': reference['ranked_option_ids'] == candidate['ranked_option_ids'],
                'topk_order_equal': reference['topk_option_ids'] == candidate['topk_option_ids'],
                'topk_set_equal': set(reference['topk_option_ids']) == set(candidate['topk_option_ids']),
                'max_absolute_logit_error': max(abs(left - right) for left, right in
                    zip(reference['logits'], candidate['logits'], strict=True))})
        report['workloads'].append({'track': workload['track'], 'task_id': workload['task_id'], 'turn': workload['turn'],
            'rankings': mode_rankings, 'comparisons': comparisons, 'snapshots': snapshots,
            'snapshot_equality': True, 'snapshot_storage_independent': True, 'snapshot_immutable_after_replay': True})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'], 'verified': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
