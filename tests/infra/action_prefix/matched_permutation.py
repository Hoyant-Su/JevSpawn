import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.matched_spawn import measure, seed_roots
from tests.infra.action_prefix.replay import request_for
from tests.infra.action_prefix.runtime import ActionPrefixTail


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = SimpleNamespace(shared=shared, backend=backend)
    config = json.loads(Path(settings['source_config']).read_text())
    source = Path(config['run_output'])
    protocol = json.loads((source / 'protocol.json').read_text())
    inference = protocol['inference']['settings']
    requests = []
    for file in settings['task_files']:
        trace = json.loads((source / file).read_text())
        turn = trace['trace']['rounds'][settings['turn']]
        records = [record for records in turn['parent_computations'].values() for record in records if 'fields' in record]
        assert len(records) == settings['fields_per_turn']
        field, = records[settings['field_index']]['requests']
        requests.append(request_for(field, field['action_messages'], trace['task_id'], service, settings))
    classes = {'reuse': ActionPrefixTail, 'recompute': StableFiniteGraphTail}
    report = {'settings': settings, 'startup': startup,
        'task_ids': [request.task_id for request in requests],
        'input_lengths': [len(request.admitted.tokens) for request in requests],
        'option_ids': [[option['id'] for option in request.field['options']] for request in requests],
        'measurements': [], 'comparisons': []}
    scores = {}
    for mode in settings['modes']:
        tail = classes[mode](backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                             inference['state_copy'], inference['graph_shape'])
        roots = PrefixCache(shared.runtime.root_batch_size)
        cold, _ = measure(backend, tail, roots, [requests], settings)
        for repeat in range(settings['repetitions']):
            for order in settings['orders']:
                ordered = [requests[index] for index in order['indices']]
                roots.clear()
                tail.prefix_cache.clear()
                root = seed_roots(backend, roots, [ordered])
                result, logits = measure(backend, tail, roots, [ordered], settings)
                assert all(batch['work']['graph_captures'] == 0 for batch in result['batches'])
                inverse = [order['indices'].index(index) for index in range(len(requests))]
                restored = logits[0][inverse].clone()
                scores[mode, repeat, order['name']] = restored
                report['measurements'].append({'mode': mode, 'repetition': repeat, 'order': order,
                    'cold': cold, 'root_initialization': root, 'warm': result,
                    'restored_logits': restored.tolist()})
        tail.graphs.clear()
        tail.prefix_cache.clear()
        roots.clear()
    for comparison in settings['comparisons']:
        for repeat in range(settings['repetitions']):
            left = scores[comparison['left_mode'], repeat, comparison['left_order']].float()
            right = scores[comparison['right_mode'], repeat, comparison['right_order']].float()
            report['comparisons'].append({'comparison': comparison, 'repetition': repeat,
                'max_absolute_error': (left - right).abs().amax(-1).tolist(),
                'argmax_equal': left.argmax(-1).eq(right.argmax(-1)).tolist(),
                'ranking_equal': left.argsort(dim=-1, descending=True, stable=True).eq(
                    right.argsort(dim=-1, descending=True, stable=True)).all(-1).tolist()})
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
