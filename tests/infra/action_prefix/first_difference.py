import argparse
from functools import reduce
from importlib import import_module
import json
from operator import getitem
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.runtime_contract import RuntimeContract
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER
from tests.infra.action_prefix.matched_spawn import measure, seed_roots
from tests.infra.action_prefix.replay import messages_for, request_for


def prepare_requests(settings, shared, backend):
    protocol = json.loads(Path(settings['protocol']).read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    service = RuntimeContract()
    service.shared, service.backend, service.execution_metadata = shared, backend, {}
    service.configure_runtime_contract(protocol['method']['settings'])
    requests = []
    for source in settings['sources']:
        traces = [json.loads(Path(path).read_text()) for path in source]
        fields = [reduce(getitem, settings['request_path'], trace) for trace in traces]
        assert fields.count(fields[0]) == len(fields)
        field = fields[0]
        requests.append(request_for(field, messages_for(field, service),
                                    traces[0]['task_id'], service, settings))
    recorded = json.loads(Path(settings['recorded_batch']['path']).read_text())[
        settings['recorded_batch']['index']]
    for request in requests:
        row = recorded['task_ids'].index(request.task_id)
        assert len(request.admitted.tokens) == recorded['input_tokens'][row]
        root_length = recorded['structured']['root_prefix_tokens'][row]
        request.root_tokens = request.admitted.tokens[:root_length]
    return requests, protocol['inference']['settings']


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    requests, inference = prepare_requests(settings, shared, backend)
    report = {'settings': settings, 'startup': startup, 'measurements': [], 'comparisons': []}
    scores = {}
    for mode, definition in settings['implementations'].items():
        cls = getattr(import_module(definition['module']), definition['class'])
        tail = cls(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                   inference['state_copy'], inference['graph_shape'])
        roots = PrefixCache(shared.runtime.root_batch_size)
        measure(backend, tail, roots, [requests], settings)
        for repeat in range(settings['repetitions']):
            for order in settings['orders']:
                roots.clear()
                tail.prefix_cache.clear()
                root_work = seed_roots(backend, roots, [requests])
                ordered = [requests[index] for index in order['indices']]
                result, logits = measure(backend, tail, roots, [ordered], settings)
                inverse = [order['indices'].index(index) for index in range(len(requests))]
                restored = logits[0][inverse].clone()
                scores[mode, repeat, order['name']] = restored
                report['measurements'].append({'mode': mode, 'repeat': repeat,
                    'order': order, 'root_work': root_work, 'work': result,
                    'restored_logits': restored.tolist()})
        tail.graphs.clear()
        tail.prefix_cache.clear()
        roots.clear()
    for comparison in settings['comparisons']:
        for repeat in range(settings['repetitions']):
            left = scores[comparison['left_mode'], repeat, comparison['left_order']]
            right = scores[comparison['right_mode'], repeat, comparison['right_order']]
            for index, request in enumerate(requests):
                count = len(request.field['options'])
                a, b = left[index, :count].float(), right[index, :count].float()
                report['comparisons'].append({'comparison': comparison, 'repeat': repeat,
                    'task_id': request.task_id, 'max_absolute_error': (a-b).abs().max().item(),
                    'argmax_equal': a.argmax().item() == b.argmax().item(),
                    'ranking_equal': torch.equal(a.argsort(descending=True, stable=True),
                                                b.argsort(descending=True, stable=True))})
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(
        json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
