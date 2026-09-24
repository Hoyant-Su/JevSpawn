import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.kernel_tuning import load_tuning, save_tuning
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER
from methods.program_execution.grouped import score_grouped
from replay_recorded_finite import ObservedPrefixCache, prepare


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    templates = [json.loads((Path(workload['source_run']) / 'protocol.json').read_text())['prompts']['option_template']
                 for workload in config['workloads']]
    template, = set(templates)
    CONTROLLER['option_template'] = template
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    settings = json.loads(Path(config['tuning_settings']).read_text())
    load_tuning(Path(config['cache_directory']) / config['cache_file'].format(rank=dist.get_rank()), backend, settings)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    cache = ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
    banks = {key: ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands) for key in config['caches']}
    tail = StableFiniteGraphTail(backend, shared.runtime, cache, config['state_copy'], config['graph_shape'])

    @torch.inference_mode()
    def compute(payload):
        cache.clear()
        for bank in banks.values():
            bank.clear()
        requests, lengths = payload['cohort']
        shapes = []
        hook = backend.model.model.register_forward_pre_hook(
            lambda module, args, kwargs: shapes.append(list(kwargs['input_ids'].shape)), with_kwargs=True)
        torch.cuda.synchronize(backend.device)
        begin = time.perf_counter()
        if payload['mode'] == 'independent':
            fields = [{**request.field, 'id': json.dumps([request.task_id, request.field['id']])}
                      for request in requests]
            result = score_grouped(backend, [fields], payload['mode'],
                                   admitted_prompts=[request.admitted for request in requests])
            for row, request in zip(result['groups'][0], requests, strict=True):
                row['id'] = request.field['id']
        else:
            result = tail.score(requests, lengths, banks['base'], banks['state'])
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - begin
        hook.remove()
        values = result['groups'][0]
        ranks = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(ranks, values, group=commands.control_group)
        assert all(row == values for row in ranks)
        report = {'phase': payload['phase'], 'mode': payload['mode'], 'workload': payload['workload'],
                  'task_ids': [r.task_id for r in requests], 'node_ids': [r.field['id'] for r in requests],
                  'input_tokens': [len(r.input_ids) for r in requests], 'forward_shapes': shapes,
                  'seconds': elapsed, 'results': result, 'allrank_exact_outputs': True}
        (output / config['rank_file'].format(rank=dist.get_rank(), **payload)).write_text(json.dumps(report, indent=2) + '\n')
        return report

    commands.register(config['command'], compute)
    if commands.is_leader:
        try:
            sources = {}
            for workload in config['workloads']:
                source = workload['source_run']
                if source not in sources:
                    prepared, elapsed = prepare({**config, 'source_run': source}, backend, shared)
                    sources[source] = prepared[config['variant']]
            reports = []
            for phase in config['phases']:
                for workload in config['workloads']:
                    for mode in config['modes']:
                        reports.append(commands.call(config['command'], {'phase': phase, 'mode': mode,
                            'workload': workload['name'], 'cohort': sources[workload['source_run']][workload['cohort']]}))
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup, 'reports': reports}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    save_tuning(output / config['cache_file'].format(rank=dist.get_rank()), backend, settings)
    tail.graphs.clear()
    cache.clear()
    for bank in banks.values():
        bank.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
