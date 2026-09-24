import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache
from baselines.common.resources import TEMPLATES
from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.finite_batch import score_finite_batch
from jev_spawn.infra.finite_graph import FiniteGraphTail, RaggedFiniteGraphTail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped
from mixed_instrumentation import instrument
from qwen35_graph_profile import summarize_trace


def main(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    protocol = json.loads(Path(config['run'], 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    banks = {variant: {name: ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
                      for name in config['caches']} for variant in config['variants']}
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    implementations = {'grouped': FiniteGraphTail, 'ragged': RaggedFiniteGraphTail}
    if 'stable' in config['graph_implementations'].values():
        implementations['stable'] = partial(StableFiniteGraphTail,
            settings=json.loads(Path(config['stable_settings']).read_text()))
    graph_tails = {variant: implementations[config['graph_implementations'][variant]](backend, shared.runtime,
        ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands), settings)
        for variant, settings in config['graph_variants'].items()}

    def graph_score(tail, requests, lengths, bank):
        return tail.score(requests, lengths, bank['base'], bank['state'])

    scorers = {config['candidate_variant']: lambda requests, lengths, bank:
        score_finite_batch(backend, requests, lengths, bank['base'], bank['state'])}
    scorers.update({variant: partial(graph_score, tail) for variant, tail in graph_tails.items()})

    def prepare(workload):
        requests = []
        for selection in workload:
            index, offset = selection
            task = protocol['tasks'][index]
            record = json.loads(Path(config['run'], config['task_file'].format(index=index)).read_text())
            call = [item for item in record['calls'] if item['kind'] == 'finite'][offset]
            environment = TaskEnvironment(task, protocol['tools'], output, deadline=lambda: None)
            context = environment.reset()
            state = protocol['prompts']['encoded_state'].format(
                value=json.dumps(call['input'], **config['serialization']))
            state = TEMPLATES['worker_state'].format(context=context, state=state)
            field = {'id': call['node'], 'context': context, 'state': state,
                     'question': call['question'], 'options': call['options']}
            user, = controller_prompts([state], field['question'], field['options'],
                list(backend.answer_labels[:len(field['options'])]), CONTROLLER['output_instruction'])
            messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                               enable_thinking=shared.generation.enable_thinking)
            tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
            assert len(tokens) <= shared.model.max_input_tokens
            assert len(tokens) == call['result']['input_tokens']
            requests.append(SimpleNamespace(task_id=task['task_id'], field=field,
                                            input_ids=tokens, admitted=AdmittedPrompt(rendered, tuple(tokens))))
        proxy = SimpleNamespace(backend=backend)
        lengths = [TaskPrefixService.task_prefix_length(proxy, [request]) for request in requests]
        return requests, lengths

    @torch.inference_mode()
    def compute(payload):
        variant, requests, lengths = payload['variant'], payload['requests'], payload['lengths']
        bank = banks[variant]
        if payload['clear']:
            if variant in graph_tails:
                graph_tails[variant].prefix_cache.clear()
            for cache in bank.values():
                cache.clear()
        shapes = []
        hook = backend.model.model.register_forward_pre_hook(
            lambda module, args, kwargs: shapes.append(list(kwargs['input_ids'].shape)), with_kwargs=True)
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        if variant == config['reference_variant']:
            groups = defaultdict(list)
            for index, request in enumerate(requests):
                groups[(request.task_id, request.field['context'])].append(index)
            answers = {}
            details = []
            for indices in groups.values():
                result = score_grouped(backend, [[requests[index].field for index in indices]], config['mode'],
                    prefix_cache=bank['state'], base_prefix_cache=bank['base'], base_prefix_length=lengths[indices[0]],
                    admitted_prompts=[requests[index].admitted for index in indices])
                answers.update(zip(indices, result['groups'][0], strict=True))
                details.append(result)
            values = [answers[index] for index in range(len(requests))]
        else:
            result = scorers[variant](requests, lengths, bank)
            values, = result['groups']
            details = [result]
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        hook.remove()
        rankmax = torch.tensor(elapsed, device=backend.device, dtype=torch.float64)
        dist.all_reduce(rankmax, op=dist.ReduceOp.MAX)
        record = {'phase': payload['phase'], 'variant': variant, 'seconds': elapsed,
                  'rankmax_seconds': rankmax.item(), 'forward_shapes': shapes,
                  'task_ids': [request.task_id for request in requests], 'answers': values, 'details': details}
        Path(output, config['rank_file'].format(rank=dist.get_rank(), phase=payload['phase'], variant=variant)).write_text(
            json.dumps(record, indent=2) + '\n')
        return record

    def score(payload):
        if payload['phase'] not in config['profile']['phases']:
            return compute(payload)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     profile_memory=False, with_stack=False) as observed, instrument(backend, config['profile']['scopes']):
            record = compute(payload)
        path = output / config['profile']['trace_file'].format(rank=dist.get_rank(), phase=payload['phase'])
        observed.export_chrome_trace(str(path))
        record['trace'] = summarize_trace(path, config['profile']['trace'])
        names = [config['profile']['scopes']['model'], *config['profile']['scopes']['functions'].values()]
        record['operator_scopes'] = [{'name': event.key, 'count': event.count,
            'cpu_time_total_us': event.cpu_time_total, 'self_cpu_time_total_us': event.self_cpu_time_total,
            'device_time_total_us': event.device_time_total, 'self_device_time_total_us': event.self_device_time_total}
            for event in observed.key_averages() if event.key in names]
        record['timing_scope'] = 'Instrumented profiling only; ordinary replay timings are reported separately.'
        Path(output, config['rank_file'].format(rank=dist.get_rank(), phase=payload['phase'], variant=payload['variant'])).write_text(
            json.dumps(record, indent=2) + '\n')
        return record

    commands.register(config['command'], score)
    if commands.is_leader:
        try:
            for phase in config['phases']:
                requests, lengths = prepare(config['workloads'][phase['workload']])
                for variant in config['variants']:
                    records.append(commands.call(config['command'], {'phase': phase['name'], 'clear': phase['clear'],
                        'requests': requests, 'lengths': lengths, 'variant': variant}))
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                                                         'records': records}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    for tail in graph_tails.values():
        tail.graphs.clear()
        tail.prefix_cache.clear()
    torch.cuda.synchronize(backend.device)
    for bank in banks.values():
        for cache in bank.values():
            cache.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    main(json.loads(args.config.read_text()))
