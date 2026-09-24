import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache
from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.finite_graph import RaggedFiniteGraphTail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts


class ObservedPrefixCache(ParallelPrefixCache):
    def __init__(self, original, commands):
        super().__init__(original, commands)
        self.plans = []

    def resident_plan(self, keys):
        hits = super().resident_plan(keys)
        self.plans.append({'hits': hits, 'token_counts': [len(key[0]) for key in keys]})
        return hits


def prepare(config, backend, shared):
    run = Path(config['source_run'])
    protocol = json.loads((run / 'protocol.json').read_text())
    completion = json.loads((run / 'completion.json').read_text())
    assert completion['tasks'] == len(protocol['tasks'])
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    batches = [batch for path in sorted(run.glob(config['batches_glob']))
               for batch in json.loads(path.read_text()) if batch.get('operation') == 'finite']
    paths = sorted((run / config['input_directory']).glob(config['input_glob']))
    assert len(paths) == len(batches)
    cohorts = []
    started = time.perf_counter()
    for index, (path, batch) in enumerate(zip(paths, batches, strict=True)):
        recorded = json.loads(path.read_text())
        assert recorded['cohort'] == index and len(recorded['requests']) == batch['batch_size']
        requests, lengths = [], []
        for item, task_id, node_id, count in zip(recorded['requests'], batch['task_ids'], batch['node_ids'], batch['input_tokens'], strict=True):
            field = item['field']
            assert item['task_id'] == task_id and field['id'] == node_id
            assert len(item['tokens']) == item['input_tokens'] == count <= shared.model.max_input_tokens
            user, = controller_prompts([field['state']], field['question'], field['options'],
                list(backend.answer_labels[:len(field['options'])]), CONTROLLER['output_instruction'])
            messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            assert messages == item['messages']
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
            assert rendered == item['rendered']
            assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == item['tokens']
            request = SimpleNamespace(task_id=task_id, field=field, input_ids=item['tokens'],
                                      admitted=AdmittedPrompt(rendered, tuple(item['tokens'])))
            assert TaskPrefixService.task_prefix_length(SimpleNamespace(backend=backend), [request]) == item['base_length']
            requests.append(request)
            lengths.append(item['base_length'])
        cohorts.append((requests, lengths))
    return {name: cohorts for name in config['variants']}, time.perf_counter() - started


def main(config, prepare_inputs):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    tails, banks = {}, {}
    for name, implementation in config['variants'].items():
        cache = ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
        arguments = (backend, shared.runtime, cache, config['state_copy'])
        constructors = {'ragged': lambda: RaggedFiniteGraphTail(*arguments),
                        'stable': lambda: StableFiniteGraphTail(*arguments, settings=config['graph_shape'])}
        tails[name] = constructors[implementation]()
        banks[name] = {key: ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
                       for key in config['caches']}

    @torch.inference_mode()
    def replay(payload):
        name = payload['variant']
        tail, bank = tails[name], banks[name]
        for current in tails.values():
            current.graphs.clear()
            current.prefix_cache.clear()
        for current in banks.values():
            for cache in current.values():
                cache.clear()
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        initial_allocated = torch.cuda.memory_allocated(backend.device)
        started = time.perf_counter()
        records = []
        for index, (requests, lengths) in enumerate(payload['cohorts']):
            caches = {**bank, 'final': tail.prefix_cache}
            for cache in caches.values():
                cache.plans.clear()
            begin = time.perf_counter()
            result = tail.score(requests, lengths, bank['base'], bank['state'])
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - begin
            rankmax = torch.tensor(elapsed, device=backend.device, dtype=torch.float64)
            dist.all_reduce(rankmax, op=dist.ReduceOp.MAX)
            records.append({'cohort': index, 'task_ids': [r.task_id for r in requests],
                            'node_ids': [r.field['id'] for r in requests],
                            'rankmax_seconds': rankmax.item(), 'cache_plans': {key: list(cache.plans) for key, cache in caches.items()},
                            'result': result})
        elapsed = time.perf_counter() - started
        choices = [[value['choice'] for value in record['result']['groups'][0]] for record in records]
        rank_choices = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(rank_choices, choices, group=commands.control_group)
        assert all(value == choices for value in rank_choices)
        report = {'phase': payload['phase'], 'variant': name, 'records': records,
                  'rank_choice_agreement': True, 'elapsed_seconds': elapsed,
                  'initial_allocated_bytes': initial_allocated,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)}
        path = output / config['rank_file'].format(rank=dist.get_rank(), phase=payload['phase'], variant=name)
        path.write_text(json.dumps(report, indent=2) + '\n')
        return report

    commands.register(config['command'], replay)
    if commands.is_leader:
        try:
            cohorts_by_variant, preparation = prepare_inputs(config, backend, shared)
            reports = []
            for phase in config['phases']:
                for name in config['variants']:
                    reports.append(commands.call(config['command'], {'phase': phase, 'variant': name, 'cohorts': cohorts_by_variant[name]}))
            comparisons = []
            for phase in config['phases']:
                reference, candidate = [report for report in reports if report['phase'] == phase]
                rows = []
                for left, right in zip(reference['records'], candidate['records'], strict=True):
                    assert left['task_ids'] == right['task_ids'] and left['node_ids'] == right['node_ids']
                    pairs = list(zip(left['result']['groups'][0], right['result']['groups'][0], strict=True))
                    rows.append({'cohort': left['cohort'], 'decisions': len(pairs),
                                 'agreements': sum(a['choice'] == b['choice'] for a, b in pairs),
                                 'maximum_absolute_logit_difference': max(abs(x - y) for a, b in pairs
                                     for x, y in zip(a['option_logits'], b['option_logits'], strict=True))})
                comparisons.append({'phase': phase, 'cohorts': rows})
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'verified_preparation_seconds': preparation, 'reports': reports,
                'comparisons': comparisons}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    for tail in tails.values():
        tail.graphs.clear()
        tail.prefix_cache.clear()
    for bank in banks.values():
        for cache in bank.values():
            cache.clear()
    torch.cuda.synchronize(backend.device)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()), prepare)
