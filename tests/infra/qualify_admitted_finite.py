import argparse
from concurrent.futures import Future
from functools import partial
import importlib.util
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache, decode_request, encode_request
from baselines.common.runtime import InferenceRuntime
from baselines.common.shared_state_prefix_service import SharedStatePrefixService
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from qualify_shared_state_prefix import workload


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def archived_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ObservedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, values, **kwargs):
        result = self.tokenizer(values, **kwargs)
        self.calls.append((values, result['input_ids']))
        return result


def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    backend.tokenizer = ObservedTokenizer(backend.tokenizer)
    archive = Path(settings['archive'])
    reference_core = archived_module('reference_core', archive / 'jevspawn/src/jev_spawn/algo/structured.py')
    reference_grouped = archived_module('reference_grouped', archive / 'src/methods/program_execution/grouped.py')
    reference_grouped.score_fields = reference_core.score_fields
    method = json.loads(Path(settings['method']).read_text())
    CONTROLLER.clear()
    CONTROLLER.update(load_prompt(settings['controller_prompt']))
    runtime = InferenceRuntime(settings['shared_config'], partial(SharedStatePrefixService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)
    service = runtime.service
    caches = {variant: {'task': ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands),
                       'state': ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)}
              for variant in settings['variants']}
    results = {phase: {variant: [] for variant in settings['variants']} for phase in settings['phases']}

    @torch.inference_mode()
    def score(payload):
        batch = [decode_request(record) for record in payload['requests']]
        variant = payload['variant']
        service.prefix_cache, service.state_cache = caches[variant]['task'], caches[variant]['state']
        backend.tokenizer.calls.clear()
        started = time.perf_counter()
        if variant == settings['reference_variant']:
            length = service.task_prefix_length(batch)
            fields = [{**request.field, 'id': str(index)} for index, request in enumerate(batch)]
            result = reference_grouped.score_grouped(backend, [fields], service.field_mode,
                prefix_cache=service.state_cache, base_prefix_cache=service.prefix_cache, base_prefix_length=length)
        else:
            assert variant == settings['candidate_variant']
            result = service._score(batch)
        torch.cuda.synchronize(backend.device)
        scoring_seconds = time.perf_counter() - started
        rendered = [request.admitted.rendered for request in batch]
        encoded = [list(request.admitted.tokens) for request in batch]
        calls = [ids for texts, ids in backend.tokenizer.calls if texts == rendered]
        exact = all(ids == encoded for ids in calls)
        assert exact
        assert bool(calls) == (variant == settings['reference_variant'])
        record = {'phase': payload['phase'], 'variant': variant, 'source_ids': payload['source_ids'],
            'admission_seconds': payload['admission_seconds'], 'scoring_seconds': scoring_seconds,
            'full_prompt_tokenizer_calls_during_scoring': len(calls),
            'original_prompt_token_ids_exact': exact, 'input_tokens': list(map(len, encoded)),
            'result': result}
        results[payload['phase']][variant].append(record)
        return record

    commands.register(settings['command'], score)
    output = Path(settings['output'])
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=False)
        source_settings = json.loads(Path(settings['source']).read_text())
        protocol, context, groups, originals, calls = workload(source_settings)
        identity = source_settings['task_id']
        runtime.deadlines.start(identity)
        save(output / 'protocol.json', {'settings': settings, 'startup': startup, 'groups': groups,
                                      'controller': CONTROLLER})
        try:
            for phase in settings['phases']:
                for variant in settings['variants']:
                    for group in groups:
                        started = time.perf_counter()
                        batch = []
                        for node in group:
                            prompt = controller_prompts([node['state']], node['question'], node['options'],
                                list(backend.answer_labels[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
                            request = service.request_type(
                                [{'role': 'system', 'content': CONTROLLER['system']},
                                 {'role': 'user', 'content': prompt}], settings['decision_tokens'],
                                shared.generation.temperature, (), identity, time.perf_counter(), Future(), field=node)
                            batch.append(request)
                        admitted = service._validate_inputs(batch)
                        assert len(admitted) == len(group)
                        admission_seconds = time.perf_counter() - started
                        payload = {'phase': phase, 'variant': variant,
                            'source_ids': [node['id'] for node in group],
                            'requests': [encode_request(request) for request in admitted],
                            'admission_seconds': admission_seconds}
                        record = commands.call(settings['command'], payload)
                        record['admission_and_dispatch_inclusive_seconds'] = time.perf_counter() - started
                        save(output / f'{phase}-{variant}-{len(results[phase][variant])}.json', record)
            runtime.close()
        finally:
            commands.finish()
    else:
        commands.serve()
        runtime.close()
    comparisons = []
    for phase in settings['phases']:
        reference, candidate = results[phase][settings['reference_variant']], results[phase][settings['candidate_variant']]
        for left, right in zip(reference, candidate, strict=True):
            assert left['source_ids'] == right['source_ids'] and left['input_tokens'] == right['input_tokens']
            assert left['result']['groups'] == right['result']['groups']
            comparisons.append({'phase': phase, 'source_ids': left['source_ids'], 'exact_logits_and_choices': True})
    ranks = [None] * shared.runtime.world_size
    dist.all_gather_object(ranks, {'rank': dist.get_rank(), 'comparisons': comparisons}, group=commands.control_group)
    if commands.is_leader:
        save(output / 'completion.json', {'settings': settings, 'results': results, 'rank_parity': ranks})
        print(json.dumps({'all_logits_and_choices_exact': True, 'rank_count': len(ranks)}), flush=True)
    for cache in caches.values():
        for prefix in cache.values():
            prefix.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
