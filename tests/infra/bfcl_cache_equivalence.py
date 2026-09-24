import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.mixed_prefix_service import MixedPrefixService
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache
from jev_spawn.algo.structured import padded
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.finite_batch import score_finite_with_tail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts


def prepare(config, backend, shared):
    source = Path(config['source_run'])
    protocol = json.loads((source / 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    calls = {}
    for path in sorted(source.glob('task-*.json')):
        task = json.loads(path.read_text())
        for call in task['calls']:
            if call['kind'] == 'boundary':
                calls[task['task_id'], call['node']] = call
    cohorts, excluded = [], []
    for path in sorted(source.glob('session-*/batches.json')):
        for index, batch in enumerate(json.loads(path.read_text())):
            if batch.get('operation') != 'finite':
                continue
            keys = list(zip(batch['task_ids'], batch['node_ids'], strict=True))
            if not all(key in calls for key in keys):
                excluded.append({'batch': index, 'keys': keys, 'reason': 'Not a complete recorded boundary cohort'})
                continue
            requests, stored = [], []
            for key, count in zip(keys, batch['input_tokens'], strict=True):
                call = calls[key]
                field = call['request']
                state = CONTROLLER['state_template'].format(context=field['context'], state=field['state'])
                user, = controller_prompts([state], field['question'], field['options'],
                    list(backend.answer_labels[:len(field['options'])]), CONTROLLER['output_instruction'])
                messages = [{'role': 'system', 'content': CONTROLLER['system']},
                            {'role': 'user', 'content': user}]
                rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
                tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
                assert len(tokens) == count == call['decision']['input_tokens']
                assert len(tokens) <= shared.model.max_input_tokens
                requests.append(SimpleNamespace(task_id=key[0], field=field, input_ids=tokens,
                    admitted=AdmittedPrompt(rendered, tuple(tokens))))
                stored.append(call['decision'])
            groups = defaultdict(list)
            for request in requests:
                groups[request.task_id, request.field['context']].append(request)
            service = SimpleNamespace(backend=backend, shared=shared)
            lengths = {key: MixedPrefixService.task_prefix_length(service, group)
                       for key, group in groups.items()}
            cohorts.append({'batch': index, 'requests': requests, 'stored': stored,
                'lengths': [lengths[r.task_id, r.field['context']] for r in requests]})
    assert cohorts
    return cohorts, excluded


@torch.inference_mode()
def main(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(config['parallel_settings']).read_text()))
    cohorts, excluded = prepare(config, backend, shared)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    banks = {name: ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
             for name in config['cache_names']}
    tail = StableFiniteGraphTail(backend, shared.runtime,
        ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands),
        config['state_copy'], config['graph_shape'])
    direct_banks = {name: ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
                    for name in config['cache_names']}
    direct_tail = StableFiniteGraphTail(backend, shared.runtime,
        ParallelPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands),
        config['state_copy'], config['graph_shape'])
    records = []
    for phase in config['phases']:
        for cohort in cohorts:
            requests = cohort['requests']
            torch.cuda.synchronize(backend.device)
            started = time.perf_counter()
            ids, mask = padded([list(r.admitted.tokens) for r in requests],
                backend.tokenizer.pad_token_id, backend.device, config['padding_side'])
            result = backend.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
            hidden = result.last_hidden_state[:, config['last_position']].clone()
            logits = F.linear(hidden.float(), backend.finite_output_weights)
            plain = [row[:len(request.field['options'])].tolist()
                     for row, request in zip(logits, requests, strict=True)]
            del result, hidden, logits, ids, mask
            torch.cuda.synchronize(backend.device)
            plain_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cached = tail.score(requests, cohort['lengths'], banks['base'], banks['state'])
            torch.cuda.synchronize(backend.device)
            cache_seconds = time.perf_counter() - started
            lengths = {tuple(request.admitted.tokens): length
                       for request, length in zip(requests, cohort['lengths'], strict=True)}
            # Change only the cache boundary; preserve the full scoring computation and all requests.
            started = time.perf_counter()
            direct = score_finite_with_tail(backend, requests, cohort['lengths'], direct_banks['base'],
                direct_banks['state'], direct_tail, direct_tail.extend_states,
                lambda sequences: lengths[tuple(sequences[0])], None)
            torch.cuda.synchronize(backend.device)
            direct_seconds = time.perf_counter() - started
            values, = cached['groups']
            pairs = []
            for request, reference, candidate, stored in zip(requests, plain, values, cohort['stored'], strict=True):
                choice = request.field['options'][max(range(len(reference)), key=reference.__getitem__)]['id']
                pairs.append({'task_id': request.task_id, 'node_id': request.field['id'],
                    'input_tokens': len(request.admitted.tokens), 'plain_choice': choice,
                    'cached_choice': candidate['choice'], 'stored_choice': stored['choice'],
                    'choice_agreement': choice == candidate['choice'],
                    'plain_logits': reference, 'cached_logits': candidate['option_logits'],
                    'stored_logits': stored['option_logits'],
                    'maximum_absolute_logit_difference': max(abs(a-b) for a,b in
                        zip(reference, candidate['option_logits'], strict=True))})
            for pair, candidate in zip(pairs, direct['groups'][0], strict=True):
                pair.update(direct_choice=candidate['choice'], direct_logits=candidate['option_logits'],
                    direct_choice_agreement=pair['plain_choice'] == candidate['choice'])
            records.append({'phase': phase, 'source_batch': cohort['batch'],
                'plain_seconds': plain_seconds, 'cached_seconds': cache_seconds,
                'direct_seconds': direct_seconds, 'direct_timings': direct['timings'],
                'cached_computed_tokens': cached['computed_input_tokens'],
                'direct_computed_tokens': direct['computed_input_tokens'],
                'cache_timings': cached['timings'], 'decisions': pairs})
            path = output / f'rank-{dist.get_rank()}.json'
            path.write_text(json.dumps({'startup': startup, 'excluded_cohorts': excluded,
                'records': records}, **config['serialization']) + '\n')
            if commands.is_leader:
                print(json.dumps({'phase': phase, 'batch': cohort['batch'],
                    'agreements': sum(row['choice_agreement'] for row in pairs), 'decisions': len(pairs),
                    'maxdiff': max(row['maximum_absolute_logit_difference'] for row in pairs),
                    'plain_seconds': plain_seconds, 'cached_seconds': cache_seconds,
                    'direct_seconds': direct_seconds,
                    'direct_agreements': sum(row['direct_choice_agreement'] for row in pairs)}), flush=True)
    tail.graphs.clear()
    tail.prefix_cache.clear()
    for bank in banks.values():
        bank.clear()
    direct_tail.graphs.clear()
    direct_tail.prefix_cache.clear()
    for bank in direct_banks.values():
        bank.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
