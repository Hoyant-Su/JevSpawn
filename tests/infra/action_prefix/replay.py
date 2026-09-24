import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.jevspawn_service import DecisionRequest
from baselines.common.parallel_run import initialize_parallel
from baselines.common.runtime_contract import RuntimeContract
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prefix, controller_prompts
from tests.infra.action_prefix.messages import ActionMessages
from tests.infra.action_prefix.runtime import ActionPrefixTail


def messages_for(field, service):
    user, = controller_prompts([field['state']], field['question'], field['options'],
        service.backend.answer_labels, CONTROLLER['output_instruction'],
        contexts=[field['context']], histories=[field['history']])
    return service.contract_messages([
        {'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}])


def request_for(field, messages, task_id, service, settings):
    tokenizer = service.backend.tokenizer
    rendered = tokenizer.apply_chat_template(messages, **settings['chat_template'])
    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
    root_user = controller_prefix(field['context'], '')
    root = tokenizer.apply_chat_template([messages[0], {'role': 'user', 'content': root_user}],
                                          **settings['chat_template'])
    root_tokens = tokenizer(root, add_special_tokens=False)['input_ids']
    length = common_prefix([tokens, root_tokens])
    request = DecisionRequest(messages, settings['finite_new_tokens'], service.shared.generation.temperature,
                              (), task_id, time.perf_counter(), None, field=field)
    request.admitted = AdmittedPrompt(rendered, tuple(tokens))
    request.root_tokens = tuple(tokens[:length])
    return request


def prepare(backend, shared, settings):
    protocol = json.loads(Path(settings['protocol']).read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    service = RuntimeContract()
    service.shared, service.backend, service.execution_metadata = shared, backend, {}
    service.configure_runtime_contract(protocol['method']['settings'])
    groups = []
    for source in settings['sources']:
        trace = json.loads(Path(source).read_text())
        computations = trace['trace']['rounds'][settings['turn']]['parent_computations'][settings['parent']]
        first = next(record for record in computations if 'fields' in record)
        first_field, = first['fields']
        original, = first['requests']
        definition = json.loads(original['state'])['active_declaration']
        fields = {field['id']: field for field in definition['fields']}
        next_field = definition['fields'][definition['fields'].index(first_field) + 1]
        first_messages = messages_for(original, service)
        conversation = ActionMessages(first_messages, first_field, {}, backend.answer_labels,
            load_prompt(settings['prompts']), CONTROLLER, settings['serialization'])
        pairs = [(request_for(original, first_messages, trace['task_id'], service, settings),
                  request_for({**original, 'action_messages': first_messages}, first_messages,
                              trace['task_id'], service, settings))]
        for computation in computations:
            if 'fields' not in computation or computation['fields'][0]['id'] != next_field['id']:
                continue
            field, = computation['requests']
            bindings = json.loads(field['state'])['bound_fields']
            messages = conversation.fork().append(next_field, bindings, fields, field['options'])
            candidate = {**field, 'action_messages': messages}
            pairs.append((request_for(field, messages_for(field, service), trace['task_id'], service, settings),
                          request_for(candidate, messages, trace['task_id'], service, settings)))
        assert len(pairs) == protocol['method']['settings']['rollout']['branch_width'] + 1
        groups.append(pairs)
    return groups, protocol['inference']['settings']


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    groups, inference = prepare(backend, shared, settings)
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'workloads': []}
    for count in settings['task_counts']:
        selected = groups[:count]
        batches = [[group[0] for group in selected]]
        extensions = [pair for group in selected for pair in group[1:]]
        batches.extend(extensions[start:start + shared.runtime.branch_batch_size]
                       for start in range(0, len(extensions), shared.runtime.branch_batch_size))
        results, all_logits = {}, {}
        for mode in settings['modes']:
            tail_class = ActionPrefixTail if mode['reuse_action_prefix'] else StableFiniteGraphTail
            tail = tail_class(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                              inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            rows, logits = [], []
            for pairs in batches:
                requests = [pair[mode['message_index']] for pair in pairs]
                shapes = []

                def observe(module, args, kwargs):
                    shapes.append(list(kwargs['input_ids'].shape))

                hook = backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
                torch.cuda.synchronize(backend.device)
                started = time.perf_counter()
                result = tail.score(requests, [len(request.root_tokens) for request in requests], roots)
                torch.cuda.synchronize(backend.device)
                wall = time.perf_counter() - started
                hook.remove()
                logits.append(tail.last_logits.clone())
                rows.append({'wall_seconds': wall, 'forward_shapes': shapes,
                    'choices': [value['choice'] for value in result['groups'][0]],
                    'work': {key: result[key] for key in settings['workload_fields']},
                    'action_prefix_reused_by_row': result.get('action_prefix_reused_by_row', [])})
            results[mode['name']], all_logits[mode['name']] = rows, logits
            tail.graphs.clear()
        left, right = settings['cache_comparison']
        differences = []
        for cached, recomputed in zip(all_logits[left], all_logits[right], strict=True):
            delta = cached.float() - recomputed.float()
            differences.append({'max_absolute_error': delta.abs().max().item(),
                'relative_norm_error': (delta.norm() / recomputed.float().norm()).item(),
                'argmax_equal': torch.equal(cached.argmax(-1), recomputed.argmax(-1))})
        report['workloads'].append({'task_count': count, 'request_count': sum(map(len, batches)),
                                   'modes': results, 'cache_numerical_comparison': differences})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
