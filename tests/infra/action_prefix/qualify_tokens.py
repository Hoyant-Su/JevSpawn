import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import time
from types import SimpleNamespace

from transformers import AutoTokenizer
import torch
import yaml

from baselines.common.config import SharedConfig
from baselines.common.runtime_contract import RuntimeContract
from baselines.common.parallel_service import decode_request, encode_request
from jev_spawn.algo.composition import rank_extensions
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import AdmittedPrompt, native_labels
from jev_spawn.schema import CONTROLLER, controller_prompts
from tests.infra.action_prefix.messages import ActionMessages
from tests.infra.action_prefix.runtime import ActionExecution, ActionRequest, install


def run(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)
    label_settings = json.loads(Path(settings['labels']).read_text())
    labels, label_ids = native_labels(tokenizer, label_settings)
    source = json.loads(Path(settings['source']).read_text())
    protocol = json.loads(Path(settings['protocol']).read_text())
    records = source['trace']['rounds'][settings['turn']]['parent_computations'][settings['parent']]
    first = next(record for record in records if 'fields' in record)
    request, = first['requests']
    field, = first['fields']
    definition = json.loads(request['state'])['active_declaration']
    fields = {item['id']: item for item in definition['fields']}
    following = definition['fields'][definition['fields'].index(field) + 1]
    controller = load_prompt(settings['controller_prompts'])
    controller['option_template'] = protocol['prompts']['option_template']
    CONTROLLER.update(controller)
    user, = controller_prompts([request['state']], request['question'], request['options'], labels,
        controller['output_instruction'], contexts=[request['context']], histories=[request['history']])
    service = RuntimeContract()
    service.shared = SharedConfig.load(settings['shared_config'])
    service.backend = SimpleNamespace(answer_labels=labels)
    service.execution_metadata = {}
    service.configure_runtime_contract(protocol['method']['settings'])
    messages = service.contract_messages([
        {'role': 'system', 'content': controller['system']}, {'role': 'user', 'content': user}])
    rendered = tokenizer.apply_chat_template(messages, **settings['chat_template'])
    before = tokenizer(rendered, add_special_tokens=False)['input_ids']
    execution = json.loads(Path(settings['execution']).read_text())
    options = [{'id': execution['candidate_id'].format(index=index),
                'description': json.dumps(value, **settings['serialization'])}
               for index, value in enumerate(following['values'])]
    state = ActionMessages(messages, field, {}, labels, load_prompt(settings['prompts']),
                           controller, settings['serialization'])
    report = {'model': shared['model']['path'], 'source': settings['source'], 'cases': []}
    for index in settings['selected_indices']:
        branch = state.fork()
        bindings = {field['id']: field['values'][index]}
        after_messages = branch.append(following, bindings, fields, options)
        text = tokenizer.apply_chat_template(after_messages, **settings['chat_template'])
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        expected = before + [label_ids[index]]
        assert tokens[:len(expected)] == expected
        assert after_messages[-2]['content'] == labels[index]
        report['cases'].append({'choice': index, 'label': labels[index], 'value': bindings[field['id']],
            'prior_tokens': len(before), 'next_tokens': len(tokens), 'common_prefix': common_prefix([before, tokens]),
            'prefix_plus_actual_label_matches': True, 'messages': after_messages})
    assert state.messages == messages and not state.bindings
    install(json.loads(Path(settings['action_prefix_settings']).read_text()))
    runtime = ActionExecution(request['context'], service, source['task_id'], execution)
    runtime.fields = fields
    runtime.active_declaration = definition
    runtime.action_template = definition['action']
    prepared = runtime.prepare([field])
    runtime.commit([field], *prepared, first['decisions'])
    distribution = torch.tensor(first['decisions'][0]['probabilities'])
    parent_indices, choices, scores = rank_extensions([distribution], distribution.new_zeros((1,)),
        protocol['method']['settings']['rollout']['branch_width'])
    report['beam_cases'] = []
    singleton = settings['singleton_field']
    for parent, choice, score in zip(parent_indices, choices, scores.tolist(), strict=True):
        branch = runtime.fork()
        branch.values[field['id']] = field['values'][choice]
        branch.bound_fields[field['id']] = field['values'][choice]
        branch.fields[singleton['id']] = singleton
        branch.values[singleton['id']] = singleton['values'][0]
        branch.bound_fields[singleton['id']] = singleton['values'][0]
        _, branch_requests, _ = branch.prepare([following])
        branch_messages = branch_requests[0]['action_messages']
        assert branch_messages[-2]['content'] == labels[choice]
        expected_binding = json.dumps(singleton['values'][0], **settings['serialization'])
        assert expected_binding in branch_messages[-1]['content']
        assert json.dumps(field['values'][choice]) in branch_messages[-1]['content']
        packet = ActionRequest(branch_messages, label_settings['native_token_count'],
            service.shared.generation.temperature, (), source['task_id'], time.perf_counter(),
            Future(), field=branch_requests[0])
        packet_text = tokenizer.apply_chat_template(branch_messages, **settings['chat_template'])
        packet.input_ids = tokenizer(packet_text, add_special_tokens=False)['input_ids']
        packet.admitted = AdmittedPrompt(packet_text, tuple(packet.input_ids))
        packet.root_tokens = tuple(before)
        wire = encode_request(packet)
        assert encode_request(decode_request(wire)) == wire
        report['beam_cases'].append({'parent': parent, 'choice': choice, 'actual_value': field['values'][choice],
            'log_probability': score, 'assistant_label': branch_messages[-2]['content'],
            'singleton_value': singleton['values'][0], 'request_codec_roundtrip': True})
    assert not runtime.bound_fields and len(runtime.action_messages.messages) == len(messages)
    report['template_generation_suffix'] = rendered[len(tokenizer.apply_chat_template(
        messages, **{**settings['chat_template'], 'add_generation_prompt': False})):]
    report['passed'] = True
    Path(settings['output']).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
