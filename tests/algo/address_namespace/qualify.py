from copy import deepcopy
import json
from pathlib import Path

from transformers import AutoTokenizer
import yaml

from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import PROMPTS, load_prompt, resolve_prompts
from jev_spawn.runtime.reference_transport import unpack_references


config = json.loads(Path('tests/algo/address_namespace/config.json').read_text())
source = json.loads(Path(config['source_method']).read_text())
candidate = json.loads(Path(config['candidate_method']).read_text())
spec = json.loads(Path(config['specification']).read_text())
shared = yaml.safe_load(Path(spec['shared_config']).read_text())
original_prompts = json.loads(Path(config['prompt_snapshot']).read_text())
assert all(PROMPTS['templates'][key] == value for key, value in original_prompts.items())
left, right = deepcopy(source), deepcopy(candidate)
for method in (left, right):
    method.pop('prompts')
    method['settings']['reference_transport'].pop('prompts')
    method['settings']['finite_recording'].pop('directory')
assert left == right
old = resolve_prompts(load_prompt(source['prompts']))
new = resolve_prompts(load_prompt(candidate['prompts']))
transport = load_resource(source['settings']['reference_transport']['resource'])
serialization = load_resource('finite_control')['serialization']
task = next(json.loads(line) for line in Path(config['task_file']).read_text().splitlines()
            if json.loads(line)['task_id'] == config['task_id'])
requests = {request['field']['id']: request
            for path in sorted(Path(config['source_run'], 'finite-inputs').glob('*.json'))
            for request in json.loads(path.read_text())['requests']
            if request['task_id'] == config['task_id']}
tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'])
rows, inputs = [], []
for node in config['node_ids']:
    original = requests[node]
    field = original['field']
    semantic = unpack_references(field['input'], transport)
    payload = semantic['input']
    template = payload['evidence']['domain_template']
    index = template['position']
    ordinal = index + config['count_step']
    assert payload['path'] == template['template_path']
    assert payload['path'][-1] == index
    assert payload['schema'] == config['scalar_schema']
    address = old['values']['domain_template_address'].format(ordinal=ordinal, length=template['child_length'])
    assert payload['address'][-1] == address
    assert all(json.loads(option['description']) == value
               for option, value in zip(field['options'], field['candidate_values'], strict=True))
    assert [item['value'] for item in semantic['candidates']] == field['candidate_values']
    for value in field['candidate_values']:
        if 'reference' in value:
            reference = value['reference']
            observed = task
            for part in reference['path']:
                observed = observed[part]
            start, stop = reference['span']
            assert observed[start:stop] == value['description']
    assert json.dumps(task['input'], ensure_ascii=False) in field['context']
    assert json.dumps(task['answer_schema'], ensure_ascii=False) in field['context']
    changed = deepcopy(original)
    changed_payload = changed['field']['input']['root']['input']
    changed_payload['address'][-1] = new['values']['domain_template_address'].format(
        ordinal=ordinal, length=template['child_length'])
    old_question = old['values']['addressed_question'].format(assignment=payload['assignment'],
        address=old['values']['address']['separator'].join(payload['address']),
        question=old['values']['domain_template_choice'])
    assert field['question'] == old_question
    new_question = new['values']['addressed_question'].format(assignment=payload['assignment'],
        address=new['values']['address']['separator'].join(changed_payload['address']),
        question=new['values']['domain_template_choice'])
    old_state = old['encoded_state'].format(value=json.dumps(field['input'], **serialization))
    new_state = new['encoded_state'].format(value=json.dumps(changed['field']['input'], **serialization))
    assert old_state in original['rendered']
    changed['field']['state'] = field['state'].replace(old_state, new_state)
    changed['field']['question'] = new_question
    for message in changed['messages']:
        message['content'] = message['content'].replace(old_state, new_state).replace(old_question, new_question)
    changed['rendered'] = tokenizer.apply_chat_template(changed['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    assert tokenizer(original['rendered'], add_special_tokens=False)['input_ids'] == original['tokens']
    changed['tokens'] = tokenizer(changed['rendered'], add_special_tokens=False)['input_ids']
    changed['input_tokens'] = len(changed['tokens'])
    assert changed['input_tokens'] <= shared['model']['max_input_tokens']
    restored = unpack_references(changed['field']['input'], transport)
    restored['input']['address'] = payload['address']
    assert json.dumps(restored, sort_keys=True) == json.dumps(semantic, sort_keys=True)
    assert changed['field']['options'] == field['options']
    inputs.append(changed)
    rows.append({'node': node, 'path': payload['path'], 'stored_index': index,
                 'human_ordinal': ordinal, 'original_tokens': original['input_tokens'],
                 'candidate_tokens': changed['input_tokens'], 'options_preserved': True,
                 'complete_task_and_schema_preserved': True, 'source_references_resolve': True})
Path(config['inputs_output']).write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')
summary = {'scope': 'CPU prompt and source-reference qualification; no inference outputs or quality claim.',
           'original_prompts_unchanged': True, 'method_changed_only_prompt_keys_and_recording': True,
           'shared_config': spec['shared_config'], 'task_count': spec['task_count'], 'requests': rows}
Path(config['summary_output']).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False))
