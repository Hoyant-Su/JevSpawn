import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.readout_labels import AdmittedPrompt, native_labels
from jev_spawn.schema import controller_prompts, CONTROLLER


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sources(settings):
    frontiers = []
    for source in settings['sources']:
        snapshot = json.loads(Path(source['snapshot']).read_text())
        available = {node['id']: node for group in snapshot['groups'] for node in group}
        calls = json.loads(Path(source['calls']).read_text())['calls']
        payloads = {call['node']: {**{key: call['input'][key] for key in source['leading_input_keys']
                    if key in call['input']}, **call['input']} for call in calls}
        for selection in source['field_sets']:
            fields = [available[identity] for identity in selection['ids']]
            headers = []
            for field in fields:
                encoded = json.dumps(payloads[field['id']], **source['serialization'])
                assert field['state'].endswith(encoded)
                headers.append(field['state'][:-len(encoded)])
            assert len(set(headers)) == 1
            frontiers.append({'id': selection['id'], 'fields': fields,
                'payloads': [payloads[field['id']] for field in fields], 'header': headers[0],
                'controller': snapshot['controller'], 'serialization': source['serialization']})
    return frontiers


def catalog(fields, payloads, labels):
    shared = {key: value for key, value in payloads[0].items()
              if all(key in payload and payload[key] == value for payload in payloads)}
    questions, options, entries = [], [], []
    for field, payload in zip(fields, payloads):
        question = field['question']
        menu = [{**option, 'label': label} for option, label in zip(field['options'], labels)]
        if question not in questions:
            questions.append(question)
        if menu not in options:
            options.append(menu)
        private = {key: value for key, value in payload.items() if key not in shared}
        assert {**shared, **private} == payload
        entries.append({'field': field['id'], 'question_index': questions.index(question),
            'options_index': options.index(menu), 'input': private})
    return {'shared_input': shared, 'questions': questions, 'option_sets': options, 'fields': entries}


def render(tokenizer, texts, system):
    return [tokenizer.apply_chat_template([{'role': 'system', 'content': system},
             {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True,
             enable_thinking=False) for text in texts]


def admission(backend, group, variant, templates, settings):
    controller = group['controller']
    CONTROLLER.clear()
    CONTROLLER.update(controller)
    if variant == 'reference':
        texts = [controller_prompts([field['state']], field['question'], field['options'],
                 list(backend.answer_labels[:len(field['options'])]), controller['output_instruction'])[0]
                 for field in group['fields']]
        system = controller['system']
        partial = controller['user_template'].split('{state}')[0] + group['fields'][0]['context']
    else:
        value = catalog(group['fields'], group['payloads'], backend.answer_labels)
        shared = templates['catalog'].format(header=group['header'],
            catalog=json.dumps(value, **settings['serialization']))
        texts = [templates['request'].format(catalog=shared, field=field['id'],
                instruction=controller['output_instruction']) for field in group['fields']]
        system = templates['system']
        partial = templates['catalog'].split('{header}')[0] + group['header']
    rendered = render(backend.tokenizer, texts, system)
    tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
    partial_text = backend.tokenizer.apply_chat_template([{'role':'system','content':system},
        {'role':'user','content':partial}], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    partial_tokens = backend.tokenizer(partial_text, add_special_tokens=False)['input_ids']
    base_length = common_prefix([partial_tokens, *tokens])
    assert tokens == group['layouts'][variant]['input_ids']
    assert max(map(len,tokens)) <= backend.config['max_input_tokens']
    return {'group':group, 'variant':variant, 'base_length':base_length,
            'admitted':[AdmittedPrompt(text,tuple(ids))for text,ids in zip(rendered,tokens)]}


def prepare(settings):
    shared = SharedConfig.load(settings['shared_config'])
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path)
    labels, native_ids = native_labels(tokenizer, load_resource('readout_labels'))
    templates = json.loads(Path(settings['templates']).read_text())
    groups = []
    for frontier in sources(settings):
        CONTROLLER.clear()
        CONTROLLER.update(frontier['controller'])
        for start in range(0, len(frontier['fields']), settings['catalog_group_size']):
            fields = frontier['fields'][start:start + settings['catalog_group_size']]
            payloads = frontier['payloads'][start:start + settings['catalog_group_size']]
            value = catalog(fields, payloads, labels)
            catalog_text = templates['catalog'].format(header=frontier['header'],
                catalog=json.dumps(value, **settings['serialization']))
            texts = [templates['request'].format(catalog=catalog_text, field=field['id'],
                        instruction=CONTROLLER['output_instruction']) for field in fields]
            candidate = render(tokenizer, texts, templates['system'])
            ordinary = [controller_prompts([field['state']], field['question'], field['options'],
                list(labels[:len(field['options'])]), CONTROLLER['output_instruction'])[0] for field in fields]
            reference = render(tokenizer, ordinary, CONTROLLER['system'])
            layouts = {}
            for name, rendered in [('reference', reference), ('catalog', candidate)]:
                ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
                prefix = common_prefix(ids)
                suffixes = [len(row)-prefix for row in ids]
                layouts[name] = {'rendered': rendered, 'input_ids': ids,
                    'input_tokens': [len(row) for row in ids], 'prefix_tokens': prefix,
                    'suffix_tokens': suffixes, 'cold_computed_tokens': prefix + sum(suffixes),
                    'cold_padded_tokens': prefix + max(suffixes)*len(ids),
                    'within_context': max(map(len, ids)) <= shared.model.max_input_tokens}
            groups.append({'frontier_id': frontier['id'], 'start': start, 'fields': fields,
                'catalog': value, 'payloads': payloads, 'header': frontier['header'],
                'controller': frontier['controller'], 'layouts': layouts})
    result = {'settings': settings, 'answer_labels': list(labels), 'answer_label_ids': list(native_ids),
        'groups': groups, 'all_fields_preserved': sum(len(x['fields'])for x in groups)==sum(len(x['fields'])for x in sources(settings)),
        'within_context': all(layout['within_context']for group in groups for layout in group['layouts'].values()),
        'scope': 'Changed conditioning; full exact recorded input dictionaries, questions, option descriptions and identities retained. No gold labels or recorded prediction values loaded for catalog construction. Explicit fixed logical field selections; original GPU arrival batches do not define catalog membership.'}
    output=Path(settings['feasibility_output']);output.parent.mkdir(parents=True,exist_ok=True)
    save(output,result)
    print(json.dumps({'within_context':result['within_context'],'all_fields_preserved':result['all_fields_preserved'],
        'groups':[{'frontier':g['frontier_id'],'ids':[f['id']for f in g['fields']],
        'layouts':{k:{x:v[x]for x in ['input_tokens','prefix_tokens','suffix_tokens','cold_computed_tokens','cold_padded_tokens','within_context']}for k,v in g['layouts'].items()}}for g in groups]},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    prepare(json.loads(args.config.read_text()))
