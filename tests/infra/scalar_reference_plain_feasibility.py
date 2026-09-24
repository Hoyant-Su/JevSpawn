import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer
import yaml

from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import load_prompt, resolve_prompts
from jev_spawn.infra.readout_labels import native_labels, validate_boundaries
from jev_spawn.runtime.reference_transport import pack_references, unpack_references
from jev_spawn.schema import CONTROLLER, controller_prompts


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def main(config):
    run = Path(config['source_run'])
    task = json.loads((run / config['task_file']).read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    shared = yaml.safe_load(Path(config['shared_config']).read_text())
    transport = load_resource('reference_transport')
    serialization = load_resource('finite_control')['serialization']
    prompts = resolve_prompts(load_prompt(config['control_prompts']))
    values = resolve_prompts(load_prompt(config['value_prompts']))
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)
    labels, label_ids = native_labels(tokenizer, load_resource('readout_labels'))
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    calls = {call['node']: call for call in task['calls']}
    pairs = []
    for path in sorted((run / config['input_directory']).glob(config['input_glob'])):
        cohort = json.loads(path.read_text())
        for recorded in cohort['requests']:
            field = recorded['field']
            if recorded['task_id'] != task['task_id'] or not field['question'].endswith(values['scalar_source']):
                continue
            call = calls[field['id']]
            semantic = unpack_references(field['input'], transport)
            assert canonical(call['input']) == canonical(field['input'])
            assert canonical(semantic['candidates']) == canonical([
                {'id': option['id'], 'value': value} for option, value in
                zip(field['options'], field['candidate_values'], strict=True)])
            assert canonical(unpack_references(pack_references(semantic, transport), transport)) == canonical(semantic)
            plain_input = {transport['dictionary_key']: [], transport['root_key']: semantic}
            assert canonical(unpack_references(plain_input, transport)) == canonical(semantic)
            packed_text = prompts['encoded_state'].format(value=json.dumps(field['input'], **serialization))
            plain_text = prompts['encoded_state'].format(value=json.dumps(plain_input, **serialization))
            assert field['state'].endswith(packed_text)
            state = field['state'].removesuffix(packed_text) + plain_text
            options = [{'id': option['id'], 'description': json.dumps(value, ensure_ascii=False)}
                       for option, value in zip(field['options'], field['candidate_values'], strict=True)]
            candidate = {**field, 'input': plain_input, 'state': state, 'options': options}
            variants = {}
            for name, current in [('packed', field), ('plain', candidate)]:
                user, = controller_prompts([current['state']], current['question'], current['options'],
                                          labels[:len(options)], CONTROLLER['output_instruction'])
                messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=shared['generation']['enable_thinking'])
                tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
                validate_boundaries(tokenizer, [rendered], [tokens], labels, label_ids, [len(options)])
                variants[name] = {'field': current, 'messages': messages, 'rendered': rendered,
                                  'tokens': tokens, 'input_tokens': len(tokens)}
            assert variants['packed']['messages'] == recorded['messages']
            assert variants['packed']['rendered'] == recorded['rendered']
            assert variants['packed']['tokens'] == recorded['tokens']
            pairs.append({'source_cohort': cohort['cohort'], 'task_id': recorded['task_id'],
                          'node': field['id'], 'semantic_input': semantic, 'variants': variants,
                          'labels': labels[:len(options)], 'label_ids': label_ids[:len(options)],
                          'original_result': call['result']})
    assert len(pairs) == config['expected_requests']
    totals = {name: {'total': sum(p['variants'][name]['input_tokens'] for p in pairs),
                    'minimum': min(p['variants'][name]['input_tokens'] for p in pairs),
                    'maximum': max(p['variants'][name]['input_tokens'] for p in pairs),
                    'counts': dict(Counter(p['variants'][name]['input_tokens'] for p in pairs))}
              for name in ['packed', 'plain']}
    overflow = [{'node': p['node'], 'tokens': p['variants']['plain']['input_tokens']} for p in pairs
                if p['variants']['plain']['input_tokens'] > shared['model']['max_input_tokens']]
    report = {'config': config, 'requests': len(pairs), 'exact_semantic_equality': True,
              'original_admission_exact': True, 'candidate_ids_values_order_exact': True,
              'scalar_schema_address_assignment_unchanged': True, 'native_label_boundaries_exact': True,
              'input_tokens': totals, 'max_input_tokens': shared['model']['max_input_tokens'],
              'overflow': overflow, 'gpu_executed': False,
              'interpretation': 'Feasibility only. Fully expanded values and direct menus change conditioning, not semantic inputs; no causal or quality conclusion.'}
    Path(config['record']).write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + '\n')
    Path(config['result']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
