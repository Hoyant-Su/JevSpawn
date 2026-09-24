import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
import yaml

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.configuration import CORE, load_resource
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import native_labels


def transform(field, reference, candidate, serialization):
    changed = deepcopy(field)
    state = json.loads(field['state'])
    removed = []
    for location in [state, *([state['state']] if 'state' in state else [])]:
        if 'format' in location:
            assert location['format'] == reference['history_format']
            removed.append(location.pop('format'))
    assert removed
    changed['state'] = json.dumps(state, **serialization)
    old = reference['query_execution']['select_field']
    new = candidate['query_execution']['select_field']
    guidance = old.removeprefix(new).lstrip()
    assert old.startswith(new) and guidance in candidate['controller']['system']
    if field['question'].startswith(new.partition('{identity}')[0]):
        lead, tail = old.split('{identity}')
        assert field['question'].startswith(lead) and field['question'].endswith(tail)
        identity = field['question'][len(lead):-len(tail)]
        assert field['question'] == old.format(identity=identity)
        changed['question'] = new.format(identity=identity)
    assert all(text in candidate['controller']['system'] for text in removed)
    restored = json.loads(changed['state'])
    original = json.loads(field['state'])
    for left, right in [(restored, original), *([(restored['state'], original['state'])] if 'state' in original else [])]:
        if 'format' in right:
            left['format'] = right['format']
    assert restored == original
    assert all(changed[key] == field[key] for key in ('context', 'history', 'options'))
    return changed


def render(backend, controller, system, field, labels):
    prefix = controller['prefix_template'].format(context=field['context'], history=field['history'])
    menu = '\n'.join(controller['option_template'].format(label=label, **option)
                     for label, option in zip(labels, field['options'], strict=True))
    text = controller['user_template'].format(prefix=prefix, context=field['context'], state=field['state'],
        question=field['question'], menu=menu, output_instruction=controller['output_instruction'])
    rendered, = backend._render([text], system)
    root, = backend._render([controller['prefix_template'].format(context=field['context'], history='')], system)
    tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
    root_tokens = backend.tokenizer(root, add_special_tokens=False)['input_ids']
    cacheable = common_prefix([tokens, root_tokens])
    return tokens, {'total_tokens': len(tokens), 'root_tokens': cacheable,
                    'recomputed_tokens_after_root': len(tokens) - cacheable}


def main(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    reference = json.loads(Path(settings['reference']).read_text())
    candidate = {'controller': load_prompt('jevspawn.controller'), 'query_execution': load_prompt('jevspawn.query_execution')}
    inference = json.loads(Path(settings['inference_config']).read_text())
    option_template = load_prompt(inference['prompts'])['option_template']
    reference['controller']['option_template'] = option_template
    candidate['controller']['option_template'] = option_template
    serialization = json.loads(Path(settings['method']).read_text())['settings']['rollout']['execution']['serialization']
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], **settings['tokenizer'])
    backend = SimpleNamespace(tokenizer=tokenizer)
    source = ast.parse(Path(settings['backend_source']).read_text())
    definition, = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == settings['backend_class']]
    method, = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == settings['render_method']]
    namespace = {'CORE': CORE}
    exec(compile(ast.Module(body=[method], type_ignores=[]), settings['backend_source'], 'exec'), namespace)
    backend._render = namespace[settings['render_method']].__get__(backend)
    requests = []
    for path in settings['profile_sources']:
        for request in json.loads(Path(path).read_text())['requests']:
            requests.append((path, request))
    assert settings['trace_selection'] == 'first_model_field_per_task'
    for directory in settings['trace_runs']:
        run = Path(directory)
        runtime = json.loads((run / 'session-0000/runtime.json').read_text())['service']
        contract = '\n\n' + load_prompt('shared.runtime_contract').format(contract=json.dumps(runtime['runtime_contract']))
        for path in sorted(run.glob(settings['task_glob'])):
            task = json.loads(path.read_text())
            fields = [field for turn in task['trace']['rounds'] if 'parent_computations' in turn
                      for blocks in turn['parent_computations'].values() for block in blocks if 'requests' in block
                      for field, decision in zip(block['requests'], block['decisions'], strict=True) if decision['input_tokens']]
            if fields:
                requests.append((str(path), {'task_id': task['task_id'], 'field': fields[0],
                    'messages': [{'role': 'system', 'content': reference['controller']['system'] + contract}]}))
    labels, _ = native_labels(tokenizer, load_resource('readout_labels'))
    rows = []
    for source, request in requests:
        field = request['field']
        system, = [message['content'] for message in request['messages'] if message['role'] == 'system']
        assert system.startswith(reference['controller']['system'])
        suffix = system[len(reference['controller']['system']):]
        changed = transform(field, reference, candidate, serialization)
        selected_labels = labels[:len(field['options'])]
        original_tokens, original = render(backend, reference['controller'], system, field, selected_labels)
        _, proposed = render(backend, candidate['controller'], candidate['controller']['system'] + suffix, changed, selected_labels)
        if 'input_ids' in request:
            assert original_tokens == request['input_ids']
            assert original['root_tokens'] == len(request['root_tokens'])
        rows.append({'source': source, 'task_id': request['task_id'], 'field_id': field['id'],
            'field_guidance_relocated': changed['question'] != field['question'],
            'reference': original, 'candidate': proposed,
            'recomputed_tokens_saved': original['recomputed_tokens_after_root'] - proposed['recomputed_tokens_after_root'],
            'reference_token_ids_verified': 'input_ids' in request})
    totals = {name: {key: sum(row[name][key] for row in rows) for key in rows[0][name]}
              for name in ('reference', 'candidate')}
    report = {'settings': settings, 'all_context_history_options_preserved': True,
        'all_state_values_preserved_except_relocated_fixed_legend': True,
        'removed_legend_and_field_guidance_present_in_system': True,
        'scope': 'CPU layout/token accounting on actual recorded requests. Root reuse assumes a warm exact task/system prefix; no GPU speed or semantic-equivalence claim.',
        'totals': totals, 'rows': rows}
    Path(settings['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'requests': len(rows), 'field_requests': sum(row['field_guidance_relocated'] for row in rows), 'totals': totals}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
