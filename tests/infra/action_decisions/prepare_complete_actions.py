import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration, template_references


def unbound(value, bindings, settings):
    result = set()
    for identity in template_references(value, settings):
        if identity in bindings:
            result.update(unbound(bindings[identity], bindings, settings))
        else:
            result.add(identity)
    return result


def combinations(declaration, bindings, settings):
    pending = unbound(declaration['action'], bindings, settings)
    if not pending:
        yield bindings
        return
    field = next(field for field in declaration['fields'] if field['id'] in pending)
    for value in field['values']:
        yield from combinations(declaration, {**bindings, field['id']: value}, settings)


def prepare(config):
    prompts = load_prompt(config['prompts'])
    manifest = []
    for source in config['sources']:
        artifact = json.loads(Path(source['artifact']).read_text())
        protocol = json.loads(Path(source['protocol']).read_text())
        settings = protocol['method']['settings']['rollout']['execution']
        round_record = artifact['trace']['rounds'][source['round_index']]
        computation = round_record['parent_computations'][source['parent_id']][source['computation_index']]
        original = computation['requests'][source['request_index']]
        state = json.loads(original['state'])
        assert state['bound_fields'] == {} and state['computed_values'] == []
        raw = artifact['trace']['rounds'][source['declaration_round']]['revision']['declaration']
        declaration = compile_declaration(raw, settings)
        assert json.dumps(declaration, **settings['serialization']) in original['context']
        assert state['action_template'] == declaration['action']
        execution = QueryExecution(original['context'], None, artifact['task_id'], settings)
        candidates, requests = [], []
        for index, bindings in enumerate(combinations(declaration, {}, settings)):
            execution.values = bindings
            action = execution.materialize(declaration['action'])
            identity = prompts['request_id'].format(index=index)
            candidates.append({'id': identity, 'action': action, 'bindings': deepcopy(bindings)})
            requests.append({'id': identity, 'context': original['context'], 'state': original['state'],
                'question': prompts['question'].format(action=json.dumps(action)),
                'options': deepcopy(prompts['options'])})
        assert len(candidates) == len({json.dumps(row['action'], sort_keys=True) for row in candidates})
        fixture = {'task_id': artifact['task_id'], 'parent_id': source['parent_id'],
                   'declaration': declaration, 'candidates': candidates, 'requests': requests}
        Path(source['fixture_path']).write_text(json.dumps(fixture, indent=2) + '\n')
        manifest.append({**source, 'task_id': artifact['task_id'], 'request_count': len(requests),
            'pending_bindings_empty': True, 'source_context_and_state_unchanged': True,
            'candidate_source': 'Full reachable product of the actual model declaration, including inapplicable actions.'})
    result = {'scope': config['scope'], 'prompts': config['prompts'], 'fixtures': manifest,
              'total_requests': sum(row['request_count'] for row in manifest)}
    Path(config['manifest']).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = prepare(json.loads(Path(args.config).read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
