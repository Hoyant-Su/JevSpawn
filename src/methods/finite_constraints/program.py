from itertools import combinations, product
import json
from pathlib import Path
import re
import time

import jsonschema

from methods.generated_schema.run import measured_call
from methods.program_execution.grouped import score_grouped

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


def parse_puzzle(puzzle):
    preamble, clue_text = puzzle.split('## Clues:')
    bounds = re.search(r'There are (\d+) houses, numbered 1 to (\d+)', preamble)
    houses = int(bounds[1])
    assert houses == int(bounds[2])
    attributes, variables = [], []
    for line in preamble.splitlines():
        if not line.startswith(' - '):
            continue
        description, values_text = line[3:].split(':', 1)
        values = re.findall(r'`([^`]+)`', values_text)
        assert len(values) == houses and len(set(values)) == houses
        index = len(attributes)
        attribute = []
        for slot, value in enumerate(values):
            identifier = f'v{index}_{slot}'
            attribute.append(identifier)
            variables.append({'id': identifier, 'description': description,
                              'attribute_index': index, 'value': value})
        attributes.append(attribute)
    numbered = re.findall(r'^(\d+)\. (.+)$', clue_text, flags=re.MULTILINE)
    assert [int(number) for number, _ in numbered] == list(range(1, len(numbered) + 1))
    assert attributes and numbered
    return {'positions': list(range(1, houses + 1)), 'variables': variables,
            'attributes': attributes, 'clues': [text for _, text in numbered]}


def domains(state):
    return {variable['id']: state['positions'] for variable in state['variables']}


def distinct_factors(state):
    allowed = [list(values) for values in product(state['positions'], repeat=2) if values[0] != values[1]]
    return [{'scope': list(pair), 'allowed': allowed}
            for attribute in state['attributes'] for pair in combinations(attribute, 2)]


def assignment_rows(state, assignment):
    values = {variable['id']: variable['value'] for variable in state['variables']}
    return [[next(values[identifier] for identifier in attribute if assignment[identifier] == position)
             for attribute in state['attributes']] for position in state['positions']]


def compile_scopes(backend, state, settings):
    started = time.perf_counter()
    directory = ROOT / 'configs/methods/finite_constraints/schema'
    prompts = load_prompt('configs/methods/finite_constraints/schema/prompts.json')
    schema = json.loads((directory / 'scopes.json').read_text())
    schema.update(minItems=len(state['clues']), maxItems=len(state['clues']))
    schema['items']['items']['enum'] = list(domains(state))
    prompt = prompts['scope_user'].format(
        variables=json.dumps({key: state[key] for key in ['positions', 'variables']}),
        clues=json.dumps(list(enumerate(state['clues'], 1))), schema=json.dumps(schema))
    generation = measured_call(backend, lambda: backend.generate(
        [prompt], prompts['scope_system'], settings['compiler_tokens']), True)
    return {'generation': generation, 'schema': schema, 'prompt': prompt,
            'elapsed_seconds': time.perf_counter() - started}


def instantiate(state, compilation, settings):
    scopes = json.loads(compilation['generation']['result']['texts'][0])
    jsonschema.validate(scopes, compilation['schema'])
    count = sum(len(state['positions']) ** len(scope) for scope in scopes)
    assert count <= settings['max_factor_entries'], 'Compiled scopes exceed the declared table budget.'
    prompts = load_prompt('configs/methods/finite_constraints/schema/prompts.json')
    context = json.dumps({key: state[key] for key in ['positions', 'variables']})
    nodes, entries = [], []
    for factor, (clue, scope) in enumerate(zip(state['clues'], scopes)):
        tuples = list(product(*(domains(state)[identifier] for identifier in scope)))
        entries.append({'scope': scope, 'tuples': [list(value) for value in tuples]})
        for index, value in enumerate(tuples):
            nodes.append({'id': f'c{factor}/a{index}', 'state': context,
                          'question': prompts['predicate_question'].format(
                              clue=clue, assignment=json.dumps(dict(zip(scope, value)))),
                          'options': prompts['predicate_options']})
    return {'scopes': scopes, 'entries': entries, 'nodes': nodes}


def execute(backend, program, mode):
    return measured_call(backend, lambda: score_grouped(backend, [program['nodes']], mode))


def assemble_factors(program, execution):
    decisions = execution['result']['groups'][0]
    factors, offset = [], 0
    for entry in program['entries']:
        stop = offset + len(entry['tuples'])
        factors.append({'scope': entry['scope'],
                        'allowed': [value for value, decision in zip(entry['tuples'], decisions[offset:stop])
                                    if decision['choice'] == 'satisfied']})
        offset = stop
    assert offset == len(decisions)
    return factors
