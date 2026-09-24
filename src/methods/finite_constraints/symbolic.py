from itertools import product
import json
import operator
from pathlib import Path
import time

import jsonschema

from methods.generated_schema.run import measured_call

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


OPERATORS = {'add': operator.add, 'sub': operator.sub, 'abs': abs,
             'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt,
             'le': operator.le, 'gt': operator.gt, 'ge': operator.ge,
             'and': lambda *values: all(values), 'or': lambda *values: any(values),
             'not': operator.not_}


class SymbolicCompilationError(ValueError):
    def __init__(self, message, generation, elapsed_seconds):
        super().__init__(message)
        self.generation = generation
        self.elapsed_seconds = elapsed_seconds


def references(expression):
    if type(expression) is int:
        return set()
    if expression['op'] == 'var':
        return {expression['id']}
    return set().union(*(references(argument) for argument in expression['args']))


def interpret(expression, assignment):
    if type(expression) is int:
        return expression
    if expression['op'] == 'var':
        return assignment[expression['id']]
    values = [interpret(argument, assignment) for argument in expression['args']]
    return OPERATORS[expression['op']](*values)


def build_factors(state, expressions, max_factor_entries):
    schema = resolve_prompts(json.loads((ROOT / 'configs/methods/finite_constraints/schema/symbolic.json').read_text()))
    jsonschema.validate({'expressions': expressions}, schema)
    if len(expressions) != len(state['clues']):
        raise ValueError('Compilation must contain one expression per original clue, in original order.')
    variable_ids = [variable['id'] for variable in state['variables']]
    positions = state['positions']
    assert positions and all(type(value) is int for value in positions)
    assert len(positions) == len(set(positions))
    assert len(variable_ids) == len(set(variable_ids))
    assert max_factor_entries > 0
    scopes = []
    for expression in expressions:
        used = references(expression)
        unknown = used - set(variable_ids)
        if unknown:
            raise ValueError(f'Unknown variable references, {sorted(unknown)}.')
        scopes.append([identifier for identifier in variable_ids if identifier in used])
    entries = sum(len(positions) ** len(scope) for scope in scopes)
    if entries > max_factor_entries:
        raise ValueError(f'Constraint tables require {entries} entries, exceeding the declared {max_factor_entries}.')
    return [{'scope': scope, 'allowed': [list(values) for values in product(positions, repeat=len(scope))
             if interpret(expression, dict(zip(scope, values)))]}
            for expression, scope in zip(expressions, scopes)]


def compile_symbolic(backend, state, settings):
    started = time.perf_counter()
    directory = ROOT / 'configs/methods/finite_constraints/schema'
    schema = json.loads((directory / 'symbolic.json').read_text())
    prompts = load_prompt('configs/methods/finite_constraints/schema/symbolic_prompts.json')
    source = {key: state[key] for key in ('positions', 'variables', 'clues')}
    user = prompts['user'].format(state=json.dumps(source, ensure_ascii=False),
                                  schema=json.dumps(schema, separators=(',', ':')))
    generation = measured_call(backend, lambda: backend.generate(
        [user], prompts['system'], settings['compiler_tokens']), record_tokens=True)
    postprocess = time.perf_counter()
    try:
        parsed = json.loads(generation['result']['texts'][0])
        jsonschema.validate(parsed, schema)
        expressions = parsed['expressions']
        factors = build_factors(source, expressions, settings['max_factor_entries'])
    except (ValueError, jsonschema.ValidationError) as error:
        raise SymbolicCompilationError(str(error), generation, time.perf_counter() - started) from error
    return {'generation': generation, 'expressions': expressions, 'factors': factors,
            'postprocess_seconds': time.perf_counter() - postprocess,
            'elapsed_seconds': time.perf_counter() - started}
