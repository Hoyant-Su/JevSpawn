from itertools import product
import json
from pathlib import Path
import time

import jsonschema

from methods.finite_constraints.program import (
    assemble_factors, assignment_rows, distinct_factors, domains, execute,
)
from methods.finite_constraints.solver import propagate, solve

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


def validated_scopes(state, compilation, settings):
    scopes = json.loads(compilation['generation']['result']['texts'][0])
    jsonschema.validate(scopes, compilation['schema'])
    count = sum(len(state['positions']) ** len(scope) for scope in scopes)
    assert count <= settings['max_factor_entries'], 'Compiled scopes exceed the declared table budget.'
    return scopes, count


def instantiate_wave(state, scopes, current_domains, selected, prompts):
    original = domains(state)
    context = json.dumps({key: state[key] for key in ['positions', 'variables']})
    nodes, entries = [], []
    for factor in selected:
        scope = scopes[factor]
        tuples = list(product(*(current_domains[identifier] for identifier in scope)))
        entries.append({'factor_index': factor, 'scope': scope,
                        'tuples': [list(value) for value in tuples]})
        for value in tuples:
            index = 0
            for identifier, position in zip(scope, value):
                index = index * len(original[identifier]) + original[identifier].index(position)
            nodes.append({'id': f'c{factor}/a{index}', 'state': context,
                          'question': prompts['predicate_question'].format(
                              clue=state['clues'][factor], assignment=json.dumps(dict(zip(scope, value)))),
                          'options': prompts['predicate_options']})
    return {'entries': entries, 'nodes': nodes}


def adaptive_run(backend, state, settings, compilation):
    started = time.perf_counter()
    mode = settings['adaptive_execution_mode']
    assert mode in {'tiled_shared', 'tiled_independent'}
    scopes, potential_workers = validated_scopes(state, compilation, settings)
    prompts = load_prompt('configs/methods/finite_constraints/schema/prompts.json')
    original = domains(state)
    current = {identifier: list(values) for identifier, values in original.items()}
    factors = distinct_factors(state)
    remaining = list(range(len(scopes)))
    selected = [index for index in remaining if len(scopes[index]) == 1]
    selection = 'unary' if selected else 'seed'
    selected = selected or remaining[:1]
    completed, previous_nodes, waves, executions = [], [], [], []
    while selected:
        wave_started = time.perf_counter()
        before = {identifier: list(values) for identifier, values in current.items()}
        program = instantiate_wave(state, scopes, current, selected, prompts)
        scoring_started = time.perf_counter()
        execution = execute(backend, program, mode)
        scoring_finished = time.perf_counter()
        node_ids = [node['id'] for node in program['nodes']]
        decisions = execution['result']['groups'][0]
        assert [decision['id'] for decision in decisions] == node_ids
        added_factors = assemble_factors(program, execution)
        factors.extend(added_factors)
        propagation_started = time.perf_counter()
        propagation = propagate(current, factors)
        propagation_finished = time.perf_counter()
        propagation_seconds = propagation_finished - propagation_started
        current = propagation['domains']
        completed.extend(selected)
        remaining = [index for index in remaining if index not in selected]
        waves.append({'index': len(waves), 'selection': selection,
                      'factor_indices': list(selected), 'domains_before': before,
                      'domains_after': current, 'node_ids': node_ids,
                      'dependencies': list(previous_nodes),
                      'dependency_semantics': 'Every node in this wave depends conservatively on all listed earlier nodes. Dependencies control instantiation and do not alter the original local predicate prompt.',
                      'original_cartesian_entries': sum(len(state['positions']) ** len(scopes[index]) for index in selected),
                      'instantiated_entries': len(node_ids), 'entries': program['entries'],
                      'decisions': [{'id': decision['id'], 'choice': decision['choice']} for decision in decisions],
                      'allowed_factors': added_factors, 'propagation': propagation,
                      'execution_seconds': execution['elapsed_seconds'],
                      'propagation_seconds': propagation_seconds,
                      'host_wall_spans_seconds': {
                          'instantiation': [wave_started - started, scoring_started - started],
                          'scoring': [scoring_started - started, scoring_finished - started],
                          'factor_assembly': [scoring_finished - started, propagation_started - started],
                          'propagation': [propagation_started - started, propagation_finished - started]},
                      'elapsed_seconds': time.perf_counter() - wave_started})
        executions.append(execution)
        previous_nodes.extend(node_ids)
        if propagation['status'] == 'unsat':
            break
        selected = [index for index in remaining if any(
            len(current[identifier]) < len(original[identifier]) for identifier in scopes[index])]
        selection = 'reduced_scope' if selected else 'seed'
        selected = selected or remaining[:1]
    solver_started = time.perf_counter()
    solution = solve(current, factors, settings['max_solver_nodes'])
    solver_seconds = time.perf_counter() - solver_started
    prediction = assignment_rows(state, solution['assignment']) if solution['status'] == 'sat' else None
    return {'status': solution['status'], 'prediction_rows': prediction, 'scopes': scopes,
            'finite_workers': len(previous_nodes), 'potential_finite_workers': potential_workers,
            'worker_model_depth': len(executions), 'execution_records': executions,
            'execution_schedule': 'Synchronous scoring and propagation waves, without ready queue overlap.',
            'host_timeline_origin': 'Adaptive execution entry after compilation. Host wall spans do not measure GPU utilization.',
            'waves': waves, 'completed_factor_indices': completed,
            'unevaluated_factor_indices': remaining, 'final_domains': current,
            'solver': solution, 'solver_seconds': solver_seconds,
            'total_seconds': time.perf_counter() - started + compilation['elapsed_seconds'],
            'compilation': compilation}
