from itertools import count
import json
import time

import jsonschema

from jev_spawn.infra.prompts import ROOT, load_prompt
from jev_spawn.runtime.answer import finalize_answer
from jev_spawn.runtime.branches import Branch, spawn_scored
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.runtime.state import extend_event_history, history_frontier, history_state, pack_state
from jev_spawn.schema.declaration import compile_declaration, declaration_contract
from jev_spawn.schema.declaration_builder import DeclarationBuilder


def branch_state(branch):
    state = branch.execution.state()
    state.pop('execution_events')
    state['declaration_feedback'] = [{key: value for key, value in item.items() if key != 'declaration'}
                                     for item in state['declaration_feedback']]
    return state


def revise(branch, query, task_id, service, settings, prompts, budget, record):
    builder_settings = json.loads((ROOT / settings['builder']).read_text())
    feedback = {'operation': settings['revise_operation'],
        'execution': pack_state(branch.execution.state()), 'active_declaration': branch.declaration,
        'current_action_observations': branch.execution.latest_feedback}
    builder = DeclarationBuilder(query, branch.environment.display_tool_interface(True),
        branch.environment.display_answer_schema(), service, task_id, budget, builder_settings,
        load_prompt(builder_settings['prompts']), settings['execution'], record, feedback=feedback)
    contract = declaration_contract(settings['declaration_schema'], len(service.backend.answer_labels))
    try:
        declaration = builder.build_action()
        jsonschema.validate(declaration, contract)
        compiled = compile_declaration(declaration, settings['execution'])
    except (SyntaxError, ValueError, AssertionError, jsonschema.ValidationError) as error:
        observation = {'accepted': False, 'error_type': type(error).__name__, 'error': str(error),
                       'proposed_declaration': builder.program}
    else:
        branch.declaration = compiled
        branch.execution.active_declaration = compiled
        branch.execution.values.clear()
        branch.execution.fields.clear()
        branch.execution.bound_fields.clear()
        branch.execution.field_evidence.clear()
        branch.execution.action_template = compiled['action']
        observation = {'accepted': True, 'declaration': compiled}
    observation['source_signatures'] = [output['text'] for call in builder.calls
        if call['kind'] == 'signature' for output in call['outputs']]
    branch.execution.declaration_feedback.append(observation)
    record['observation'] = observation


def select_control(frontier, query, task_id, service, settings, prompts, remaining, selection_only,
                   history, state, branch_records, record):
    request = {'id': settings['frontier_id'], 'context': query, 'history': history,
        'state': json.dumps(state, **settings['execution']['serialization']),
        'question': prompts['frontier'].format(remaining=remaining),
        'options': [{'id': identity, 'description': json.dumps(branch_records[identity],
            **settings['execution']['serialization'])} for identity in frontier]}
    branch_decision, = service.decide([request], task_id=task_id)
    ranked = branch_decision['ranked_option_ids']
    selected_terminal = next(iter(ranked))
    branch = frontier[selected_terminal]
    if selection_only or branch.environment.done:
        operations = [settings['submit_operation']]
    elif branch.declaration is None or any(
            not feedback['accepted'] for feedback in branch.execution.declaration_feedback[-1:]):
        operations = [settings['revise_operation']]
    else:
        operations = [settings['expand_operation'], settings['revise_operation'], settings['submit_operation']]
    operation_request = {'id': settings['operation_id'], 'context': query, 'history': history,
        'state': json.dumps({'selected_branch': selected_terminal,
            'state': history_state(branch.execution.state()), 'declaration': branch.declaration},
            **settings['execution']['serialization']),
        'question': prompts['operation'].format(remaining=remaining) + prompts['selected_feedback'].format(
            branch=selected_terminal, feedback=json.dumps(branch.execution.latest_feedback,
                **settings['execution']['serialization'])),
        'options': [{'id': operation, 'description': prompts['operations'][operation]} for operation in operations]}
    operation_decision, = service.decide([operation_request], task_id=task_id)
    operation = operation_decision['choice']
    record.update(frontier_request=request, frontier_decision=branch_decision,
        operation_request=operation_request, operation_decision=operation_decision,
        selected=[selected_terminal], selected_operation=operation)
    return ranked, selected_terminal, operation


def solve(query, *, task_id, service, complete, settings, prompts, budget, trace, environment):
    started = time.perf_counter()
    root = QueryExecution(query, service, task_id, settings['execution'])
    frontier = {settings['root_id']: Branch(root, environment, None)}
    events, tool_timings = {}, []
    history = ''
    declarations, declaration_ids = {}, {id(None): None}
    identities = (settings['node_id'].format(index=index) for index in count(settings['initial_node_index']))
    trace.update(rounds=[])
    for turn in count():
        remaining = budget['max_turns'] - turn
        selection_only = turn == budget['max_turns']
        record = {'turn': turn, 'remaining_control_rounds': remaining, 'selection_only': selection_only}
        trace['rounds'].append(record)
        history_events = tuple(events.values())
        for branch in frontier.values():
            branch.execution.history = history
            branch.execution.history_events = history_events
        state, branch_records = history_frontier({'execution_events': list(events.values()), 'declarations': declarations,
                 'branches': {identity: {'state': branch_state(branch),
                     'declaration_id': declaration_ids[id(branch.declaration)], 'done': branch.environment.done}
                     for identity, branch in frontier.items()}})
        ranked, selected_terminal, operation = select_control(
            frontier, query, task_id, service, settings, prompts, remaining, selection_only,
            history, state, branch_records, record)
        branch = frontier[selected_terminal]
        if operation == settings['revise_operation']:
            record['revision'] = {}
            revise(branch, query, task_id, service, settings, prompts, budget, record['revision'])
            if record['revision']['observation']['accepted']:
                identity = settings['declaration_id'].format(turn=turn)
                declarations[identity] = branch.declaration
                declaration_ids[id(branch.declaration)] = identity
            if branch.declaration is None:
                continue
        if operation == settings['submit_operation']:
            if not branch.environment.done:
                submission = {'branch_id': selected_terminal,
                    'history_event_ids': [event['id'] for event in branch.execution.tool_observations]}
                record['submission'] = submission
                arguments = finalize_answer(query, branch.execution.state(),
                    branch.environment.display_answer_schema(), service, task_id, budget,
                    settings['terminal_answer'], submission)
                submission['resolved_arguments'] = arguments
                previous_timings = len(branch.environment.tool_timings)
                submission['feedback'] = branch.environment.observe(settings['submission_tool'], arguments)
                tool_timings.extend(branch.environment.tool_timings[previous_timings:])
                if not branch.environment.done or branch.environment.answer is None:
                    environment.tool_timings = tool_timings
                    return {'answer': None, 'elapsed_seconds': time.perf_counter() - started,
                            'termination': settings['submission_rejected'], 'selected_branch': selected_terminal}
            environment.answer, environment.done = branch.environment.answer, branch.environment.done
            environment.tool_timings = tool_timings
            return {'answer': environment.answer, 'elapsed_seconds': time.perf_counter() - started,
                'selected_terminal': selected_terminal,
                'terminal_branches': {identity: item.environment.answer for identity, item in frontier.items()
                                     if item.environment.done}}
        selected = [identity for identity in ranked if frontier[identity].declaration is not None
                    and not frontier[identity].environment.done][:settings['parent_width']]
        parents = {identity: frontier.pop(identity) for identity in selected}
        children = spawn_scored(parents, settings['branch_width'], settings['host_workers'], identities)
        record.update(selected=selected, parents=list(parents),
            parent_computations={identity: list(parent.execution.trace) for identity, parent in parents.items()},
            children={identity: list(child.execution.trace) for identity, child in children.items()})
        tool_timings.extend(timing for child in children.values() for timing in child.execution.trace[-2]['tool_timings'])
        observations = [event for child in children.values() for event in child.execution.latest_feedback]
        history += extend_event_history(observations, events.values())
        events.update({event['id']: event for event in observations})
        frontier.update(children)
