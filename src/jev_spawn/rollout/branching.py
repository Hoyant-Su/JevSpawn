from contextlib import contextmanager
from itertools import count
import json
import time
from traceback import format_exception_only

import jsonschema

from jev_spawn.infra.prompts import ROOT, load_prompt
from jev_spawn.runtime.answer import finalize_answer
from jev_spawn.runtime.branches import Branch
from jev_spawn.algo.joint_fields import spawn_blocks
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.runtime.state import extend_event_history, history_state, pack_state
from jev_spawn.schema.declaration import compile_declaration, declaration_contract
from jev_spawn.schema.declaration_builder import DeclarationBuilder


class Policy:
    def __init__(self, settings):
        self.settings = settings

    def history(self, branch, history, events):
        branch.execution.history = history
        branch.execution.history_events = events

    def rank(self, frontier, request, service, task_id):
        result, = service.decide([request], task_id=task_id)
        return result

    def operations(self, branch, selection_only):
        return control_options(branch, self.settings, selection_only)

    def operation_question(self, question):
        return question

    def width(self, parents):
        return self.settings['branch_width']

    def observe(self, children):
        pass


def revise(branch, query, task_id, service, settings, prompts, budget, record):
    builder_settings = json.loads((ROOT / settings['builder']).read_text())
    feedback = {'operation': settings['revise_operation'],
        'execution': pack_state(branch.execution.state()), 'active_declaration': branch.declaration,
        'current_action_observations': branch.execution.latest_feedback,
        'declaration_feedback': branch.execution.declaration_feedback[-1:]}
    builder = DeclarationBuilder(query, branch.environment.display_tool_interface(True),
        branch.environment.display_answer_schema(), service, task_id, budget, builder_settings,
        load_prompt(builder_settings['prompts']), settings['execution'], record, feedback=feedback)
    contract = declaration_contract(settings['declaration_schema'], len(service.backend.answer_labels))
    try:
        declaration = builder.build_action()
        jsonschema.validate(declaration, contract)
        compiled = compile_declaration(declaration, settings['execution'])
    except (SyntaxError, ValueError, AssertionError, jsonschema.ValidationError) as error:
        observation = {'accepted': False, 'error_type': type(error).__name__,
                       'error': ''.join(format_exception_only(error)).strip(),
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
                   history, record, policy):
    record.update(active_frontier=list(frontier))
    observed = {identity: {'return_value': index,
        'current_path': [event['id'] for event in branch.execution.tool_observations],
        'done': branch.environment.done, 'execution_events': branch.execution.latest_feedback}
        for index, (identity, branch) in enumerate(frontier.items(), start=settings['frontier_index_start'])}
    request = {'id': settings['frontier_id'], 'context': query, 'history': history,
        'state': json.dumps({'branches': observed, 'format': prompts['observations']},
            **settings['execution']['action_identity_serialization']),
        'question': prompts['frontier'].format(remaining=remaining) + prompts['retention'].format(
            width=settings['retained_frontier_width']),
        'options': [{'id': identity, 'values': [observed[identity]['return_value']], 'description': json.dumps(observed[identity],
            **settings['execution']['action_identity_serialization'])} for identity in frontier]}
    record['frontier_request'] = request
    branch_decision = policy.rank(frontier, request, service, task_id)
    ranked = branch_decision['ranked_option_ids']
    record['ranked_compared'] = ranked
    selected_terminal = next(iter(ranked))
    branch = frontier[selected_terminal]
    operations = policy.operations(branch, selection_only)
    operation_request = {'id': settings['operation_id'], 'context': query, 'history': branch.execution.history,
        'state': json.dumps({'selected_branch': selected_terminal,
            'state': history_state(branch.execution.state()), 'declaration': branch.declaration,
            'operations': {operation: prompts['operations'][operation] for operation in operations}},
            **settings['execution']['action_identity_serialization']),
        'question': policy.operation_question(prompts['operation'].format(remaining=remaining)) + prompts['selected_feedback'].format(
            branch=selected_terminal, feedback=json.dumps(branch.execution.latest_feedback,
                **settings['execution']['serialization'])),
        'options': [{'id': operation, 'values': [operation], 'description': prompts['operations'][operation]}
                    for operation in operations]}
    operation_decision, = service.decide([operation_request], task_id=task_id)
    operation = operation_decision['choice']
    record.update(frontier_request=request, frontier_decision=branch_decision,
        operation_request=operation_request, operation_decision=operation_decision,
        selected=[selected_terminal], selected_operation=operation)
    return ranked, selected_terminal, operation


def control_options(branch, settings, selection_only):
    if selection_only:
        return [settings['submit_operation']]
    if branch.environment.done:
        return [settings['submit_operation'], settings['discard_operation']]
    if branch.declaration is None or any(
            not feedback['accepted'] for feedback in branch.execution.declaration_feedback[-1:]):
        return [settings['revise_operation']]
    return [settings['expand_operation'], settings['revise_operation'], settings['submit_operation']]


@contextmanager
def report_turn(record, callback):
    yield
    if callback is not None:
        callback(record)


def solve(query, *, task_id, service, complete, settings, prompts, budget, trace, environment, on_turn=None):
    started = time.perf_counter()
    policy = Policy(settings)
    root = QueryExecution(query, service, task_id, settings['execution'])
    frontier = {settings['root_id']: Branch(root, environment, None)}
    events, tool_timings = {}, []
    history = ''
    identities = (settings['node_id'].format(index=index) for index in count(settings['initial_node_index']))
    trace.update(rounds=[])
    for turn in count():
        remaining = budget['max_turns'] - turn
        selection_only = turn == budget['max_turns']
        record = {'turn': turn, 'remaining_control_rounds': remaining, 'selection_only': selection_only}
        trace['rounds'].append(record)
        with report_turn(record, on_turn):
            history_events = tuple(events.values())
            all_branches = frontier
            for branch in all_branches.values():
                policy.history(branch, history, history_events)
            ranked, selected_terminal, operation = select_control(
                frontier, query, task_id, service, settings, prompts, remaining, selection_only,
                history, record, policy)
            record['pruned_frontier'] = ranked[settings['retained_frontier_width']:]
            frontier = {identity: frontier[identity] for identity in ranked[:settings['retained_frontier_width']]}
            branch = frontier[selected_terminal]
            if operation == settings['discard_operation']:
                frontier.pop(selected_terminal)
                if not frontier:
                    environment.tool_timings = tool_timings
                    return {'answer': None, 'elapsed_seconds': time.perf_counter() - started,
                            'termination': settings['frontier_exhausted']}
                continue
            if operation == settings['revise_operation']:
                record['revision'] = {}
                revise(branch, query, task_id, service, settings, prompts, budget, record['revision'])
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
            selected = [identity for identity in frontier if frontier[identity].declaration is not None
                        and not frontier[identity].environment.done][:settings['parent_width']]
            parents = {identity: frontier.pop(identity) for identity in selected}
            record['retained_frontier'] = list(frontier)
            children = spawn_blocks(parents, policy.width(parents), settings['host_workers'], identities,
                                    prompt=load_prompt(settings['joint_fields_prompt'])['question'])
            policy.observe(children)
            record.update(selected=selected, parents=list(parents),
                parent_computations={identity: list(parent.execution.trace) for identity, parent in parents.items()},
                children={identity: list(child.execution.trace) for identity, child in children.items()})
            tool_timings.extend(timing for child in children.values() for timing in child.execution.trace[-2]['tool_timings'])
            observations = [event for child in children.values() for event in child.execution.latest_feedback]
            history += extend_event_history(observations, events.values())
            events.update({event['id']: event for event in observations})
            frontier.update(children)
