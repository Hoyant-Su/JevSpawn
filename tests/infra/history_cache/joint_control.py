import json


def select_control(frontier, query, task_id, service, settings, prompts, remaining, selection_only,
                   history, state, branch_records, record):
    candidates = {}
    for identity, branch in frontier.items():
        if selection_only or branch.environment.done:
            operations = [settings['submit_operation']]
        elif branch.declaration is None or any(
                not feedback['accepted'] for feedback in branch.execution.declaration_feedback[-1:]):
            operations = [settings['revise_operation']]
        else:
            operations = [settings['expand_operation'], settings['revise_operation'], settings['submit_operation']]
        for operation in operations:
            candidates[json.dumps([identity, operation], **settings['execution']['serialization'])] = (identity, operation)
    request = {'id': settings['frontier_id'], 'context': query, 'history': history,
        'state': json.dumps({**state, 'branches': branch_records}, **settings['execution']['serialization']),
        'question': prompts['joint_control'].format(remaining=remaining),
        'options': [{'id': key, 'description': json.dumps({'branch': identity, 'operation': operation},
                     **settings['execution']['serialization'])}
                    for key, (identity, operation) in candidates.items()]}
    decision, = service.decide([request], task_id=task_id)
    selected, operation = candidates[decision['choice']]
    ranked = list(dict.fromkeys(candidates[key][0] for key in decision['ranked_option_ids']))
    record.update(control_request=request, control_decision=decision,
                  selected=[selected], selected_operation=operation)
    return ranked, selected, operation
