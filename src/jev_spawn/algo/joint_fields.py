from itertools import product
import json
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from jev_spawn.runtime.branches import required_fields
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.runtime.actions import (
    dispatch, field_layers, merge_extensions, prepare_actions, substitute_bound
)


def next_block(execution, declaration, layers, capacity):
    fields = {field['id']: field for field in declaration['fields']}
    pending = required_fields(execution, declaration['action'], fields) - execution.bound_fields.keys()
    for layer in layers:
        block, size = [], 1
        for field in layer:
            if field['id'] not in pending:
                continue
            expanded = size * len(field['values'])
            if block and expanded > capacity:
                break
            block.append(field)
            size = expanded
        if block:
            return block
    return []


def assignments(fields):
    return [dict(zip((field['id'] for field in fields), values, strict=True))
            for values in product(*(field['values'] for field in fields))]


def candidates_for(execution, declaration, layers, capacity):
    block = next_block(execution, declaration, layers, capacity)
    return block, assignments(block)


def request_for(execution, identity, block, choices, prompt):
    _, requests, evidence = prepare_actions(QueryExecution.prepare, execution, block)
    request = requests[0]
    if len(block) > 1:
        request['question'] = prompt
        request['options'] = [
            {'id': execution.settings['candidate_id'].format(index=index),
             'description': json.dumps({'bindings': bindings,
                 'resulting_action': substitute_bound(execution.action_template,
                     {**execution.bound_fields, **bindings})}, **execution.settings['serialization'])}
            for index, bindings in enumerate(choices)]
    request['value_fields'] = [{'id': field['id'], 'question': field['question']} for field in block]
    request['readout_prompt'] = execution.settings['action_prompt']
    for option, bindings in zip(request['options'], choices, strict=True):
        option['values'] = substitute_bound(execution.action_template, {**execution.bound_fields, **bindings})
    ordered = sorted(zip(request['options'], choices, strict=True),
        key=lambda pair: json.dumps(pair[1], **execution.settings['action_identity_serialization']))
    request['options'] = [option for option, _ in ordered]
    choices[:] = [bindings for _, bindings in ordered]
    request['id'] = json.dumps([identity, [field['id'] for field in block]])
    return request, evidence


def rank_joint(distributions, prefix_scores, width):
    """Rank conditional extensions by cumulative action probability."""
    probabilities = pad_sequence(distributions, batch_first=True)
    scores = prefix_scores[:, None] + probabilities.log()
    lengths = torch.tensor([len(row) for row in distributions], device=scores.device)
    valid = torch.arange(scores.shape[-1], device=scores.device)[None, :] < lengths[:, None]
    flat = scores.flatten()
    available = valid.flatten().nonzero().flatten()
    selected = available[flat[available].argsort(descending=True, stable=True)[:width]]
    return (selected // scores.shape[-1]).tolist(), (selected % scores.shape[-1]).tolist(), flat[selected]


def spawn_blocks(branches, branch_width, workers, identities, *, prompt):
    started = time.perf_counter()
    source = next(iter(branches.values())).execution
    capacity = len(source.service.backend.answer_labels)
    layers, beams, scores = {}, {}, {}
    for owner, parent in branches.items():
        execution = parent.execution
        layers[owner] = field_layers(parent.declaration, execution.settings)
        execution.values.clear()
        execution.bound_fields.clear()
        execution.field_evidence.clear()
        execution.action_template = parent.declaration['action']
        execution.fields.update({field['id']: field for field in parent.declaration['fields']})
        beams[owner] = [(owner, execution.fork())]
        scores[owner] = execution.service.initial_scores()
    rounds, readouts, joint_readouts = 0, 0, 0
    while True:
        pending, requests = {}, []
        for owner, beam in beams.items():
            for index, (identity, execution) in enumerate(beam):
                block, choices = candidates_for(execution, branches[owner].declaration, layers[owner], capacity)
                if not block:
                    continue
                request, evidence = request_for(execution, identity, block, choices, prompt)
                pending[owner, index] = (request, choices, evidence)
                if len(choices) > 1:
                    requests.append(request)
                    joint_readouts += len(block) > 1
        if not pending:
            break
        decisions = source.field_readout(source.service, requests, source.task_id,
                                         source.settings['field_readout'])
        by_id = {decision['id']: decision for decision in decisions}
        readouts += len(requests)
        rounds += bool(requests)
        for owner, beam in beams.items():
            active = [index for index in range(len(beam)) if (owner, index) in pending]
            if not active:
                continue
            distributions = []
            for index in active:
                request, choices, evidence = pending[owner, index]
                if len(choices) > 1:
                    distributions.append(source.service.field_distributions.pop((source.task_id, request['id'])))
                    branches[owner].execution.trace.append({'joint_field_request': request,
                        'decision': by_id[request['id']], 'based_on_feedback': evidence})
                else:
                    distributions.append(scores[owner].new_ones((len(choices),)))
            indices = torch.tensor(active, device=scores[owner].device)
            selected, choices, extensions = rank_joint(
                distributions, scores[owner].index_select(0, indices), branch_width)
            candidates, scores[owner] = merge_extensions(
                beam, active, selected, choices, extensions, scores[owner], branch_width)
            next_beam = []
            for index, choice in candidates:
                identity, execution = beam[index]
                if choice is None:
                    next_beam.append((identity, execution))
                    continue
                request, choices, evidence = pending[owner, index]
                bindings = choices[choice]
                child = execution.fork()
                child.values.update(bindings)
                child.bound_fields.update(bindings)
                child.field_evidence.update({field: evidence for field in bindings})
                next_beam.append((json.dumps([identity, bindings]), child))
            beams[owner] = next_beam
    prepared = []
    for owner, beam in beams.items():
        for (_, execution), score in zip(beam, scores[owner].tolist(), strict=True):
            action = execution.materialize(branches[owner].declaration['action'])
            key = json.dumps([owner, action['tool'], action['arguments']],
                             **execution.settings['action_identity_serialization'])
            prepared.append((key, owner, execution, score))
    children = dispatch(branches, prepared, workers, identities,
        {'score_seconds': time.perf_counter() - started, 'readout_layers': rounds,
         'readout_requests': readouts, 'joint_block_requests': joint_readouts})
    for child in children.values():
        record = child.execution.trace[-1]
        record['block_conditional_log_probability'] = record.pop('layer_factored_log_probability')
    return children
