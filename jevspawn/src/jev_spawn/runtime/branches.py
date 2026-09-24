from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time

import torch

from jev_spawn.schema.declaration import template_references
from jev_spawn.runtime.query_execution import QueryExecution


@dataclass
class Branch:
    execution: QueryExecution
    environment: object
    declaration: dict | None


def spawn(branches, proposals, workers):
    """Execute selected actions on independent snapshots of their declared parents."""
    unique = {}
    for proposal in proposals:
        execution = branches[proposal['parent']].execution.fork()
        execution.values.update(proposal['values'])
        action = execution.materialize(proposal['action'])
        signature = json.dumps([proposal['parent'], action['tool'], action['arguments']],
                               **execution.settings['action_identity_serialization'])
        unique.setdefault(signature, (proposal, execution))

    def execute(prepared):
        proposal, execution = prepared
        parent = branches[proposal['parent']]
        fork_started = time.perf_counter()
        child = Branch(execution, parent.environment.fork(), parent.declaration)
        fork_seconds = time.perf_counter() - fork_started
        call = {**proposal['action'], 'id': proposal['id']}
        child.execution.execute_actions([call], lambda calls: {
            action['id']: child.environment.observe(action['tool'], action['arguments'])
            for action in calls})
        child.execution.trace[-1]['fork_seconds'] = fork_seconds
        child.execution.trace[-1]['tool_timings'] = child.environment.tool_timings[
            len(parent.environment.tool_timings):]
        return proposal['id'], child

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(execute, unique.values()))


def required_fields(execution, action, fields):
    pending = template_references(action, execution.settings)
    visited, required = set(), set()
    while pending:
        identity = pending.pop()
        visited.add(identity)
        if identity in execution.bound_fields:
            value = execution.bound_fields[identity]
        else:
            required.add(identity)
            value = fields[identity]['values']
        pending.update(template_references(value, execution.settings) - visited)
    return required


def merge_extensions(beam, active, selected, choices, extension_scores, prefix_scores, width):
    candidates = [(active[index], choice) for index, choice in zip(selected, choices, strict=True)]
    inactive = [index for index in range(len(beam)) if index not in active]
    if not inactive:
        return candidates, extension_scores
    carried = torch.tensor(inactive, device=prefix_scores.device)
    combined = torch.cat((extension_scores, prefix_scores.index_select(0, carried)))
    candidates.extend((index, None) for index in inactive)
    order = sorted(range(len(candidates)), key=lambda index: (candidates[index][0],
                   -1 if candidates[index][1] is None else candidates[index][1]))
    ordered = torch.tensor(order, device=combined.device)
    ranking = combined.index_select(0, ordered).argsort(descending=True, stable=True)[:width]
    retained = ordered.index_select(0, ranking)
    return [candidates[index] for index in retained.tolist()], combined.index_select(0, retained)


def spawn_scored(branches, branch_width, workers, identities):
    """Complete fields conditionally, batch every beam, then execute complete calls."""
    score_started = time.perf_counter()
    fields = {owner: parent.declaration['fields'] for owner, parent in branches.items()}
    field_by_id = {owner: {field['id']: field for field in declared} for owner, declared in fields.items()}
    beams, scores = {}, {}
    for identity, parent in branches.items():
        parent.execution.values.clear()
        parent.execution.bound_fields.clear()
        parent.execution.field_evidence.clear()
        parent.execution.action_template = parent.declaration['action']
        parent.execution.fields.update(field_by_id[identity])
        beams[identity] = [(identity, parent.execution.fork())]
        scores[identity] = parent.execution.service.initial_scores()
    for position in range(max(len(declared) for declared in fields.values())):
        current = {owner: declared[position] for owner, declared in fields.items() if position < len(declared)}
        partials = {identity: execution for owner, field in current.items()
                    for identity, execution in beams[owner]
                    if field['id'] in required_fields(execution, branches[owner].declaration['action'], field_by_id[owner])}
        requested = {identity: [field] for owner, field in current.items() if len(field['values']) > 1
                     for identity, _ in beams[owner] if identity in partials}
        if requested:
            QueryExecution.evaluate_branches({identity: partials[identity] for identity in requested}, requested)
        next_beams = dict(beams)
        for owner, field in current.items():
            beam = beams[owner]
            parent = branches[owner]
            active = [index for index, (identity, _) in enumerate(beam) if identity in partials]
            if not active:
                next_beams[owner] = beam
                continue
            if len(field['values']) == 1:
                candidates = [(index, 0 if index in active else None) for index in range(len(beam))]
            else:
                parent.execution.trace.extend(beam[index][1].trace[-1] for index in active)
                requests = [json.dumps([beam[index][0], field['id']]) for index in active]
                active_scores = scores[owner] if len(active) == len(beam) else scores[owner].index_select(
                    0, torch.tensor(active, device=scores[owner].device))
                selected, choices, extension_scores = parent.execution.service.extend(
                    requests, active_scores, task_id=parent.execution.task_id, width=branch_width)
                candidates, scores[owner] = merge_extensions(beam, active, selected, choices,
                    extension_scores, scores[owner], branch_width)
            next_beams[owner] = []
            for index, choice in candidates:
                identity, execution = beam[index]
                if choice is None:
                    next_beams[owner].append((identity, execution))
                    continue
                child = execution.fork()
                child.values[field['id']] = field['values'][choice]
                child.bound_fields[field['id']] = field['values'][choice]
                key = json.dumps([identity, field['id'], choice])
                next_beams[owner].append((key, child))
        beams = next_beams
    prepared = []
    for owner, beam in beams.items():
        for (_, execution), log_probability in zip(beam, scores[owner].tolist(), strict=True):
            action = execution.materialize(branches[owner].declaration['action'])
            signature = json.dumps([owner, action['tool'], action['arguments']],
                                   **execution.settings['action_identity_serialization'])
            prepared.append((signature, owner, execution, log_probability))
    # Probability selects the beam; action identity determines its persistent node labels.
    proposals, selections = [], {}
    for _, owner, execution, log_probability in sorted(prepared, key=lambda item: item[0]):
        child_id = next(identities)
        values = dict(execution.bound_fields)
        selections[child_id] = (values, log_probability)
        proposals.append({'id': child_id, 'parent': owner, 'values': values,
                          'action': branches[owner].declaration['action']})
    score_seconds = time.perf_counter() - score_started
    dispatch_started = time.perf_counter()
    children = spawn(branches, proposals, workers)
    dispatch_seconds = time.perf_counter() - dispatch_started
    for identity, child in children.items():
        selected, log_probability = selections[identity]
        child.execution.trace.append({'selected_values': selected, 'conditional_log_probability': log_probability,
            'shared_batch_timing': {'score_seconds': score_seconds,
                                   'dispatch_seconds': dispatch_seconds}})
    return children
