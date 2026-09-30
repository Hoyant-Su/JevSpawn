from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time


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
