from jev_spawn.infra.prompts import load_prompt
from jev_spawn.rollout import branching
from jev_spawn.runtime import query_execution, state


def compact(payload, settings):
    result = dict(payload)
    result['declaration_feedback_records'] = [
        {key: value for key, value in record.items() if key not in settings['compiler_payload_fields']}
        for record in result['declaration_feedback_records']
    ]
    result['format'] = load_prompt(settings['prompt'])
    return result


def install(settings):
    def history_state(value):
        return compact(state.history_state(value), settings)

    def history_frontier(value):
        payload, branches = state.history_frontier(value)
        return compact(payload, settings), branches

    branching.history_state = query_execution.history_state = history_state
    branching.history_frontier = history_frontier
