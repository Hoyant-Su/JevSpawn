from jev_spawn.service.errors import FlowInputError, InputLimitError, InvalidOutputError, TaskLimitError
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.rollout.branching import solve as solve_branching


def solve(task, environment, complete, settings, prompts, on_turn=None):
    rollout = {**settings['rollout'], 'terminal_answer': settings['terminal_answer']}
    trace = {'query': task['context']}
    try:
        result = solve_branching(trace['query'], task_id=task['task_id'],
            service=complete.func.__self__, complete=complete, settings=rollout,
            prompts=load_prompt(rollout['prompts']),
            budget={name: settings[name] for name in ('max_turns', 'max_new_tokens', 'temperature')},
            trace=trace, environment=environment, on_turn=on_turn)
    except (TimeoutError, InvalidOutputError, FlowInputError, InputLimitError, TaskLimitError) as error:
        error.trace = trace
        raise
    return {'task_id': task['task_id'], **result, 'trace': trace}
