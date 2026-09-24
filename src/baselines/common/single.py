import json

import jsonschema

from baselines.common.errors import InvalidOutputError, TaskLimitError
from baselines.common.resources import TEMPLATES


def solve(task, environment, complete, settings, prompts):
    messages = [{'role': 'system', 'content': prompts['system']},
                {'role': 'user', 'content': environment.reset()}]
    for _ in range(settings['max_turns']):
        text = complete(messages, settings['max_new_tokens'], settings['temperature'])[0]
        messages.append({'role': 'assistant', 'content': text})
        try:
            action = json.loads(text)
            jsonschema.validate(action, prompts['action_schema'])
        except (ValueError, jsonschema.ValidationError) as error:
            raise InvalidOutputError(str(error)) from error
        try:
            observation, done = environment.execute(action['tool'], action['arguments'])
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
            observation, done = TEMPLATES['tool_input_error'].format(error=error), False
        if done:
            return {'task_id': task['task_id'], 'answer': environment.answer,
                    'actions': environment.actions, 'messages': messages}
        messages.append({'role': 'user', 'content': observation})
    raise TaskLimitError('Declared action budget exhausted.')
