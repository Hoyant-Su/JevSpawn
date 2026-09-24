import json
import re

from baselines.official_react.configured_adapter import original_function
from baselines.common.errors import TaskLimitError
from baselines.common.resources import TEMPLATES
from baselines.tool_agents.tools import ActionError


class EnvironmentAdapter:
    def __init__(self, environment, prompts):
        self.environment, self.prompts = environment, prompts

    def reset(self, idx=None):
        return self.environment.reset()

    def step(self, action):
        match = re.fullmatch(TEMPLATES['react_action_pattern'], action.strip())
        if match is None:
            feedback = self.environment.reject(None, action, ActionError(self.prompts['invalid_action']))
            return json.dumps(feedback), 0, False, {}
        name, raw = match.groups()
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as error:
            feedback = self.environment.reject(name.lower(), raw, error)
            return json.dumps(feedback), 0, False, {}
        observation, done = self.environment.execute(name.lower(), arguments)
        return observation, 0, done, {'answer': self.environment.answer}


def solve(task, environment, complete, settings, prompts):
    adapter = EnvironmentAdapter(environment, prompts)
    calls = []

    def llm(prompt, stop):
        messages = [{'role': 'system', 'content': prompts['instruction']},
                    {'role': 'user', 'content': prompt.removeprefix(prompts['instruction'])}]
        output = complete(messages, settings['max_new_tokens'],
                          settings['temperature'], stop=stop)[0]
        calls.append({'prompt': prompt, 'messages': messages, 'output': output, 'stop': stop})
        return output

    namespace = {'env': adapter, 'llm': llm, 'step': lambda env, action: env.step(action),
                 'webthink_prompt': prompts['instruction']}
    core = original_function(settings['notebook'], namespace,
                             episode_action_limit=settings['max_turns'],
                             original_episode_range=settings['original_episode_range'])
    _, result = core(idx=task['task_id'], prompt=prompts['instruction'], to_print=False)
    if environment.answer is None:
        environment.deadline()
        raise TaskLimitError('ReAct terminated within its declared limits without submitting an answer.')
    return {'task_id': task['task_id'], 'answer': environment.answer,
            'trajectory': result['traj'], 'calls': calls, 'actions': environment.actions,
            'core_calls': result['n_calls'], 'core_format_retries': result['n_badcalls']}
