"""Invoke the original ReAct notebook function with task and model adapters."""

import ast
import json
from pathlib import Path
import re

from baselines.tool_agents.tools import calculate


class MathEnvironment:
    def __init__(self, task, prompts, limits):
        self.task = task
        self.prompts = prompts
        self.limits = limits
        self.actions = []

    def reset(self, idx=None):
        field = self.task['fields']['q0']
        options = '\n'.join(option['id'] + '. ' + option['description'] for option in field['options'])
        return self.prompts['question'].format(state=self.task['state'],
                                               question=field['question'], options=options)

    def step(self, action):
        self.actions.append(action)
        matched = re.fullmatch(r'([A-Za-z]+)\[(.*)\]', action.strip(), flags=re.S)
        if matched is None:
            return self.prompts['invalid_action'], 0, False, {}
        name, value = matched.groups()
        if name == 'finish':
            return self.prompts['finished'], 0, True, {'answer': value.strip()}
        if name == 'calculate':
            try:
                observation = str(calculate(value, self.limits)['value'])
            except (ValueError, SyntaxError, ArithmeticError) as error:
                observation = str(error)
            return observation, 0, False, {}
        return self.prompts['invalid_action'], 0, False, {}


def original_function(notebook, namespace):
    cells = json.loads(Path(notebook).read_text())['cells']
    source = next(''.join(cell['source']) for cell in cells
                  if cell['cell_type'] == 'code' and '\ndef webthink(' in ''.join(cell['source']))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'webthink')
    for node in ast.walk(function):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'action' and isinstance(node.slice, ast.Constant)
                and node.slice.value == 0):
            node.slice = ast.Slice(upper=ast.Constant(value=1))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(notebook), 'exec'), namespace)
    return namespace['webthink']


def solve(task, complete, config, prompts):
    environment = MathEnvironment(task, prompts, config['tool_limits'])
    calls = []

    def llm(prompt, stop):
        if len(calls) >= config['max_model_calls']:
            raise RuntimeError('Model call budget exhausted.')
        messages = [{'role': 'user', 'content': prompt}]
        output = complete(messages, config['max_new_tokens'], config['temperature'], n=1, stop=stop)[0]
        calls.append({'stop': stop, 'output': output})
        return output

    namespace = {'env': environment, 'llm': llm,
                 'step': lambda env, action: env.step(action),
                 'webthink_prompt': prompts['instruction']}
    function = original_function(config['notebook'], namespace)
    reward, result = function(idx=task['task_id'], prompt=prompts['instruction'], to_print=False)
    return {'task_id': task['task_id'], 'answer': result['answer'],
            'actions': environment.actions, 'calls': calls, 'trajectory': result['traj'],
            'core_calls': result['n_calls'], 'core_format_retries': result['n_badcalls']}
