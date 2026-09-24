import ast
from functools import partial
import json
import logging
from pathlib import Path
import random
import re
from types import SimpleNamespace

import numpy as np

from baselines.common.errors import TaskLimitError
from baselines.tool_agents.tools import ActionError
from baselines.common.resources import ADAPTER_SETTINGS, TEMPLATES


class EvaluatorPrompt(str):
    def __new__(cls, instruction, context, prompts):
        prompt = super().__new__(cls, instruction + context)
        prompt.messages = [{'role': 'system', 'content': prompts['value_role_system'].format(instruction=instruction)},
                           {'role': 'user', 'content': prompts['value_role_context'].format(context=context)}]
        return prompt


class ActionTransport(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id in {'action_type', 'action_param'}
               for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_JoinedStr(self, node):
        names = {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
        if names == {'action_type', 'action_param'}:
            return ast.copy_location(ast.Name(id='action_line', ctx=ast.Load()), node)
        return self.generic_visit(node)


class SearchInterface(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'lats_search':
            for child in ast.walk(node):
                if isinstance(child, ast.While) and ast.unparse(child.test).startswith('node is None or'):
                    child.body.insert(0, ast.parse('if node is None: break').body[0])
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'rollout':
                    for keyword in child.keywords:
                        if keyword.arg == 'max_depth':
                            keyword.value = ast.parse('args.rollout_depth', mode='eval').body
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                    call = child.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == 'basicConfig':
                        child.value = ast.Constant(None)
            loop = next(child for child in node.body if isinstance(child, ast.For))
            expansion = next(index for index, child in enumerate(loop.body)
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name) and child.value.func.id == 'expand_node')
            loop.body[expansion + 1:expansion + 1] = ast.parse('''
terminal_nodes.extend(child for child in node.children if child.is_terminal)
success = next((child for child in node.children if child.is_terminal and child.reward == 1), None)
if success is not None:
    return success.state, success.value, [(item, item.value) for item in collect_all_nodes(root)], success.reward, success.em
if node.children and all(child.is_terminal for child in node.children):
    for child in node.children:
        backpropagate(child, child.reward)
    continue
''').body
            for index, child in enumerate(loop.body):
                if isinstance(child, ast.While) and ast.unparse(child.test) == 'node.is_terminal or not node.children':
                    child.body.insert(2, ast.parse('if node is None: break').body[0])
                    loop.body.insert(index + 1, ast.parse('if node is None: break').body[0])
                    break
        if node.name == 'select_node':
            loop = node.body[0]
            loop.body = loop.body[:3] + loop.body[4:6] + loop.body[3:4] + loop.body[6:]
        if node.name == 'expand_node':
            for child in ast.walk(node):
                if isinstance(child, ast.Compare) and ast.unparse(child.left) == 'node.depth':
                    child.comparators = [ast.parse('args.max_depth', mode='eval').body]
        if node.name == 'rollout':
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'n' for t in child.targets):
                    child.value = ast.parse('args.rollout_samples', mode='eval').body
        if node.name == 'generate_new_states':
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == 'split' and len(child.args) == 1 and isinstance(child.args[0], ast.Constant) and child.args[0].value == ':':
                    child.args.append(ast.Constant(1))
                if isinstance(child, ast.Compare) and ast.unparse(child.left) == 'r' and len(child.comparators) == 1 and isinstance(child.comparators[0], ast.Constant) and child.comparators[0].value == 0:
                    child.ops = [ast.Lt()]
                    child.comparators = [ast.Constant(1)]
            node = ActionTransport().visit(node)
        return self.generic_visit(node)


def load_core(source, gpt, environment):
    path = Path(source) / 'hotpot/lats.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    tree = ast.fix_missing_locations(SearchInterface().visit(tree))
    namespace = {'np': np, 'partial': partial, 'logging': logging, 'gpt': gpt, 'env': environment,
                 'reflection_map': [], 'failed_trajectories': []}
    exec(compile(tree, str(path), 'exec'), namespace)
    task_path = Path(source) / 'hotpot/hotpotqa.py'
    task_tree = ast.parse(task_path.read_text())
    task_tree.body = [node for node in task_tree.body if isinstance(node, ast.ClassDef)]
    task_namespace = {'Task': object, 'gpt': gpt, 'random': random, 're': re, 'logging': logging}
    exec(compile(task_tree, str(task_path), 'exec'), task_namespace)
    return namespace, task_namespace['HotPotQATask']


def solve(task, environment, complete, settings, prompts):
    return solve_with_core(task, environment, complete, settings, prompts, load_core)


def solve_with_core(task, environment, complete, settings, prompts, core_loader):
    if settings['terminal_reward'] != 'original_lats_lm_value':
        raise ValueError('Unsupported LATS terminal reward mode.')
    calls, terminal_estimates, value_estimates = [], [], []

    def gpt(prompt, n=ADAPTER_SETTINGS['lats']['completion_samples'], stop=None, model=None, temperature=None):
        environment.deadline()
        messages = prompt.messages if isinstance(prompt, EvaluatorPrompt) else [{'role': 'user', 'content': prompt}]
        outputs = complete(messages, settings['max_new_tokens'],
                           settings['temperature'], n=n, stop=stop)
        calls.extend({'prompt': prompt, 'messages': messages, 'output': output, 'stop': stop,
                      'temperature': settings['temperature']} for output in outputs)
        return outputs

    core, original_task = core_loader(settings['source_directory'], gpt, environment)

    class Task(original_task):
        def cot_prompt_wrap(self, x, y='', reflection_mapping_list=None):
            reflections = [] if reflection_mapping_list is None else reflection_mapping_list
            return prompts['policy'].format(trajectory=x, reflections=json.dumps(reflections)) + y

        def value_prompt_wrap(self, x, y, z=None, reflections=None):
            context = prompts['value_context'].format(trajectory=y, failures=json.dumps(z),
                                                       reflections=json.dumps(reflections))
            return EvaluatorPrompt(prompts['value_instruction'], context, prompts)

        def value_outputs_unwrap(self, outputs):
            value = original_task.value_outputs_unwrap(outputs)
            value_estimates.append({'outputs': outputs, 'value': value})
            return value

        def generate_self_reflection(self, trajectories, question):
            return [{'question': question, 'trajectory': trajectory,
                     'reflection': gpt(prompts['reflection'].format(trajectory=trajectory, question=question))[0]}
                    for trajectory in trajectories]

    search_task = Task()
    args = SimpleNamespace(backend='shared', temperature=settings['temperature'], prompt_sample=ADAPTER_SETTINGS['lats']['prompt_sample'],
        n_generate_sample=settings['expansion_samples'], n_evaluate_sample=settings['evaluation_samples'],
        rollout_depth=settings['rollout_depth'], max_depth=settings['max_turns'],
        rollout_samples=settings['rollout_samples'])
    current_node = [None]
    current_environment = [None]
    state_environments = {}
    root_environment = environment.fork()
    original_node = core['Node']

    class Node(original_node):
        def __init__(self, state, question, parent=None):
            super().__init__(state, question, parent)
            self.environment = root_environment if parent is None else current_environment[0]
            state_environments[id(self.state)] = self.environment

    core['Node'] = Node
    original_generate = core['generate_new_states']
    original_prompt = core['generate_prompt']

    def generate_prompt(node):
        trajectory = original_prompt(node)
        return environment.reset() + trajectory[len(node.question):]

    def generate_new_states(node, args, task, n):
        current_node[0] = node
        return original_generate(node, args, task, n)

    def step(unused, action):
        branch = current_node[0].environment.fork()
        current_environment[0] = branch

        def rejected(message):
            feedback = branch.reject(None, action, ActionError(message))
            environment.actions.append(branch.actions[-1])
            return json.dumps(feedback), 0, False, {}

        match = re.fullmatch(TEMPLATES['lats']['action_pattern'], action.strip())
        if match is None:
            return rejected(TEMPLATES['lats']['invalid_action'])
        name, raw = match.groups()
        name = name.lower()
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as error:
            return rejected(TEMPLATES['lats']['invalid_json'].format(error=error))
        action_count, timing_count = len(branch.actions), len(branch.tool_timings)
        observation, done = branch.execute(name, arguments)
        environment.actions.extend(branch.actions[action_count:])
        environment.tool_timings.extend(branch.tool_timings[timing_count:])
        reward = 0
        if done:
            trajectory = TEMPLATES['lats']['trajectory_action'].format(
                trajectory=core['generate_prompt'](current_node[0]), action=action)
            reward = core['get_value'](search_task, environment.reset(), trajectory, settings['evaluation_samples'])
            terminal_estimates.append({'action': action, 'value': reward,
                                       'source': 'original_lats_lm_value_estimator'})
            observation = json.dumps({'submitted': True, 'lm_value_estimate': reward,
                                      'correctness_observed': False})
        return observation, reward, done, {}

    def evaluate_node(node, args, task):
        children = [child for child in node.children if not child.is_terminal]
        values = core['get_values'](task, node.question, [core['generate_prompt'](child) for child in children],
                                  args.n_evaluate_sample)
        for child, value in zip(children, values, strict=True):
            child.value = value
        for child in node.children:
            if child.is_terminal:
                child.value = child.reward
        return sum(child.value for child in node.children) / len(node.children)

    original_select = core['select_node']

    def select_node(node):
        environment.deadline()
        return original_select(node)

    class Environment:
        def reset(self, idx):
            return environment.reset()

    core.update(env=Environment(), step=step, generate_new_states=generate_new_states, generate_prompt=generate_prompt,
                evaluate_node=evaluate_node, select_node=select_node)
    state, value, nodes, reward, _ = core['lats_search'](args, search_task, task['task_id'],
                                                       iterations=settings['search_iterations'], to_print=False)
    selected_environment = state_environments[id(state)]
    if selected_environment.answer is None:
        raise TaskLimitError('LATS search ended without selecting a completed candidate.')
    environment.answer, environment.done = selected_environment.answer, selected_environment.done
    return {'task_id': task['task_id'], 'answer': environment.answer, 'actions': environment.actions,
            'calls': calls, 'selected_state': state, 'selected_value': value,
            'terminal_value': reward, 'terminal_estimates': terminal_estimates,
            'value_estimates': value_estimates,
            'nodes': [{'state': node.state, 'value': node.value, 'visits': node.visits,
                       'depth': node.depth, 'reward': node.reward} for node, _ in nodes],
            'reflections': core['reflection_map'], 'upstream_commit': settings['upstream_commit'],
            'adaptations': ['terminal rewards use original LATS LM value estimation, not oracle correctness',
                            'shared tools and JSON arguments; full structured answer retained',
                            'shared generation policy and explicit search budgets',
                            'raw evaluator outputs use the original LATS value parser; exhausted selection loop terminates',
                            'child values preserve child identities when terminal and nonterminal children mix']}
