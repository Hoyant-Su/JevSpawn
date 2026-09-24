import argparse
import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.errors import InputLimitError
from baselines.common.resources import ADAPTER_SETTINGS
from baselines.common.tasks import read, rows
from baselines.hiagent.adapter import original_agent
from jev_spawn.infra.prompts import load_prompt


def upstream(directory):
    namespace = {'json': json, 'os': os, 're': re, 'print': lambda *args, **kwargs: None}
    for filename in ['base_agent.py', 'summarize.py', 'cme_final.py']:
        path = Path(directory) / filename
        definitions = [node for node in ast.parse(path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        for node in definitions:
            node.decorator_list = []
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ContextEfficientAgentV2']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    config = read(parser.parse_args().config)
    shared = SharedConfig.load(config['shared_config'])
    method = read(config['method'])
    prompts = load_prompt(method['prompts'])
    tokenizer = AutoTokenizer.from_pretrained(shared.backend()['model_path'])
    observed = []

    def count(messages):
        tokens = len(tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False))
        observed.append(tokens)
        return tokens

    model = SimpleNamespace(engine=shared.backend()['model_path'],
        context_length=shared.model.max_input_tokens + shared.generation.max_new_tokens,
        max_tokens=shared.generation.max_new_tokens, num_tokens_from_messages=count)

    def build(core, task):
        os.environ['EVALTASK'] = task['dataset']
        environment = TaskEnvironment(task, read(config['environment']),
            Path(config['work_dir']) / task['dataset'], deadline=lambda: shared.runtime.sample_timeout_seconds)
        agent = core(model, memory_size=method['settings']['memory_size'],
            instruction=prompts['instruction'], examples=ADAPTER_SETTINGS['hiagent']['examples'],
            system_message=prompts['system'], need_goal=ADAPTER_SETTINGS['hiagent']['need_goal'],
            check_actions=prompts['check_actions'], use_parser=ADAPTER_SETTINGS['hiagent']['use_parser'])
        agent.reset(environment.reset(), prompts['initial_observation'])
        return agent

    def render(agent):
        return agent.make_prompt(need_goal=agent.need_goal, check_actions=agent.check_actions,
            check_inventory=agent.check_inventory, system_message=prompts['system'])

    fixed = original_agent(method['settings']['source_directory'])
    valid = rows(config['valid_tasks'])[config['task_index']]
    reference = render(build(upstream(method['settings']['source_directory']), valid))
    actual = render(build(fixed, valid))
    assert actual == reference
    history = read(config['valid_history'])
    assert history['task_id'] == valid['task_id']
    reference_agent = build(upstream(method['settings']['source_directory']), valid)
    actual_agent = build(fixed, valid)
    reference_agent.memory = history['memory'][:config['valid_history_entries']]
    actual_agent.memory = history['memory'][:config['valid_history_entries']]
    assert render(actual_agent) == render(reference_agent)
    oversized = rows(config['oversized_tasks'])[config['task_index']]
    observed.clear()
    try:
        render(build(fixed, oversized))
    except InputLimitError as error:
        failure = str(error)
    else:
        raise AssertionError('Real oversized fixed goal did not produce the required input-limit outcome.')
    destination = Path(config['output'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'valid_task_id': valid['task_id'], 'valid_prompt_exact': True,
        'recorded_history_prompt_exact': True, 'recorded_history_entries': config['valid_history_entries'],
        'oversized_task_id': oversized['task_id'], 'token_counts': observed,
        'error': failure, 'model_generation_invoked': False,
        'scope': 'Only termination of the upstream overlength loop after removable history is empty.'}, indent=2) + '\n')
    print(destination.read_text())


if __name__ == '__main__':
    main()
