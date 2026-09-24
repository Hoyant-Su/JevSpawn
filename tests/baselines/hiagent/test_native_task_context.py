from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from baselines.common.resources import ADAPTER_SETTINGS
from baselines.hiagent.adapter import original_agent
from environments.ppnl import PPNLEnvironment
from jev_spawn.infra.prompts import load_prompt


@pytest.fixture(scope='module')
def setup():
    shared = SharedConfig.load('configs/shared_config_tp4_v2.yaml')
    method = json.loads(Path('configs/baselines/common/methods/hiagent.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True)
    model = SimpleNamespace(engine=shared.model.path,
        context_length=shared.model.max_input_tokens + shared.generation.max_new_tokens,
        max_tokens=shared.generation.max_new_tokens,
        num_tokens_from_messages=lambda messages: len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)))
    tasks = [json.loads(line) for line in Path('data/qualification/native_screen/ppnl/tasks.jsonl').read_text().splitlines()]
    return method, load_prompt(method['prompts']), model, tasks


def render(core, context, setup):
    method, prompts, model, _ = setup
    agent = core(model, memory_size=method['settings']['memory_size'],
        instruction=prompts['instruction'], examples=ADAPTER_SETTINGS['hiagent']['examples'],
        system_message=prompts['system'], need_goal=ADAPTER_SETTINGS['hiagent']['need_goal'],
        check_actions=prompts['check_actions'], use_parser=ADAPTER_SETTINGS['hiagent']['use_parser'])
    agent.reset(context, prompts['initial_observation'])
    prompt = agent.make_prompt(need_goal=agent.need_goal, check_actions=agent.check_actions,
        check_inventory=agent.check_inventory, system_message=prompts['system'])
    return agent.task, prompt


def test_real_native_context_is_isolated_between_concurrent_agents(setup, tmp_path, monkeypatch):
    method, _, _, tasks = setup
    monkeypatch.delenv('EVALTASK', raising=False)
    contexts = [PPNLEnvironment(task, {}, tmp_path / str(index), deadline=lambda: 300,
        configuration='configs/environments/ppnl_runtime.json').reset() for index, task in enumerate(tasks[:2])]

    def invoke(context):
        return render(original_agent(method['settings']['source_directory'],
            environment={'EVALTASK': context}), context, setup)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, contexts))
    assert contexts[0] != contexts[1]
    for context, (task, prompt) in zip(contexts, results, strict=True):
        assert task == context
        assert context in prompt
    assert 'EVALTASK' not in os.environ


def test_private_namespace_preserves_official_environment_semantics(setup, tmp_path, monkeypatch):
    method, _, _, tasks = setup
    context = PPNLEnvironment(tasks[0], {}, tmp_path, deadline=lambda: 300,
        configuration='configs/environments/ppnl_runtime.json').reset()
    monkeypatch.setenv('EVALTASK', context)
    expected = render(original_agent(method['settings']['source_directory']), context, setup)
    actual = render(original_agent(method['settings']['source_directory'],
        environment={'EVALTASK': context}), context, setup)
    assert actual == expected
    assert os.environ['EVALTASK'] == context
