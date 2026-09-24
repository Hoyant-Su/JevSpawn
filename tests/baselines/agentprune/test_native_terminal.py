import json
from pathlib import Path

import pytest

from baselines.agentprune.common_core import task_graph
from baselines.common.agentprune import solve
from baselines.common.config import SharedConfig
from baselines.common.transition_budget import TransitionBudget
from environments.plancraft import PlancraftEnvironment
from environments.ppnl import PPNLEnvironment
from jev_spawn.infra.prompts import load_prompt


@pytest.fixture(scope='module')
def configuration():
    method = json.loads(Path('configs/baselines/common/methods/agentprune.json').read_text())
    shared = SharedConfig.load('configs/shared_config_tp4_v2.yaml')
    return shared.method_settings(method['settings']), load_prompt(method['prompts'])


def episode(dataset, index, factory, configuration, directory):
    task = json.loads(Path(f'data/qualification/native_screen/{dataset}/tasks.jsonl').read_text().splitlines()[index])
    native = factory(task, {}, directory, deadline=lambda: 300, configuration=configuration)
    return task, TransitionBudget(native, 36, 'finish', 0, [])


def responses(values):
    pending = iter(values)
    calls = []

    def complete(messages, max_tokens, temperature):
        calls.append(messages)
        return [next(pending)]

    return complete, calls


def test_real_graph_stops_on_native_goal_without_fabricated_node_outputs(configuration, tmp_path):
    settings, prompts = configuration
    task, environment = episode('ppnl', 0, PPNLEnvironment,
        'configs/environments/ppnl_runtime.json', tmp_path)
    official = json.loads(Path('data/context_datasets/ppnl/ICL_test_set.json').read_text())[0]
    action = {'tool': 'execute', 'arguments': {'actions': official['agent_as_a_point']}}
    complete, generated = responses(['Actual first-node analysis.', prompts['tool_prefix'] + json.dumps(action)])
    result = solve(task, environment, complete, settings, prompts)
    assert len(generated) == len(result['calls']) == 2
    assert len(result['executed_node_ids']) == len(result['nodes']) == 2
    outputs = {node['id']: node['outputs'] for node in result['nodes']}
    assert outputs[result['executed_node_ids'][0]] == ['Actual first-node analysis.']
    assert outputs[result['executed_node_ids'][1]] == []
    assert result['calls'][-1]['tool_call'] == action
    assert result['calls'][-1]['observation'] == environment.observation
    assert result['calls'][-1]['done'] is True
    assert environment.done and environment.evaluate(result['answer']) is True
    assert len(environment.actions) == len(environment.transitions) == 1
    assert result['termination'] == 'native_environment_terminal'
    assert result['raw_answers'] == result['decision_node_outputs'] == []
    assert result['topology_log_probability'] is None
    assert result['topology_log_probability_status'] == 'not_returned'
    assert result['topology_seed'] == settings['seed']
    graph, _ = task_graph(settings)
    assert graph.arun.__code__.co_filename.endswith('external/AgentPrune/AgentPrune/graph/graph.py')
    assert 'await asyncio.wait_for' in result['provenance']['executed_graph_methods']


def test_native_terminal_failure_is_not_inferred_as_success(configuration, tmp_path):
    settings, prompts = configuration
    task, environment = episode('plancraft', 1, PlancraftEnvironment,
        'configs/environments/plancraft_runtime.json', tmp_path)
    assert task['source']['example']['impossible'] is False
    action = {'tool': 'execute', 'arguments': {'action': 'impossible: test terminal boundary'}}
    complete, generated = responses([prompts['tool_prefix'] + json.dumps(action)])
    result = solve(task, environment, complete, settings, prompts)
    assert len(generated) == len(result['executed_node_ids']) == 1
    assert environment.done and result['answer'] == environment.answer
    assert environment.evaluate(result['answer']) is False
    assert result['termination'] == 'native_environment_terminal'
    assert result['nodes'][0]['outputs'] == []
    assert result['decision_node_outputs'] == []
    assert len(environment.actions) == len(environment.transitions) == 1


def test_nonterminal_graph_keeps_all_original_nodes_and_finalrefer(configuration, tmp_path):
    settings, prompts = configuration
    task, environment = episode('ppnl', 0, PPNLEnvironment,
        'configs/environments/ppnl_runtime.json', tmp_path)
    graph, _ = task_graph(settings)
    official = json.loads(Path('data/context_datasets/ppnl/ICL_test_set.json').read_text())[0]
    answer = {'actions': official['agent_as_a_point']}
    analyses = [f'Actual node analysis {index}.' for index in range(len(graph.nodes))]
    complete, generated = responses([*analyses, json.dumps(answer)])
    result = solve(task, environment, complete, settings, prompts)
    assert len(generated) == len(graph.nodes) + 1
    assert len(result['nodes']) == len(graph.nodes)
    assert len(result['executed_node_ids']) == len(graph.nodes) + 1
    assert result['raw_answers'] == result['decision_node_outputs'] == [json.dumps(answer)]
    assert result['termination'] == 'graph_completed'
    assert result['topology_log_probability_status'] == 'returned'
    assert isinstance(result['topology_log_probability'], float)
    assert environment.evaluate(result['answer']) is True
