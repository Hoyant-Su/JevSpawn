import asyncio
import copy
import json

import jsonschema
import torch

from baselines.agentprune.common_core import RNG, task_graph
from baselines.common.errors import InvalidOutputError, TaskLimitError
from baselines.common.resources import ADAPTER_SETTINGS


class NativeTerminal(Exception):
    """The actual environment ended during a graph node's tool dialogue."""


def task_context(task, environment):
    return environment.context(False, {'ensure_ascii': False})


class DecisionPrompts:
    def __init__(self, original, constraint):
        self.original, self.constraint = original, constraint

    def get_decision_role(self):
        return self.original.get_decision_role()

    def get_decision_constraint(self):
        return self.constraint

    def get_decision_few_shot(self):
        return self.original.get_decision_few_shot()


class Provider:
    def __init__(self, node, decision, environment, complete, settings, prompts, calls):
        self.node, self.decision = node, decision
        self.environment, self.complete = environment, complete
        self.settings, self.prompts, self.calls = settings, prompts, calls

    async def agen(self, messages):
        messages = copy.deepcopy(messages)
        protocol = self.prompts['decision_protocol' if self.decision else 'analysis_protocol']
        messages[0]['content'] += '\n\n' + protocol
        for _ in range(self.settings['node_iterations']):
            self.environment.deadline()
            record = {'node_id': self.node.id, 'role': self.node.role, 'decision': self.decision,
                      'messages': copy.deepcopy(messages)}
            self.calls.append(record)
            output = (await asyncio.to_thread(self.complete, messages, self.settings['max_new_tokens'],
                                             self.settings['temperature']))[0]
            record['output'] = output
            messages.append({'role': 'assistant', 'content': output})
            control = output.strip()

            def rejected(error):
                observation = json.dumps(self.environment.reject(None, output, error))
                record.update(observation=observation, done=False)
                messages.append({'role': 'user', 'content': observation})

            if not control:
                rejected(InvalidOutputError('AgentPrune node produced an empty response.'))
                continue
            if not control.startswith(self.prompts['tool_prefix']):
                if not self.decision:
                    return output
                try:
                    answer = json.loads(output)
                except json.JSONDecodeError as error:
                    rejected(error)
                    continue
                observation, done = self.environment.execute('finish', answer)
                record.update(tool_call={'tool': 'finish', 'arguments': answer},
                              observation=observation, done=done)
                if done:
                    return output
                messages.append({'role': 'user', 'content': observation})
                continue
            try:
                action = json.loads(control[len(self.prompts['tool_prefix']):])
                jsonschema.validate(action, self.prompts['tool_schema'])
            except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                rejected(error)
                continue
            if action['tool'] == 'finish':
                try:
                    jsonschema.validate(action['arguments'], self.environment.display_answer_schema())
                except jsonschema.ValidationError as error:
                    rejected(error)
                    continue
                record['tool_call'] = action
                if not self.decision:
                    record.update(node_completion='proposal', proposal=action['arguments'])
                    return json.dumps(action['arguments'], ensure_ascii=False)
                observation, done = self.environment.execute('finish', action['arguments'])
                record.update(observation=observation, done=done)
                if done:
                    return json.dumps(action['arguments'], ensure_ascii=False)
                messages.append({'role': 'user', 'content': observation})
                continue
            observation, done = self.environment.execute(action['tool'], action['arguments'])
            record.update(tool_call=action, observation=observation, done=done)
            if done:
                assert self.environment.done and self.environment.answer is not None
                raise NativeTerminal()
            messages.append({'role': 'user', 'content': observation})
        raise TaskLimitError('Declared AgentPrune node action budget exhausted.')


def solve(task, environment, complete, settings, prompts):
    graph, provenance = task_graph(settings)
    assert settings['num_rounds'] == provenance['training_config']['training']['num_rounds']
    calls = []
    for node in graph.nodes.values():
        node.llm = Provider(node, False, environment, complete, settings, prompts, calls)
    graph.decision_node.llm = Provider(graph.decision_node, True, environment, complete, settings, prompts, calls)
    graph.decision_node.prompt_set = DecisionPrompts(graph.decision_node.prompt_set, prompts['decision_constraint'])
    seed = settings['seed']
    token = RNG.set(torch.Generator(device='cpu').manual_seed(seed))
    try:
        answers, log_probability = asyncio.run(graph.arun(
            {'task': task_context(task, environment)}, num_rounds=settings['num_rounds'], max_tries=ADAPTER_SETTINGS['agentprune']['max_tries'],
            max_time=environment.deadline(), aggregate_mode=settings['aggregate_mode']))
    except NativeTerminal:
        assert environment.done and environment.answer is not None
        answers, log_probability = [], None
        termination = 'native_environment_terminal'
    else:
        assert all(node.outputs for node in graph.nodes.values()) and graph.decision_node.outputs
        termination = 'graph_completed'
    finally:
        RNG.reset(token)
    assert environment.answer is not None
    executed = list(dict.fromkeys(call['node_id'] for call in calls))
    return {'task_id': task['task_id'], 'answer': environment.answer,
            'actions': environment.actions, 'calls': calls, 'raw_answers': answers,
            'nodes': [{'id': node.id, 'role': node.role, 'outputs': node.outputs}
                      for node in graph.nodes.values() if node.id in executed],
            'executed_node_ids': executed,
            'decision_node_outputs': graph.decision_node.outputs,
            'termination': termination,
            'spatial_edges': graph.spatial_adj_matrix.tolist(),
            'temporal_edges': graph.temporal_adj_matrix.tolist(),
            'topology_seed': seed,
            'topology_log_probability': None if log_probability is None else log_probability.detach().item(),
            'topology_log_probability_status': 'not_returned' if log_probability is None else 'returned',
            'provenance': provenance, 'evaluation_adaptations': prompts['evaluation_adaptations']}
