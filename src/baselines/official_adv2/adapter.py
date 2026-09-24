import asyncio
import importlib
from types import SimpleNamespace

import httpx

from baselines.official_adv2.bridge import CURRENT, ChatTransport, TaskBridge, async_client, embedding_client
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.usage_tracking import TrackedOpenAIChatCompletionClient
from experiments import run_aqua_adversarial_reasoning_pick_metric_reflection as upstream


class CompleteSupervisor(Supervisor):
    def _finding_to_judgement(self, metric, finding):
        assert type(finding['is_flawed']) is bool, 'Verification requires an explicit boolean verdict.'
        return super()._finding_to_judgement(metric, finding)

    def _safe_parse_json(self, text, source_stage):
        try:
            result = super()._safe_parse_json(text, source_stage)
            if source_stage == 'Rerank':
                assert isinstance(result['selected_metrics'], list)
                assert all(isinstance(value, str) for value in result['selected_metrics'])
            return result
        except Exception as error:
            if source_stage == 'Rerank':
                CURRENT.get().failure = f'Indicator selection parse failed. {error}'
            raise

    async def _match_metrics(self, *args, **kwargs):
        result = await super()._match_metrics(*args, **kwargs)
        bridge = CURRENT.get()
        if bridge.failure is not None:
            raise RuntimeError(bridge.failure)
        return result

    def _apply_select_q_policy(self, candidates, selected_names):
        names = {row['name'] for row in candidates}
        if not all(name in names for name in selected_names):
            CURRENT.get().fail('Indicator selection referenced an unavailable candidate.')
        return super()._apply_select_q_policy(candidates, selected_names)

    async def _calc_score(self, *args, **kwargs):
        result = await super()._calc_score(*args, **kwargs)
        expected = kwargs['metrics']
        if CURRENT.get().failure is not None:
            raise RuntimeError(CURRENT.get().failure)
        if len(result) != len(expected):
            CURRENT.get().fail('Verification did not return every selected indicator judgement.')
        return result


def selector_client(**kwargs):
    kwargs.update(http_client=httpx.AsyncClient(transport=ChatTransport()), max_retries=0)
    return TrackedOpenAIChatCompletionClient(**kwargs)


def configure(config):
    upstream.args = SimpleNamespace(**config['upstream_args'])
    upstream.Supervisor = CompleteSupervisor
    upstream.TrackedOpenAIChatCompletionClient = selector_client
    for name in ['math_solver_aqua', 'supervisor_reasoning_pick_metric', 'final_decision']:
        module = importlib.import_module('AgentDropout.agents.' + name)
        module.AsyncOpenAI = async_client
    importlib.import_module('AgentDropout.agents.supervisor_reasoning_pick_metric').OpenAI = embedding_client


def make_task(row, service, embeddings, config, metrics):
    bridge = TaskBridge(row['task_id'], service, embeddings, config['generation_budgets'])
    token = CURRENT.set(bridge)
    try:
        team, final, roles, supervisor = upstream.init_team(metrics, embeddings.index.vectors.numpy())
    finally:
        CURRENT.reset(token)
    question = row['state'] + '\n' + row['fields']['q0']['question'] + '\n' + '\n'.join(
        f"{option['id']}) {option['description']}" for option in row['fields']['q0']['options'])
    return bridge, team, final, roles, supervisor, question


async def run_task(task):
    bridge, team, final, roles, supervisor, question = task
    token = CURRENT.set(bridge)
    try:
        answer, scores, reflections = await upstream.reasoning(question, team, final, roles, supervisor)
        if bridge.failure is not None:
            raise RuntimeError(bridge.failure)
        return {'task_id': bridge.task_id, 'status': 'valid' if answer in 'ABCDE' and len(answer) == 1 else 'invalid_answer',
                'answer': answer, 'roles': roles, 'scores': scores, 'reflections': reflections, 'calls': bridge.calls}
    except Exception as error:
        return {'task_id': bridge.task_id, 'status': 'error', 'error_type': type(error).__name__,
                'error': bridge.failure or str(error), 'upstream_error': str(error),
                'roles': roles, 'calls': bridge.calls}
    finally:
        CURRENT.reset(token)


async def run_tasks(tasks):
    return await asyncio.gather(*(asyncio.to_thread(asyncio.run, run_task(task)) for task in tasks))
