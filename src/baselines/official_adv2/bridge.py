import asyncio
from contextvars import ContextVar
import json
import re
import time

import httpx
from openai import AsyncOpenAI, OpenAI


CURRENT = ContextVar('adv2_task_bridge')


class ChatTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        await request.aread()
        return await CURRENT.get().chat(request)


class EmbeddingTransport(httpx.BaseTransport):
    def handle_request(self, request):
        request.read()
        return CURRENT.get().embed(request)


class TaskBridge:
    def __init__(self, task_id, service, embeddings, budgets):
        self.task_id, self.service, self.embeddings, self.budgets = task_id, service, embeddings, budgets
        assert budgets['mode'] in {'ceiling', 'upstream'}
        assert budgets['length_finish'] in {'reject', 'continue'}
        assert budgets['summary_parser'] in {'strict', 'upstream'}
        self.calls, self.failure = [], None

    def fail(self, message):
        self.failure = message
        raise RuntimeError(message)

    async def chat(self, request):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        payload = json.loads(request.content)
        assert payload.get('n', 1) == 1, 'The recovered ADv2 requests one completion per call.'
        budget = payload.get('max_tokens', self.budgets['unspecified'])
        if self.budgets['mode'] == 'ceiling':
            budget = min(budget, self.budgets['ceiling'])
        call_id = f'{self.task_id}/call-{len(self.calls):04d}'
        self.calls.append({'call_id': call_id, 'request': payload})
        try:
            texts = await asyncio.to_thread(self.service.complete, payload['messages'], budget,
                payload.get('temperature', 1.0), n=1, stop=payload.get('stop'), task_id=call_id)
        except Exception as error:
            self.fail(f'Model request failed. {type(error).__name__}. {error}')
        record = next(row for row in reversed(self.service.records) if call_id in row['task_ids'])
        index = record['task_ids'].index(call_id)
        self.calls[-1].update(text=texts[0], truncated=record['truncated'][index],
                             input_tokens=record['input_tokens'][index], output_tokens=record['output_tokens'][index])
        if record['truncated'][index] and self.budgets['length_finish'] == 'reject':
            self.fail(f'Generation exhausted the declared {budget} token limit at {call_id}.')
        if payload.get('max_tokens') == 1000 and self.budgets['summary_parser'] == 'strict':
            match = re.search(r'\{.*\}', texts[0], re.DOTALL)
            try:
                summary = json.loads(match.group(0))
                assert all(isinstance(summary[key], list) and summary[key] and
                           all(isinstance(value, str) for value in summary[key])
                           for key in ['problem_scenario', 'agent_action'])
            except (AttributeError, KeyError, ValueError, AssertionError) as error:
                self.fail(f'Retrieval summary interface failed. {error}')
        prompt_tokens, completion_tokens = record['input_tokens'][index], record['output_tokens'][index]
        return httpx.Response(200, json={'id': call_id, 'object': 'chat.completion',
            'created': int(time.time()), 'model': payload['model'],
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': texts[0]},
                         'finish_reason': 'length' if record['truncated'][index] else 'stop'}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens}})

    def embed(self, request):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        payload = json.loads(request.content)
        result = self.embeddings.complete(payload['input'], self.task_id)
        return httpx.Response(200, json={'object': 'list', 'model': payload['model'],
            'data': [{'object': 'embedding', 'index': i, 'embedding': vector}
                     for i, vector in enumerate(result['vectors'])],
            'usage': {'prompt_tokens': result['input_tokens'], 'total_tokens': result['input_tokens']}})


def async_client(**kwargs):
    kwargs.update(http_client=httpx.AsyncClient(transport=ChatTransport()), max_retries=0)
    return AsyncOpenAI(**kwargs)


def embedding_client(**kwargs):
    kwargs.update(http_client=httpx.Client(transport=EmbeddingTransport()), max_retries=0)
    return OpenAI(**kwargs)
