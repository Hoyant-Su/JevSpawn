import ast
import asyncio
import copy
from functools import partial
from itertools import groupby
import json
import os
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
from typing import Union
import unicodedata
import uuid
from uuid import uuid4

from baselines.hiagent.adapter import DocumentEnvironment


class TokenizerInterface:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        guard = "{%- if ns.multi_step_tool %}\n    {{- raise_exception('No user query found in messages.') }}\n{%- endif %}"
        assert guard in tokenizer.chat_template
        self.prefix_template = tokenizer.chat_template.replace(guard, '')

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs, enable_thinking=False,
                                                  chat_template=self.prefix_template, return_dict=False)


def rendered_context(agent, turn_cut=None):
    messages = agent.chat if turn_cut is None else agent.chat[:turn_cut]
    return agent.tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def load_core(directory, namespace):
    paths = [Path(directory) / name for name in ['utils.py', 'prompts.py', 'fold_agent.py']]
    utility_names = {'truncate_text', 'is_weird', 'truncate_prompt', 'AgentContext', 'Agent', 'run_action'}
    for path in paths:
        tree = ast.parse(path.read_text())
        if path.name == 'utils.py':
            tree.body = [node for node in tree.body if getattr(node, 'name', None) in utility_names]
        elif path.name == 'prompts.py':
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        else:
            tree.body = [node for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            function = next(node for node in tree.body if node.name == 'process_item')
            boundary = next(index for index, node in enumerate(function.body)
                            if isinstance(node, ast.Assign)
                            and ast.unparse(node.targets[0]) == "env.stats['session_time']")
            function.body = function.body[:boundary + 1] + ast.parse(
                "return {'agents': agent, 'branches': branches, 'environment': env, 'iterations': iteration}"
            ).body
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace


class ModelTransport:
    def __init__(self, complete, tokenizer, settings):
        self.complete = complete
        self.tokenizer = tokenizer
        self.settings = settings
        self.calls = []

    async def create_completion(self, input_ids, *, uid, max_len, messages):
        if len(self.calls) >= self.settings['max_model_calls']:
            raise RuntimeError('Declared model call budget exhausted.')
        remaining = max_len - len(input_ids)
        if remaining < self.settings['minimum_completion_tokens']:
            return None
        output = self.complete(messages, min(remaining, self.settings['max_new_tokens']),
                               self.settings['temperature'])[0]
        self.calls.append({'uid': uid, 'text': output['text'], 'token_ids': output['token_ids']})
        return {'choices': [{'message': {
            'content': output['text'], 'raw_output_ids': output['token_ids'],
            'response_log_probs': [None] * len(output['token_ids'])}}]}


class FoldEnvironment:
    def __init__(self, collection, settings, prompts, parser):
        self.documents = DocumentEnvironment(collection, settings, prompts['environment'])
        self.parser = parser
        self.stats = {}
        self.is_finish = False
        self.instance_info = {'problem_statement': collection['query'] + '\n\n' + self.documents.reset()}

    async def init_env(self, item):
        return None

    async def run_action(self, response):
        call = self.parser(response)
        if call is None:
            return {'observation': 'No function call was detected. Use a declared XML tool.'}
        name = call['function']
        if name not in ['read_documents', 'finish']:
            return {'observation': 'Unknown document environment tool. Use read_documents or finish.'}
        action = name + '(' + call['arguments'].get('ids', '') + ')'
        observation, done = self.documents.step(action)
        self.is_finish = done
        return {'action': 'finish' if done else name, 'observation': observation}


def solve(collection, complete, settings, prompts):
    tokenizer = TokenizerInterface(complete.func.__self__.agent_tokenizers[collection['task_id']])
    client = ModelTransport(complete, tokenizer, settings)
    namespace = {'os': os, 're': re, 'time': time, 'copy': copy, 'asyncio': asyncio,
                 'partial': partial, 'random': random, 'uuid': uuid, 'uuid4': uuid4,
                 'unicodedata': unicodedata, 'groupby': groupby,
                 'DataProto': SimpleNamespace, 'TaskContext': SimpleNamespace,
                 'Union': Union, 'AgentLoopOutput': dict,
                 'print': lambda *args, **kwargs: None}
    core = load_core(settings['source_directory'], namespace)
    core['AgentContext'].context = rendered_context
    environment = FoldEnvironment(collection, settings, prompts, core['extract_fn_call'])
    core['select_env'] = lambda *args: lambda *args: environment
    core['create_chat'] = lambda problem, workflow, item: [
        {'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': problem}]
    plugin = SimpleNamespace(workflow='search_branch', max_turn=settings['max_actions'],
                             max_session=settings['max_sessions'],
                             val_max_session=settings['max_sessions'],
                             session_timeout=settings['session_timeout_seconds'],
                             process_reward=None, enable_summary=False)
    rollout = SimpleNamespace(plugin=plugin, prompt_length=settings['context_length'],
                              response_length=settings['response_length'])
    context = SimpleNamespace(tokenizer=tokenizer, llm_client=client, is_train=False,
                              config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)))
    item = SimpleNamespace(non_tensor_batch={'ability': ['DocumentCollection'],
                                             'extra_info': [{'workflow': 'search_branch'}],
                                             'uid': collection['task_id']})
    result = asyncio.run(core['process_item'](item, context))
    return {'task_id': collection['task_id'],
            'ranked_document_ids': environment.documents.ranking,
            'actions': environment.documents.actions,
            'documents_read': sorted(environment.documents.read_ids),
            'branches': result['branches'], 'iterations': result['iterations'],
            'trajectories': {name: agent.messages() for name, agent in result['agents'].items()},
            'calls': client.calls}
