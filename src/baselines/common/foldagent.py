import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import jsonschema

from environments.arguments import validate_arguments

from baselines.foldagent import adapter as original
from baselines.common.errors import InputLimitError, TaskLimitError
from baselines.common.resources import ADAPTER_SETTINGS, TEMPLATES
from baselines.tool_agents.tools import ActionError


def complete_prompt(chat, prompt_length, tokenizer, prompt_turn):
    length = len(tokenizer.apply_chat_template(chat, add_generation_prompt=True))
    if length > prompt_length:
        raise InputLimitError(f'Input has {length} tokens; limit is {prompt_length}.')
    return chat


class Transport(original.ModelTransport):
    async def create_completion(self, input_ids, *, uid, max_len, messages):
        output = (await asyncio.to_thread(self.complete, messages,
            self.settings['max_new_tokens'], self.settings['temperature'], return_tokens=True))[0]
        self.calls.append({'uid': uid, **output})
        return {'choices': [{'message': {'content': output['text'], 'raw_output_ids': output['token_ids'],
                'response_log_probs': [None] * len(output['token_ids'])}}]}


class Environment:
    def __init__(self, environment, parser):
        self.environment, self.parser = environment, parser
        self.stats, self.is_finish = {}, False
        self.instance_info = {'problem_statement': environment.reset()}

    async def init_env(self, item):
        self.environment.deadline()

    async def run_action(self, response):
        call = self.parser(response)
        if call is None:
            feedback = self.environment.reject(None, response, ActionError(TEMPLATES['missing_xml_call']))
            return {'observation': json.dumps(feedback)}
        name = call['function']
        if name not in self.environment.input_schemas:
            observation, done = await asyncio.to_thread(self.environment.execute, name, call['arguments'])
            return {'action': name, 'observation': observation}
        try:
            schema = ({'type': 'object', 'properties': {'answer': self.environment.input_schemas['finish']},
                       'required': ['answer'], 'additionalProperties': False} if name == 'finish'
                      else self.environment.input_schemas[name])
            arguments = {key: value if key not in schema['properties'] or schema['properties'][key]['type'] == 'string' else json.loads(value)
                         for key, value in call['arguments'].items()}
            validate_arguments(arguments, schema)
            payload = arguments['answer'] if name == 'finish' else arguments
        except (json.JSONDecodeError, jsonschema.ValidationError) as error:
            feedback = self.environment.reject(name, call['arguments'], error)
            return {'observation': json.dumps(feedback)}
        observation, done = await asyncio.to_thread(self.environment.execute, name, payload)
        self.is_finish = done
        return {'action': 'finish' if done else name, 'observation': observation}


async def run_action(environment, response):
    result = await environment.run_action(response)
    return None if result.get('action') == 'finish' else result['observation']


def solve(task, environment, complete, settings, prompts):
    tokenizer = original.TokenizerInterface(deepcopy(complete.func.__self__.agent_tokenizers[task['task_id']]))
    return solve_with_tokenizer(task, environment, complete, settings, prompts, tokenizer)


def solve_with_tokenizer(task, environment, complete, settings, prompts, tokenizer):
    client = Transport(complete, tokenizer, settings)
    names = ['os', 're', 'time', 'copy', 'asyncio', 'partial', 'random', 'uuid', 'uuid4',
             'unicodedata', 'groupby', 'Union']
    namespace = {name: getattr(original, name) for name in names}
    namespace.update(DataProto=SimpleNamespace, TaskContext=SimpleNamespace, AgentLoopOutput=dict,
                     print=lambda *args, **kwargs: None)
    core = original.load_core(settings['source_directory'], namespace)
    core['AgentContext'].context = original.rendered_context
    core['truncate_prompt'] = complete_prompt
    core['run_action'] = run_action
    adapter = Environment(environment, core['extract_fn_call'])
    core['select_env'] = lambda *args: lambda *args: adapter
    core['create_chat'] = lambda problem, workflow, item: [
        {'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': problem}]
    plugin = SimpleNamespace(workflow=ADAPTER_SETTINGS['foldagent']['workflow'], max_turn=settings['max_turns'],
        max_session=settings['max_sessions'], val_max_session=settings['max_sessions'],
        session_timeout=environment.deadline(), process_reward=ADAPTER_SETTINGS['foldagent']['process_reward'],
        enable_summary=ADAPTER_SETTINGS['foldagent']['enable_summary'])
    rollout = SimpleNamespace(plugin=plugin, prompt_length=settings['context_length'],
                              response_length=settings['session_token_budget'])
    context = SimpleNamespace(tokenizer=tokenizer, llm_client=client, is_train=False,
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)))
    item = SimpleNamespace(non_tensor_batch={'ability': ['CommonTask'],
        'extra_info': [{'workflow': ADAPTER_SETTINGS['foldagent']['workflow']}], 'uid': task['task_id']})
    result = asyncio.run(core['process_item'](item, context))
    if environment.answer is None:
        environment.deadline()
        raise TaskLimitError('FoldAgent terminated within its declared limits without submitting an answer.')
    return {'task_id': task['task_id'], 'answer': environment.answer, 'actions': environment.actions,
            'branches': result['branches'], 'iterations': result['iterations'],
            'trajectories': {name: agent.messages() for name, agent in result['agents'].items()},
            'calls': client.calls}
