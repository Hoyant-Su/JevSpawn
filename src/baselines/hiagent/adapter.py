import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from baselines.common.errors import InputLimitError, InvalidOutputError, TaskLimitError


def original_agent(directory, *, environment=os.environ):
    namespace = {'json': json, 'os': SimpleNamespace(environ=environment), 're': re, 'print': lambda *args, **kwargs: None,
                 'InputLimitError': InputLimitError, 'InvalidOutputError': InvalidOutputError}
    for filename in ['base_agent.py', 'summarize.py', 'cme_final.py']:
        path = Path(directory) / filename
        definitions = [node for node in ast.parse(path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        for node in definitions:
            node.decorator_list = []
        if filename == 'cme_final.py':
            agent = next(node for node in definitions if node.name == 'ContextEfficientAgentV2')
            prompt = next(node for node in agent.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'make_prompt')
            loop = next(node for node in prompt.body if isinstance(node, ast.While))
            # The upstream truncation cannot shrink an oversized fixed goal after history is empty.
            guard = ast.parse('''if not history:
    raise InputLimitError(f"Input has {num_of_tokens} tokens after all removable history is exhausted; limit is {self.max_context_length - self.llm_model.max_tokens}.")''').body
            loop.body = guard + loop.body
            serializer = next(node for node in prompt.body if isinstance(node, ast.FunctionDef)
                              and node.name == 'serialize_history')
            completed = next(node for node in serializer.body if isinstance(node, ast.For)
                             and any(isinstance(item, ast.Assign) and
                                     any(isinstance(target, ast.Name) and target.id == 'obs_index'
                                         for target in item.targets) for item in node.body))
            observation = next(index for index, node in enumerate(completed.body)
                               if isinstance(node, ast.Assign) and
                               any(isinstance(target, ast.Name) and target.id == 'obs_index'
                                   for target in node.targets))
            completed.body[observation + 1:observation + 1] = ast.parse('''if obs_index == index:
    raise InvalidOutputError("Consecutive HiAgent subgoals contain no intervening action or observation.")''').body
            ast.fix_missing_locations(agent)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ContextEfficientAgentV2']


class ModelTransport:
    def __init__(self, complete, settings):
        self.complete = complete
        self.settings = settings
        self.tokenizer = complete.func.__self__.agent_tokenizers[complete.keywords['task_id']]
        self.context_length = settings['context_length'] + settings['max_new_tokens']
        self.max_tokens = settings['max_new_tokens']
        self.engine = complete.func.__self__.backend.config['model_path']
        self.calls = []

    def num_tokens_from_messages(self, messages):
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False))

    def generate(self, system_message, prompt):
        messages = [{'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt}]
        output = self.complete(messages, self.max_tokens, self.settings['temperature'])[0]
        self.calls.append({'messages': messages, 'output': output})
        return True, output


class DocumentEnvironment:
    def __init__(self, collection, settings, prompts):
        self.collection = collection
        self.settings = settings
        self.prompts = prompts
        self.documents = {str(index): row for index, row in enumerate(collection['candidates'], 1)}
        self.ranking = None
        self.actions = []
        self.read_ids = set()

    def reset(self):
        catalog = [{'id': identity, 'retrieval_rank': row['retrieval_rank']}
                   for identity, row in self.documents.items()]
        return self.prompts['initial'].format(catalog=json.dumps(catalog))

    def step(self, action):
        self.actions.append(action)
        if action == self.prompts['check_actions']:
            return self.prompts['commands'], False
        match = re.fullmatch(r'(read_documents|finish)\(([0-9, ]+)\)', action.strip())
        if match is None:
            return self.prompts['invalid_action'], False
        operation, values = match.groups()
        identities = [value.strip() for value in values.split(',')]
        if len(set(identities)) != len(identities) or not set(identities) <= self.documents.keys():
            return self.prompts['invalid_ids'], False
        if operation == 'finish':
            if len(identities) != self.settings['ranking_cutoff']:
                return self.prompts['invalid_ranking'], False
            self.ranking = [self.documents[identity]['document_id'] for identity in identities]
            return self.prompts['finished'], True
        if len(identities) > self.settings['documents_per_read']:
            return self.prompts['read_limit'], False
        self.read_ids.update(identities)
        return json.dumps([{'id': identity, 'text': self.documents[identity]['text']}
                           for identity in identities]), False


def solve(collection, complete, settings, prompts):
    model = ModelTransport(complete, settings)
    environment = DocumentEnvironment(collection, settings, prompts)
    agent = original_agent(settings['source_directory'])(
        model, memory_size=settings['memory_size'], instruction=prompts['instruction'],
        examples=[], system_message=prompts['system'], need_goal=True,
        check_actions=prompts['check_actions'], use_parser=True)
    agent.reset(collection['query'], environment.reset())
    for _ in range(settings['max_actions']):
        success, action = agent.run()
        if not success:
            raise RuntimeError('Original agent failed to generate an action.')
        observation, done = environment.step(action)
        agent.update(action, observation)
        if done:
            break
    return {'task_id': collection['task_id'], 'ranked_document_ids': environment.ranking,
            'actions': environment.actions, 'documents_read': sorted(environment.read_ids),
            'memory': agent.memory, 'calls': model.calls,
            'subgoals': sum(item[0][0] == 'Subgoal' for item in agent.memory)}
