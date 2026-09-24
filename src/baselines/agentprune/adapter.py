import asyncio
from contextvars import ContextVar
import copy
import importlib
import json


CURRENT = ContextVar('agentprune_task')
SERVICE = None
GENERATION = None
REQUESTS = []


def choice_ids(task):
    assert set(task['fields']) == {'q0'}
    identifiers = [option['id'] for option in task['fields']['q0']['options']]
    assert len(identifiers) >= 2 and len(identifiers) == len(set(identifiers))
    assert all(isinstance(identifier, str) and len(identifier) == 1 for identifier in identifiers)
    return identifiers


class Provider:
    def __init__(self, model_name):
        self.model_name = model_name

    def __deepcopy__(self, memo):
        return self

    async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        assert max_tokens is None and temperature is None and num_comps is None
        request = {'task_id': CURRENT.get(), 'messages': copy.deepcopy(messages),
                   'max_tokens': GENERATION['max_tokens'], 'temperature': GENERATION['temperature'],
                   'status': 'pending'}
        REQUESTS.append(request)
        try:
            answer = (await asyncio.to_thread(SERVICE.complete, messages, GENERATION['max_tokens'],
                        GENERATION['temperature'], n=1, task_id=CURRENT.get()))[0]
            request.update(status='returned', text=answer)
            return answer
        except Exception as error:
            request.update(status='error', error_type=type(error).__name__, error=str(error))
            raise


def configure(service, generation, schema):
    global SERVICE, GENERATION
    SERVICE, GENERATION = service, generation
    REQUESTS.clear()
    registry = importlib.import_module('AgentPrune.llm.llm_registry').LLMRegistry
    registry.get = classmethod(lambda cls, model_name: Provider(model_name))
    prompts = importlib.import_module('AgentPrune.prompt.mmlu_prompt_set').MMLUPromptSet
    prompts.get_decision_constraint = staticmethod(lambda: schema['decision_constraint'])


class Dataset:
    def __init__(self, tasks, labels, batch_size):
        self.rows = [json.loads(line) for line in tasks.read_text().splitlines()]
        self.labels = {row['task_id']: row['labels']['q0'] for row in
                       map(json.loads, labels.read_text().splitlines())}
        assert set(self.labels) == {row['task_id'] for row in self.rows}
        self.batch_size, self.position = batch_size, 0
        self.episodes = []
        self.original = importlib.import_module('dataset.mmlu_dataset').MMLUDataset

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def record_to_input(self, record):
        episode = f"iteration-{self.position // self.batch_size:03d}/{record['task_id']}"
        CURRENT.set(episode)
        self.position += 1
        self.episodes.append(episode)
        options = '\n'.join(f"Option {option['id']}: {option['description']}"
                            for option in record['fields']['q0']['options'])
        return {'task': record['state'] + '\n' + options + '\n'}

    def record_to_target_answer(self, record):
        return self.labels[record['task_id']]

    def postprocess_answer(self, answer):
        return self.original.postprocess_answer(self, answer)
