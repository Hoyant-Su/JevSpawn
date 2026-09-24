import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer
import yaml

from baselines.common import hiagent
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


class PromptCaptured(Exception):
    pass


class ContextContractTest(unittest.TestCase):
    def test_recorded_document_observation_retains_task_contract(self):
        shared = yaml.safe_load((ROOT / 'configs/shared_config_arena_v2.yaml').read_text())
        settings = json.loads((ROOT / 'configs/baselines/common/methods/hiagent.json').read_text())['settings']
        settings |= {'context_length': shared['model']['max_input_tokens'],
                     'max_new_tokens': shared['generation']['max_new_tokens']}
        batches = json.loads((ROOT / 'runs/common-hiagent-all-datasets-gen4096-001/session-0000/batches.json').read_text())
        records = [(batch['messages'][index], batch['texts'][index])
                   for batch in batches for index, identity in enumerate(batch['task_ids'])
                   if identity == 'bright_pony/3'
                   and batch['messages'][index][0]['content'].startswith('You are a problem-solving agent.')]
        first, second = records[:2]
        initial = first[0][-1]['content'].split('\nObservation: ', 1)[1]
        observation = second[0][-1]['content'].rsplit('\nObservation: ', 1)[1]
        _, tool_text = initial.split('\n\nAvailable tools:\n')
        tools = json.loads(tool_text)
        tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)

        class ReplayModel:
            def __init__(self):
                self.context_length = settings['context_length']
                self.max_tokens = settings['max_new_tokens']
                self.calls = []

            def num_tokens_from_messages(self, messages):
                return len(tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False))

            def generate(self, system_message, prompt):
                first_call = not self.calls
                self.calls.append([{'role': 'system', 'content': system_message},
                                   {'role': 'user', 'content': prompt}])
                if first_call:
                    return True, first[1]
                raise PromptCaptured

        model = ReplayModel()
        environment = SimpleNamespace(
            answer=None, reset=lambda: initial, execute=lambda name, arguments: (observation, False),
            tool_definitions={name: {key: value for key, value in definition.items()
                                    if key != 'input_schema'} for name, definition in tools.items()},
            input_schemas={name: definition['input_schema'] for name, definition in tools.items()})
        backend_owner = SimpleNamespace(backend=SimpleNamespace(config={'model_path': shared['model']['path']}))
        complete = SimpleNamespace(func=SimpleNamespace(__self__=backend_owner))
        prompts = load_prompt('configs/baselines/common/schema/hiagent.json')
        with patch.object(hiagent, 'ModelTransport', return_value=model), patch.dict(os.environ, {'EVALTASK': 'common'}):
            with self.assertRaises(PromptCaptured):
                hiagent.solve({}, environment, complete, settings, prompts)
        for messages in model.calls:
            self.assertEqual(messages[-1]['content'].count(initial), len([initial]))
            self.assertLessEqual(model.num_tokens_from_messages(messages), settings['context_length'])
        self.assertNotIn('"catalog"', second[0][-1]['content'])
        self.assertIn(observation, model.calls[-1][-1]['content'])
        self.assertEqual(model.context_length - model.max_tokens, settings['context_length'])


if __name__ == '__main__':
    unittest.main()
