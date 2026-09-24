import asyncio
import json
from pathlib import Path
import unittest

import yaml

from baselines.common.errors import TaskLimitError
from baselines.common.foldagent import Transport


FIXTURE = json.loads(Path('tests/fixtures/baselines/foldagent/common_interface.json').read_text())


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path(FIXTURE['source_settings']).read_text())['settings']
        shared = yaml.safe_load(Path(FIXTURE['shared_config']).read_text())
        self.settings.update(shared['generation'])
        self.calls = []

        def complete(messages, tokens, temperature, return_tokens):
            self.calls.append((messages, tokens, temperature, return_tokens))
            return [FIXTURE['response']]

        self.transport = Transport(complete, None, self.settings)

    def request(self, delta):
        ids = FIXTURE['input_token_ids']
        maximum = len(ids) + self.settings['minimum_completion_tokens'] + delta
        return asyncio.run(self.transport.create_completion(
            ids, uid=FIXTURE['uid'], max_len=maximum, messages=FIXTURE['messages']))

    def test_exhaustion_is_declared_limit_without_model_call(self):
        with self.assertRaises(TaskLimitError):
            self.request(FIXTURE['exhausted_remaining_delta'])
        self.assertFalse(self.calls)
        self.assertFalse(self.transport.calls)

    def test_session_boundary_preserves_shared_generation_limit(self):
        result = self.request(FIXTURE['exact_remaining_delta'])
        self.assertEqual(result['choices'][0]['message']['content'], FIXTURE['response']['text'])
        self.assertEqual(result['choices'][0]['message']['raw_output_ids'], FIXTURE['response']['token_ids'])
        self.assertEqual(self.calls, [(FIXTURE['messages'], self.settings['max_new_tokens'],
                                      self.settings['temperature'], True)])


if __name__ == '__main__':
    unittest.main()
