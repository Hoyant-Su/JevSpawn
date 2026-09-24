from copy import deepcopy
import unittest

from baselines.common.lats import EvaluatorPrompt, load_core
from baselines.common.tasks import read
from jev_spawn.infra.prompts import load_prompt
from tests.baselines.lats.value_role_requests import assessment, source_request


class ValueRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = read('configs/tests/baselines/lats/value_role_replay.json')
        cls.method = read(cls.settings['method'])
        cls.prompts = load_prompt(cls.method['prompts'])

    def test_real_failed_request_preserves_every_character(self):
        prompt, recorded = source_request(self.settings, self.prompts)
        self.assertEqual(str(prompt), recorded['messages'][0]['content'])
        context = str(prompt)[len(self.prompts['value_instruction']):]
        self.assertEqual(prompt.messages[1]['content'], self.prompts['value_role_context'].format(context=context))
        self.assertEqual([row['role'] for row in prompt.messages], ['system', 'user'])
        self.assertEqual(prompt.messages[0]['content'], self.prompts['value_role_system'].format(
            instruction=self.prompts['value_instruction']))
        self.assertEqual(self.prompts['value'], self.prompts['value_instruction'] + self.prompts['value_context'])
        self.assertEqual({str(prompt): recorded['texts']}[prompt], recorded['texts'])
        _, task = load_core(self.method['settings']['source_directory'], None, None)
        self.assertEqual(assessment(recorded['texts'], task)['upstream_value'],
                         task.value_outputs_unwrap([recorded['texts']]))

    def test_upstream_value_call_preserves_typed_prompt_and_cache(self):
        captured = []

        def complete(prompt, n, stop):
            self.assertIsInstance(prompt, EvaluatorPrompt)
            captured.append(prompt)
            return [self.fixture['value_template'].format(score=self.fixture['score'])]

        self.fixture = read('tests/fixtures/baselines/lats/common_interface.json')
        core, original = load_core(self.method['settings']['source_directory'], complete, None)
        prompt, _ = source_request(self.settings, self.prompts)

        class Task(original):
            def value_prompt_wrap(self, *args):
                return prompt

            def value_outputs_unwrap(self, outputs):
                return assessment(outputs[0], original)['upstream_value']

        task = Task()
        before = deepcopy(prompt.messages)
        first = core['get_value'](task, '', '', self.method['settings']['evaluation_samples'])
        cached = core['get_value'](task, '', '', self.method['settings']['evaluation_samples'])
        self.assertEqual(first, cached)
        self.assertEqual(len(captured), self.settings['expected_output_samples'])
        self.assertEqual(prompt.messages, before)


if __name__ == '__main__':
    unittest.main()
