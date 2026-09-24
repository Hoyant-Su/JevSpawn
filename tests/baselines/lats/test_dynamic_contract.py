from copy import deepcopy
import json
from pathlib import Path
import unittest

from baselines.common import dyflow, lats
from baselines.common.environment import TaskEnvironment
from jev_spawn.infra.prompts import load_prompt
from methods.evidence_flow.environment import EvidenceEnvironment


FIXTURE = json.loads(Path('tests/fixtures/baselines/lats/dynamic_contract.json').read_text())


class DynamicContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment_settings = json.loads(Path(FIXTURE['environment']).read_text())
        cls.corpus = EvidenceEnvironment(**cls.environment_settings['evidence'])
        cls.task = next(json.loads(line) for line in Path(FIXTURE['tasks']).read_text().splitlines()
                        if json.loads(line)['task_id'] == FIXTURE['task_id'])

    def run_method(self, method):
        original = deepcopy(self.task)
        environment = TaskEnvironment(self.task, self.environment_settings, FIXTURE['tool_directory'],
                                      deadline=lambda: None, evidence=self.corpus)
        configuration = json.loads(Path(f'configs/baselines/common/methods/{method}.json').read_text())
        settings = {**configuration['settings'], **FIXTURE['settings_override']}
        prompts = load_prompt(configuration['prompts'])
        checks = []

        def assert_contract(prompt):
            self.assertIn(environment.reset(), prompt)
            checks.append(sorted(environment.observed_source_ids))

        def arguments(tool):
            if tool == 'search':
                return {'query': self.task['input']['query'],
                        'k': self.environment_settings['evidence']['search_limit']}
            if tool == 'read':
                return {'ids': sorted(environment.observed_source_ids)}
            return FIXTURE['finish']

        def complete_lats(messages, tokens, temperature, *, n, stop):
            prompt = ''.join(message['content'] for message in messages)
            assert_contract(prompt)
            if not prompt.startswith(FIXTURE['lats']['policy_prefix']):
                return [FIXTURE['lats']['value_output']] * n
            depth = sum(line.startswith(FIXTURE['lats']['observation_marker']) for line in prompt.splitlines())
            tool = FIXTURE['stages'][depth]
            output = FIXTURE['lats']['action'].format(index=depth + 1, tool=tool,
                                                     arguments=json.dumps(arguments(tool)))
            return [output] * n

        designs = []

        def complete_dyflow(messages, tokens, temperature):
            prompt = messages[0]['content']
            if prompt.startswith(FIXTURE['dyflow']['summary_prefix']):
                return [FIXTURE['dyflow']['summary']]
            if prompt.startswith(FIXTURE['dyflow']['designer_prefix']):
                assert_contract(prompt)
                tool = FIXTURE['stages'][len(designs)]
                designs.append(tool)
                return [json.dumps({'stage_id': FIXTURE['dyflow']['stage_id'].format(index=len(designs)),
                    'stage_description': tool, 'operators': [{
                        'operator_id': FIXTURE['dyflow']['operator_id'].format(index=len(designs)),
                        'operator_description': tool, 'params': {
                            'instruction_type': 'ORGANIZE_SOLUTION' if tool == 'finish' else 'TOOL_CALL',
                            'input_keys': ['original_problem'],
                            'output_key': FIXTURE['dyflow']['action_id'].format(index=len(designs))}}]})]
            assert_contract(prompt)
            tool = designs[-1]
            return [json.dumps(arguments(tool) if tool == 'finish' else
                               {'tool': tool, 'arguments': arguments(tool)})]

        completion = complete_lats if method == 'lats' else complete_dyflow
        solver = lats.solve if method == 'lats' else dyflow.solve
        result = solver(self.task, environment, completion, settings, prompts)
        self.assertEqual(result['answer'], FIXTURE['finish'])
        self.assertEqual(self.task, original)
        self.assertFalse(checks[0])
        self.assertTrue(checks[-1])
        reads = [action for action in environment.actions if action.get('tool') == 'read']
        self.assertTrue(reads)
        for action in reads:
            expected = [{**self.corpus.units[identity], **self.corpus.metadata(self.corpus.units[identity])}
                        for identity in action['arguments']['ids']]
            self.assertEqual(action['result'], expected)

    def test_lats_policy_and_value_refresh_after_actual_search(self):
        self.run_method('lats')

    def test_dyflow_designer_and_executor_refresh_after_actual_search(self):
        self.run_method('dyflow')


if __name__ == '__main__':
    unittest.main()
