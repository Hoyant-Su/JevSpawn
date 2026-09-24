import json
from pathlib import Path
import unittest

from baselines.dyflow.adapter import load_core, solve, validate_selector

from project_paths import ROOT


CONFIG = ROOT / 'configs/baselines/dyflow'
SETTINGS = json.loads((CONFIG / 'development.json').read_text())
PROMPTS = json.loads((CONFIG / 'schema/prompts.json').read_text())


def stage(number, instruction, inputs):
    return json.dumps({'stage_id': f'stage_{number}', 'stage_description': instruction,
                       'operators': [{'operator_id': f'op_{number}_1',
                                      'operator_description': instruction,
                                      'params': {'instruction_type': instruction,
                                                 'input_keys': inputs, 'output_key': f'act_{number}',
                                                 'input_usage': 'Use the supplied inputs.'}}]})


class InterfaceTest(unittest.TestCase):
    def test_real_workflow_state_and_summary(self):
        outputs = iter([stage(0, 'GENERATE_ANSWER', ['original_problem']),
                        'The solution is option A.', 'act_0 gives option A.',
                        stage(1, 'ORGANIZE_SOLUTION', ['original_problem', 'act_0']),
                        '{"answer":"A"}'])
        observed = []

        def complete(messages, max_tokens, temperature):
            observed.append((max_tokens, temperature))
            return [next(outputs)]

        task = {'task_id': 'fixture/0', 'state': '', 'fields': {'q0': {
            'question': 'Which option is named A?', 'options': [{'id': 'A', 'description': 'A'}]}}}
        result = solve(task, complete, SETTINGS, PROMPTS)
        self.assertEqual(result['answer'], 'A')
        self.assertEqual(result['model_calls'], 5)
        self.assertEqual(observed, [(2048, .1), (2048, .1), (2048, .01), (2048, .1), (2048, .1)])
        self.assertEqual(len(result['state']['actions']), 2)
        self.assertEqual(result['state']['_summarized_stage_ids'], ['stage_0'])
        self.assertEqual(len(result['design_history']), 2)

    def test_code_is_disabled_before_execution(self):
        workflow_class = load_core(SETTINGS, PROMPTS, object)
        workflow = workflow_class('fixture', None, None, save_design_history=True)
        operator_class = workflow_class.__init__.__globals__['InstructExecutorOperator']
        client = type('Client', (), {'generate': lambda self, **kwargs: None})()
        operator = operator_class('op', 'fixture', client)
        with self.assertRaises(ValueError):
            operator.execute(workflow.state, {'instruction_type': 'TEST_CODE'})
        with self.assertRaises(ValueError):
            operator._execute_code('raise RuntimeError("must not execute")')

    def test_invalid_selector_has_no_first_candidate_fallback(self):
        for value in ['not JSON', '{}', '{"selected_index":0}', '{"selected_index":3}',
                      '{"selected_index":true}']:
            with self.assertRaises((ValueError, KeyError)):
                validate_selector(value, 2)
        validate_selector('{"selected_index":2}', 2)


if __name__ == '__main__':
    unittest.main()
