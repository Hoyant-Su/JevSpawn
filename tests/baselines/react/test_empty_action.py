import ast
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from baselines.common.react import EnvironmentAdapter
from baselines.official_react.adapter import original_function


CONFIG = json.loads(Path(__file__).with_name('empty_action.json').read_text())
PROMPTS = json.loads(Path(CONFIG['prompts']).read_text())


def fixture(outputs):
    queue = deque(outputs)
    environment = SimpleNamespace(answer=None, reset=lambda: CONFIG['question'])
    calls, executed = [], []

    def execute(name, arguments):
        executed.append((name, arguments))
        environment.answer = arguments
        return '', True

    def llm(prompt, stop):
        calls.append({'prompt': prompt, 'stop': stop})
        return queue.popleft()

    environment.execute = execute
    adapter = EnvironmentAdapter(environment, PROMPTS)
    namespace = {'env': adapter, 'llm': llm, 'step': lambda env, action: env.step(action),
                 'webthink_prompt': PROMPTS['instruction']}
    return namespace, environment, calls, executed


class EmptyActionTests(unittest.TestCase):
    def test_original_empty_action_failure_is_reproduced(self):
        namespace, _, _, _ = fixture([CONFIG['first_response'], CONFIG['empty_output']])
        cells = json.loads(Path(CONFIG['notebook']).read_text())['cells']
        source = next(''.join(cell['source']) for cell in cells
                      if cell['cell_type'] == 'code' and '\ndef webthink(' in ''.join(cell['source']))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'webthink')
        exec(compile(ast.Module(body=[function], type_ignores=[]), CONFIG['notebook'], 'exec'), namespace)
        with self.assertRaises(IndexError):
            namespace['webthink'](idx=CONFIG['task_id'], prompt=PROMPTS['instruction'], to_print=False)

    def test_empty_action_uses_existing_invalid_observation_then_real_next_action(self):
        namespace, environment, calls, executed = fixture([
            CONFIG['first_response'], CONFIG['empty_output'], CONFIG['second_response']])
        core = original_function(CONFIG['notebook'], namespace)
        _, result = core(idx=CONFIG['task_id'], prompt=PROMPTS['instruction'], to_print=False)
        self.assertEqual(environment.answer, CONFIG['expected_answer'])
        self.assertEqual(len(executed), CONFIG['expected_calls_valid'])
        self.assertEqual(len(calls), CONFIG['expected_calls_empty'])
        self.assertEqual(result['n_badcalls'], CONFIG['expected_retries_empty'])
        self.assertIn(PROMPTS['invalid_action'], result['traj'])

    def test_nonempty_action_preserves_native_parse_and_call_count(self):
        namespace, environment, calls, executed = fixture([CONFIG['valid_response']])
        core = original_function(CONFIG['notebook'], namespace)
        _, result = core(idx=CONFIG['task_id'], prompt=PROMPTS['instruction'], to_print=False)
        self.assertEqual(environment.answer, CONFIG['expected_answer'])
        self.assertEqual(len(calls), CONFIG['expected_calls_valid'])
        self.assertEqual(result['n_badcalls'], CONFIG['expected_retries_valid'])
        self.assertEqual(executed, [('finish', CONFIG['expected_answer'])])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EmptyActionTests))
    Path(CONFIG['result']).write_text(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun,
        'scope': 'Actual upstream notebook control, explicit CPU protocol fixtures only; no model inference or task quality claim.'}, indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
