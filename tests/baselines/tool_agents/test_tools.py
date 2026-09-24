import json
from pathlib import Path
import unittest

from baselines.tool_agents.agents import context
from baselines.tool_agents.dag import execute_plan, validate
from baselines.tool_agents.tools import ActionError, calculate, execute

from project_paths import ROOT


CONFIG_ROOT = ROOT / 'configs/baselines/tool_agents'
CONFIG = json.loads((CONFIG_ROOT / 'schemas/config.json').read_text())
PROMPTS = json.loads((CONFIG_ROOT / 'schemas/prompts.json').read_text())


class ToolsTest(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculate('sqrt(81) + 2 * (3 - 1)', CONFIG['calculator']), {'value': 13})
        self.assertEqual(calculate('2 ** 3 / 4', CONFIG['calculator']), {'value': 2})
        for expression in ["__import__('os').system('echo unsafe')", "open('file','w')", '(1).__class__', '[1][0]', '2 ** 1000']:
            with self.assertRaises(ActionError):
                calculate(expression, CONFIG['calculator'])

    def test_evidence(self):
        state = 'Control group: 12 patients.\n\nTreatment group: 20 patients.'
        self.assertEqual(execute('search', {'query': 'TREATMENT'}, state, CONFIG['calculator']),
                         {'matches': [{'index': 1, 'text': 'Treatment group: 20 patients.'}]})
        self.assertEqual(execute('read', {'index': 0}, state, CONFIG['calculator'])['text'], state.split('\n\n')[0])
        with self.assertRaises(ActionError):
            execute('shell', {'command': 'pwd'}, state, CONFIG['calculator'])

    def test_dag_dependencies(self):
        nodes = [
            {'id': 'a', 'tool': 'calculator', 'arguments': {'expression': '2 + 3'}, 'depends_on': []},
            {'id': 'b', 'tool': 'calculator', 'arguments': {'expression': 'sqrt(16)'}, 'depends_on': []},
            {'id': 'c', 'tool': 'calculator', 'arguments': {'expression': '${a.value} * ${b.value}'}, 'depends_on': ['a', 'b']},
        ]
        trace = []
        result = execute_plan(nodes, '', CONFIG, PROMPTS['tools'], trace)
        self.assertEqual(result['c'], {'value': 20})
        events = {row['node']: row for row in trace if row['event'] == 'observation'}
        self.assertGreaterEqual(events['c']['started'], max(events[node]['finished'] for node in ['a', 'b']))
        self.assertEqual([row['node'] for row in trace[:2]], ['a', 'b'])

    def test_search_read_dependency(self):
        nodes = [
            {'id': 'a', 'tool': 'search', 'arguments': {'query': 'second'}, 'depends_on': []},
            {'id': 'b', 'tool': 'read', 'arguments': {'index': {'ref': 'a', 'path': ['matches', 0, 'index']}}, 'depends_on': ['a']},
        ]
        self.assertEqual(execute_plan(nodes, 'First.\n\nSecond.', CONFIG, PROMPTS['tools'], [])['b']['text'], 'Second.')

    def test_history_excludes_measurement_arrays(self):
        task = {'state': 'Original problem.', 'fields': {'q0': {'question': 'Select answer.', 'options': []}}}
        trace = [{'event': 'generation', 'text': '{"thought":"Compute 2+2."}',
                  'decode': {'inter_token_seconds': [0.001] * 512}},
                 {'event': 'observation', 'tool': 'calculator', 'observation': {'value': 4}, 'started': 123}]
        prompt = context(task, PROMPTS, trace)
        self.assertIn('Compute 2+2.', prompt)
        self.assertIn('"value": 4', prompt)
        self.assertNotIn('inter_token_seconds', prompt)
        self.assertNotIn('started', prompt)

    def test_direct_join(self):
        trace = []
        self.assertEqual(execute_plan([], 'Supplied evidence.', CONFIG, PROMPTS['tools'], trace), {})
        self.assertEqual(trace, [])

    def test_invalid_dependency(self):
        node = {'id': 'a', 'tool': 'calculator', 'arguments': {'expression': '${missing.value}'}, 'depends_on': []}
        with self.assertRaises(ActionError):
            validate([node], PROMPTS['tools'], CONFIG['compiler_max_nodes'])


if __name__ == '__main__':
    unittest.main()
