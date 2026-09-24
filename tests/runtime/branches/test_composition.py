import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch
import jsonschema

from jev_spawn.algo.composition import rank_extensions
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration


ROOT = Path(__file__).resolve().parents[3]


class CompositionTests(unittest.TestCase):
    def test_conditional_extensions_preserve_parent_bindings(self):
        conditionals = [torch.tensor([.1, .9], dtype=torch.float64),
                        torch.tensor([.95, .05], dtype=torch.float64)]
        prefix = torch.tensor([.6, .4], dtype=torch.float64).log()
        parents, values, scores = rank_extensions(conditionals, prefix, 4)
        self.assertEqual(list(zip(parents, values)), [(0, 1), (1, 0), (0, 0), (1, 1)])
        torch.testing.assert_close(scores.exp(), torch.tensor([.54, .38, .06, .02], dtype=torch.float64))

    def test_field_values_fill_native_syntax_without_decoding(self):
        settings = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']['rollout']
        declaration = {'fields': [
            {'id': 'operator', 'question': 'Operation?', 'values': [
                '(combine ${left} ${right})', '(inspect ${left})']},
            {'id': 'left', 'question': 'Left object?', 'values': ['a', 'b']},
            {'id': 'right', 'question': 'Right object?', 'values': ['a', 'b']}],
            'action': {'tool': 'execute', 'arguments': {'action': '${operator}'}},
            'answer': {'calls': {settings['execution']['history_reference']: ['arguments', 'action']}}}
        jsonschema.validate(declaration, settings['declaration_schema'])
        schema = compile_declaration(declaration, settings['execution'])
        execution = QueryExecution('Generic protocol test.', Mock(), 'test', settings['execution'])
        execution.values = {'operator': schema['fields'][0]['values'][0], 'left': 'a', 'right': 'b'}
        self.assertEqual(execution.materialize(schema['action']),
                         {'tool': 'execute', 'arguments': {'action': '(combine a b)'}})
        execution.values['operator'] = schema['fields'][0]['values'][1]
        self.assertEqual(execution.materialize(schema['action']),
                         {'tool': 'execute', 'arguments': {'action': '(inspect a)'}})


if __name__ == '__main__':
    unittest.main()
