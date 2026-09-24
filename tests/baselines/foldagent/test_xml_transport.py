import ast
import asyncio
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from baselines.common.foldagent import Environment


FIXTURE = json.loads(Path('tests/fixtures/baselines/foldagent/xml_transport.json').read_text())


class XMLTransportTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse(Path(FIXTURE['parser_source']).read_text())
        function = next(node for node in source.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'extract_fn_call')
        namespace = {'re': re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), FIXTURE['parser_source'], 'exec'), namespace)
        self.parser = namespace['extract_fn_call']
        schemas = json.loads(Path(FIXTURE['input_schemas']).read_text())
        schemas['finish'] = FIXTURE['finish_schema']
        self.environment = SimpleNamespace(input_schemas=schemas, reset=lambda: FIXTURE['reset'],
            execute=Mock(return_value=(FIXTURE['observation'], False)))
        self.adapter = Environment(self.environment, self.parser)

    def test_declared_types_round_trip_through_upstream_parser(self):
        for case in FIXTURE['cases']:
            with self.subTest(tool=case['name']):
                name, arguments = case['name'], case['arguments']
                parameters = ''.join(f'<parameter={key}>' +
                    (value if isinstance(value, str) else json.dumps(value)) + '</parameter>'
                    for key, value in arguments.items())
                self.environment.execute.return_value = (FIXTURE['observation'], name == 'finish')
                result = asyncio.run(self.adapter.run_action(f'<function={name}>{parameters}</function>'))
                payload = arguments['answer'] if name == 'finish' else arguments
                self.environment.execute.assert_called_with(name, payload)
                self.assertEqual(result['observation'], FIXTURE['observation'])
                self.assertEqual(self.adapter.is_finish, name == 'finish')

    def test_wrong_numeric_type_is_visible_and_not_executed(self):
        result = asyncio.run(self.adapter.run_action(FIXTURE['invalid_call']))
        self.assertIn('observation', result)
        self.assertNotIn('action', result)
        self.environment.execute.assert_not_called()

    def test_original_raw_query_is_accepted_without_rewriting(self):
        evidence = FIXTURE['real_request']
        batch = json.loads(Path(evidence['batches']).read_text())[evidence['batch_index']]
        response = batch['texts'][batch['task_ids'].index(evidence['task_id'])]
        call = self.parser(response)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(call['arguments']['query'])
        asyncio.run(self.adapter.run_action(response))
        self.environment.execute.assert_called_once_with(call['function'], {
            'query': call['arguments']['query'], 'k': json.loads(call['arguments']['k'])})


if __name__ == '__main__':
    unittest.main()
