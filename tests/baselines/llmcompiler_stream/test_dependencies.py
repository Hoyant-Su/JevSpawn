import asyncio
from functools import partial
from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

from baselines.tool_agents.tools import calculate
from project_paths import ROOT


FIXTURE = json.loads(Path(__file__).with_name('dependency_fixture.json').read_text())
EVIDENCE = {}


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        method = json.loads((ROOT / FIXTURE['method']).read_text())
        sys.path[:0] = [str((ROOT / path).resolve()) for path in method['python_paths']]
        cls.adapter = import_module(method['module'])
        cls.source = import_module('src.llm_compiler.task_fetching_unit')
        cls.Tool = import_module('src.tools.base').Tool
        cls.protocol_error = import_module('langchain.schema').OutputParserException
        cls.limits = json.loads((ROOT / FIXTURE['environment']).read_text())['calculator']

    def parse(self, text, calls):
        async def invoke(expression, *, fields):
            value = calculate(expression, self.limits)
            calls.append({'expression': expression, 'value': value})
            return str(value[FIXTURE['result_key']])

        tool = self.Tool(name=FIXTURE['tool_name'],
                         func=partial(invoke, fields=FIXTURE['tool_fields']), description='')
        parser = self.adapter.SchemaStreamingGraphParser(tools=[tool])
        tasks = [task for line in text.splitlines(keepends=True)
                 if (task := parser.ingest_token(line)) is not None]
        last = parser.finalize()
        if last is not None:
            tasks.append(last)
        return tasks

    def test_recorded_missing_dependency(self):
        tasks = self.parse(FIXTURE['invalid_plan']['text'], [])
        original = self.source.TaskFetchingUnit()
        original.set_tasks({task.idx: task for task in tasks})
        with self.assertRaises(KeyError) as failure:
            original._get_all_executable_tasks()
        self.assertEqual(failure.exception.args, (FIXTURE['missing_dependency'],))
        checked = self.adapter.DependencyCheckedTaskFetchingUnit()
        with self.assertRaises(self.protocol_error) as failure:
            checked.set_tasks({task.idx: task for task in tasks})
        self.assertFalse(checked.tasks)
        EVIDENCE['actual_invalid_plan'] = FIXTURE['invalid_plan']
        EVIDENCE['explicit_protocol_error'] = str(failure.exception)

    def test_actual_dependency_execution_matches_upstream(self):
        async def execute(constructor):
            calls = []
            tasks = self.parse(FIXTURE['valid_plan']['text'], calls)
            queue = asyncio.Queue()
            for task in tasks:
                await queue.put(task)
            await queue.put(None)
            scheduler = constructor()
            await asyncio.wait_for(scheduler.aschedule(queue, None), FIXTURE['timeout_seconds'])
            return calls, {identity: task.observation for identity, task in scheduler.tasks.items()}

        original = asyncio.run(execute(self.source.TaskFetchingUnit))
        checked = asyncio.run(execute(self.adapter.DependencyCheckedTaskFetchingUnit))
        self.assertEqual(checked, original)
        EVIDENCE['actual_valid_plan'] = FIXTURE['valid_plan']
        EVIDENCE['unchanged_calculator_calls'] = checked[0]
        EVIDENCE['unchanged_observations'] = checked[1]

    def test_upstream_orchestration_is_preserved(self):
        self.assertIs(self.adapter.StreamingSchemaCompiler._acall.__code__,
                      self.adapter.SchemaCompiler._acall.__code__)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DependencyTests))
    (ROOT / FIXTURE['output']).write_text(json.dumps({'passed': result.wasSuccessful(),
        'tests': result.testsRun, 'evidence': EVIDENCE,
        'scope': 'Actual saved model plans replayed through the original parser/scheduler and exact calculator; no model inference.'},
        indent=2) + '\n')
    raise SystemExit(not result.wasSuccessful())
