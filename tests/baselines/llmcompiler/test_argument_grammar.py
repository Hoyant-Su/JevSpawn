import ast
import asyncio
from functools import partial
from importlib import import_module
from io import StringIO
import json
import re
import sys
import tokenize
import unittest

from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


FIXTURE = json.loads((ROOT / 'tests/baselines/llmcompiler/argument_fixtures.json').read_text())


class RecordedEnvironment:
    def __init__(self):
        self.input_schemas = json.loads((ROOT / FIXTURE['tool_schema']).read_text())
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({'value': arguments}), False


class ArgumentGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        method = json.loads((ROOT / FIXTURE['method_config']).read_text())
        sys.path[:0] = [str((ROOT / path).resolve()) for path in method['python_paths']]
        cls.adapter = import_module(method['module'])
        cls.fetching = import_module('src.llm_compiler.task_fetching_unit')
        cls.definitions = load_prompt(FIXTURE['tool_prompts'])

    def test_literal_binding_grammar_preserves_nested_values_and_code_strings(self):
        for example in FIXTURE['valid']:
            with self.subTest(source=example['source']):
                self.assertEqual(self.adapter.literal_bindings(example['source'], FIXTURE['fields']), example['expected'])

    def test_ambiguous_or_nonliteral_bindings_fail(self):
        for source in FIXTURE['invalid']:
            with self.subTest(source=source), self.assertRaises((SyntaxError, ValueError, tokenize.TokenError)):
                self.adapter.literal_bindings(source, FIXTURE['fields'])

    def test_four_actual_failing_plans_use_original_parser_and_scheduler(self):
        for source in FIXTURE['actual_plans']:
            with self.subTest(source=source):
                batches = json.loads((ROOT / source['run'] / 'session-0000/batches.json').read_text())
                plan = next(batch['texts'][index] for batch in batches
                            for index, task_id in enumerate(batch['task_ids']) if task_id == source['task_id'])
                matches = re.findall(self.adapter.output_parser.ACTION_PATTERN, plan)
                expected = []
                for index, name, arguments, _ in matches:
                    if name == self.adapter.ARGUMENT_GRAMMAR['join_tool']:
                        continue
                    with self.assertRaises(SyntaxError):
                        ast.parse(f'{name}({arguments})', mode='eval')
                    literals = [json.loads(token.string) for token in tokenize.generate_tokens(StringIO(arguments).readline)
                                if token.type == tokenize.STRING]
                    fields = list(self.definitions[name]['arguments'])
                    expected.append((name, dict(zip(fields, literals, strict=True))))
                environment = RecordedEnvironment()
                tools = [self.adapter.Tool(name=name,
                            func=partial(self.adapter.invoke, environment=environment, name=name,
                                         fields=list(definition['arguments'])), description=definition['description'])
                         for name, definition in self.definitions.items() if name != 'finish']
                tasks = self.adapter.output_parser.LLMCompilerPlanParser(tools).parse(plan)
                for task in tasks.values():
                    if task.is_join:
                        self.assertEqual(set(task.dependencies), set(tasks) - {task.idx})
                    else:
                        self.assertEqual(dict(zip(self.definitions[task.name]['arguments'], task.args, strict=True)),
                                         dict(expected)[task.name])
                async def execute():
                    scheduler = self.fetching.TaskFetchingUnit()
                    scheduler.set_tasks(tasks)
                    await asyncio.wait_for(scheduler.schedule(), timeout=FIXTURE['scheduler_timeout_seconds'])
                asyncio.run(execute())
                self.assertEqual(environment.calls, expected)

    def test_recorded_truncated_joiner_has_no_answer_for_either_parser(self):
        source = FIXTURE['truncated_joiner']
        batches = json.loads((ROOT / source['run'] / 'session-0000/batches.json').read_text())
        batch, index = next((batch, index) for batch in batches
                            for index, task_id in enumerate(batch['task_ids'])
                            if task_id == source['task_id'] and batch['truncated'][index])
        self.assertEqual(batch['finish_reasons'][index], source['finish_reason'])
        self.assertEqual(batch['output_tokens'][index], batch['max_new_tokens'])
        compiler = self.adapter.SchemaCompiler.construct()
        _, answer, replan = self.adapter.LLMCompiler._parse_joinner_output(compiler, batch['texts'][index])
        self.assertFalse(answer)
        self.assertFalse(replan)
        with self.assertRaises(self.adapter.InvalidOutputError):
            compiler._parse_joinner_output(batch['texts'][index])

    def test_schema_invalid_tool_arguments_are_not_executed(self):
        environment = RecordedEnvironment()
        example = FIXTURE['tool_error_case']
        result = asyncio.run(self.adapter.invoke(*example['values'], environment=environment,
                                                 name=example['name'], fields=example['fields']))
        self.assertFalse(environment.calls)
        self.assertIn(str(next(iter(example['values']))), result)


if __name__ == '__main__':
    unittest.main()
