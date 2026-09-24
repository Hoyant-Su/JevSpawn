from functools import partial
from importlib import import_module
import json
import re
import sys
import tokenize
import unittest
from unittest.mock import patch

from baselines.common.dependency_arguments import dependency_bindings
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


FIXTURE = json.loads((ROOT / 'tests/baselines/llmcompiler/dependency_replay.json').read_text())


class DependencyArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        method = json.loads((ROOT / FIXTURE['method_config']).read_text())
        sys.path[:0] = [str((ROOT / path).resolve()) for path in method['python_paths']]
        cls.upstream = import_module('src.llm_compiler.output_parser')
        cls.original_args = staticmethod(cls.upstream._parse_llm_compiler_action_args)
        cls.original_instantiate = staticmethod(cls.upstream.instantiate_task)
        cls.adapter = import_module(method['module'])
        cls.settings = json.loads((ROOT / FIXTURE['candidate_config']).read_text())
        cls.definitions = load_prompt(FIXTURE['tool_prompts'])
        cls.bind = staticmethod(partial(dependency_bindings,
            literal_bindings=cls.adapter.literal_bindings, argument_segments=cls.adapter.argument_segments,
            grammar=cls.adapter.ARGUMENT_GRAMMAR, **cls.settings))
        cls.tools = [cls.adapter.Tool(name=name, func=partial(cls.adapter.invoke,
            environment=None, name=name, fields=list(definition['arguments'])), description=definition['description'])
            for name, definition in cls.definitions.items() if name != 'finish']

    def reference(self, plan):
        with patch.multiple(self.upstream, _parse_llm_compiler_action_args=self.original_args,
                            instantiate_task=self.original_instantiate):
            return self.upstream.LLMCompilerPlanParser(self.tools).parse(plan)

    def calls(self, tasks):
        return [(task.idx, task.name, task.args, task.dependencies, task.is_join)
                for task in tasks.values()]

    def test_all_recorded_plans_preserve_call_ids_and_dependencies(self):
        records = []
        for case in FIXTURE['cases']:
            with self.subTest(task_id=case['task_id']):
                with patch.object(self.adapter, 'literal_bindings', self.bind):
                    candidate = self.adapter.SchemaPlanParser(self.tools).parse(case['plan'])
                named = any(token in case['plan'] for token in ('query:', 'ids:'))
                reference_plan = case['plan']
                if named:
                    reference_plan = '\n'.join(f'{task.idx}. {task.name}(' +
                        ', '.join(map(repr, task.args)) + ')' for task in candidate.values())
                reference = self.reference(reference_plan)
                self.assertEqual(self.calls(candidate), self.calls(reference))
                original_ids = [int(index) for index, _, _, _ in
                                re.findall(self.upstream.ACTION_PATTERN, case['plan'])]
                self.assertEqual(list(candidate), original_ids)
                records.append({'task_id': case['task_id'], 'comparison':
                    'upstream canonical positional serialization of named bindings' if named else
                    'unmodified upstream parser on identical original plan',
                    'calls': self.calls(candidate)})
        self.records = records

    def test_declared_bare_references_and_nonliteral_rejection(self):
        fields = list(self.definitions['read']['arguments'])
        for source in FIXTURE['reference_examples']:
            self.assertEqual(self.bind(source, fields), list(self.original_args(source)))
        for source in FIXTURE['unsupported_expressions']:
            with self.subTest(source=source), self.assertRaises((SyntaxError, ValueError, tokenize.TokenError)):
                self.bind(source, fields)

    def test_existing_literal_and_named_binding_behavior_is_preserved(self):
        literals = json.loads((ROOT / 'tests/baselines/llmcompiler/argument_fixtures.json').read_text())
        for example in literals['valid']:
            self.assertEqual(self.bind(example['source'], literals['fields']), example['expected'])
        for source in literals['invalid']:
            with self.subTest(source=source), self.assertRaises((SyntaxError, ValueError, tokenize.TokenError)):
                self.bind(source, literals['fields'])


if __name__ == '__main__':
    unittest.main()
