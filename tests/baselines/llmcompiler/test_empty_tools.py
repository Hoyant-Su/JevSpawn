from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from baselines.official_llmcompiler.planner_compat import support_empty_tools
from project_paths import ROOT


class EmptyToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / 'empty_tools.json').read_text())
        method = json.loads((ROOT / cls.fixture['method_config']).read_text())
        sys.path[:0] = [str((ROOT / path).resolve()) for path in method['python_paths']]
        cls.planner = import_module('src.llm_compiler.planner')
        cls.original = staticmethod(cls.planner.generate_llm_compiler_prompt)
        cls.fixed = staticmethod(support_empty_tools(cls.original))

    def test_original_empty_tool_defect(self):
        with self.assertRaises(UnboundLocalError):
            self.original([], example_prompt=self.fixture['examples'][0])

    def test_nonempty_prefix_is_identical(self):
        for count in self.fixture['tool_counts']:
            tools = [SimpleNamespace(description=self.fixture['tool_description'].format(index=index))
                     for index in range(count)]
            for examples in self.fixture['examples']:
                for replan in self.fixture['replan_modes']:
                    with self.subTest(count=count, replan=replan, examples=examples):
                        self.assertEqual(self.fixed(tools, examples, replan),
                                         self.original(tools, examples, replan))

    def test_empty_prefix_contains_only_native_join(self):
        for examples in self.fixture['examples']:
            for replan in self.fixture['replan_modes']:
                prefix = self.fixed([], examples, replan)
                self.assertIn(f"{self.fixture['join_number']}. {self.planner.JOIN_DESCRIPTION}", prefix)
                self.assertNotIn('fixture_tool', prefix)
                self.assertTrue(prefix.endswith(examples))

    def test_common_adapter_installs_fix_and_parses_join_only(self):
        adapter = import_module('baselines.common.llmcompiler')
        self.assertEqual(self.planner.generate_llm_compiler_prompt([], self.fixture['examples'][0]),
                         self.fixed([], self.fixture['examples'][0]))
        tasks = adapter.SchemaPlanParser(tools=[]).parse(self.fixture['plan'])
        self.assertEqual(list(tasks), [self.fixture['join_number']])
        join = tasks[self.fixture['join_number']]
        self.assertTrue(join.is_join)
        self.assertEqual(join.dependencies, [])


if __name__ == '__main__':
    unittest.main()
