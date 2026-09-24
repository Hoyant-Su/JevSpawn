import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, Mock
import time

from baselines.common.config import SharedConfig
from baselines.common.foldagent import Transport
from project_paths import ROOT


class SharedBudgetTests(IsolatedAsyncioTestCase):
    async def test_transport_uses_shared_output_cap_after_local_session_budget(self):
        method = json.loads((ROOT / 'configs/baselines/common/methods/foldagent.json').read_text())
        settings = SharedConfig.load(ROOT / 'configs/shared_config_tp4_v2.yaml').method_settings(method['settings'])
        complete = Mock(return_value=[{'text': 'test response', 'token_ids': [1], 'finish_reason': 'stop'}])
        transport = Transport(complete, None, settings)
        messages = [{'role': 'user', 'content': 'Budget boundary test'}]
        await transport.create_completion([1, 2], uid='test', max_len=1, messages=messages)
        complete.assert_called_once_with(messages, 2048, 0.0, return_tokens=True)
        self.assertNotIn('minimum_completion_tokens', settings)
        self.assertNotIn('control_iterations', settings)

    async def test_original_branch_budget_still_returns_a_summary(self):
        method = json.loads((ROOT / 'configs/baselines/common/methods/foldagent.json').read_text())
        source = ROOT / method['settings']['source_directory'] / 'utils.py'
        tree = ast.parse(source.read_text())
        agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Agent')
        react = next(node for node in agent.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'react')
        namespace = {'time': time}
        exec(compile(ast.Module(body=[react], type_ignores=[]), str(source), 'exec'), namespace)
        branch = SimpleNamespace(config=SimpleNamespace(response_length=method['settings']['session_token_budget']),
            prompt_turn=1, chat=[{'role': 'user', 'content': 'Branch task'}],
            context=lambda turn_cut=None: [] if turn_cut is not None else [0] * 6000,
            rollback=Mock(), append=Mock(), step=AsyncMock(return_value='test summary'))
        run_action = AsyncMock()
        result = await namespace['react'](branch, run_action, max_turn=36, session_timeout=300,
                                          summary_prompt='Return branch progress.')
        self.assertEqual(result, {'last_response': 'test summary', 'iteration': 0})
        branch.rollback.assert_called_once_with(k=2)
        branch.step.assert_awaited_once_with(max_new_tokens=4096)
        run_action.assert_not_awaited()


if __name__ == '__main__':
    main()
