import ast
import json
import logging
from pathlib import Path
import re
import unittest

from baselines.common.lats import SearchInterface
from baselines.common.lats_frames import load_frame_core
from baselines.common.policy_frames import policy_sample


CONFIG = json.loads(Path('configs/baselines/common/candidates/lats_frame_boundary_v1.json').read_text())


class PolicyFrameTests(unittest.TestCase):
    def test_real_core_loader_changes_only_sample_boundary(self):
        saved = json.loads(Path('results/baselines/common/monitoring/lats_action_prefix_failure_001.json').read_text())
        raw = [row['raw_output'] for row in saved['rows']]
        core, _ = load_frame_core(CONFIG['source_directory'], lambda *args, **kwargs: raw,
                                 object(), grammar=CONFIG)
        task = type('ReplayTask', (), {'cot_prompt_wrap': lambda *args: 'Recorded loader replay'})()
        prefix = saved['rows'][0]['injected_prefix']
        actual = core['get_samples'](task, '', prefix, len(raw), 'cot', None)
        self.assertEqual(actual, raw)

    def test_every_recorded_policy_response_preserves_valid_continuations(self):
        batches = json.loads((Path(CONFIG['source_run']) / 'session-0000/batches.json').read_text())
        checked, recovered = [], []
        for batch_index, batch in enumerate(batches):
            for row_index, messages in enumerate(batch['messages']):
                prefix = re.search(CONFIG['continuation_prefix'] + '$', messages[-1]['content'])
                if prefix is None:
                    continue
                raw = batch['texts'][row_index]
                original = prefix.group() + raw
                result = policy_sample(prefix.group(), raw, CONFIG)
                action = lambda text: next((line.split(':', 1)[1].strip() for line in text.splitlines()
                                           if line.startswith('Action') and ':' in line), None)
                if re.match(CONFIG['complete_frame'].format(step=prefix.group('step')), raw):
                    self.assertEqual(result, raw)
                else:
                    self.assertEqual(result, original)
                if action(original) is not None:
                    self.assertEqual(action(result), action(original))
                if action(original) is None and action(result) is not None:
                    recovered.append({'task_id': batch['task_ids'][row_index],
                                      'batch_index': batch_index, 'row_index': row_index})
                checked.append((batch_index, row_index))
        saved = json.loads(Path('results/baselines/common/monitoring/lats_action_prefix_failure_001.json').read_text())
        self.assertEqual(recovered, [{key: row[key] for key in ('task_id', 'batch_index', 'row_index')}
                                    for row in saved['rows']])
        self.assertTrue(checked)

    def test_actual_downstream_core_accepts_action_first_without_invented_thought(self):
        path = Path(CONFIG['source_directory']) / 'hotpot/lats.py'
        source = ast.parse(path.read_text())
        chosen = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                  and node.name in {'Node', 'generate_new_states'}]
        tree = ast.fix_missing_locations(SearchInterface().visit(ast.Module(body=chosen, type_ignores=[])))
        saved = json.loads(Path('results/baselines/common/monitoring/lats_action_prefix_failure_001.json').read_text())
        for row in saved['rows']:
            calls = []
            namespace = {'logging': logging, 'failed_trajectories': [], 'env': object(),
                         'generate_prompt': lambda node: node.question,
                         'get_samples': lambda *args, **kwargs: [policy_sample(row['injected_prefix'], row['raw_output'], CONFIG)],
                         'step': lambda environment, action: (calls.append(action) or '', 0, False, {})}
            exec(compile(tree, str(path), 'exec'), namespace)
            node = namespace['Node'](state={}, question='Recorded parser replay')
            args = type('ReplayArguments', (), {'prompt_sample': 'cot'})()
            children = namespace['generate_new_states'](node, args, None, len(row['raw_action_lines']))
            self.assertEqual(len(children), len(row['raw_action_lines']))
            expected = row['raw_action_lines'][0].split(':', 1)[1].strip()
            self.assertEqual(calls, [expected])
            self.assertEqual(children[0].state['action'], expected)
            self.assertEqual(children[0].state['thought'], '')


if __name__ == '__main__':
    unittest.main()
