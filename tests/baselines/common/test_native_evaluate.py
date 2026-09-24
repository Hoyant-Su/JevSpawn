import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from baselines.common.evaluate import score_task, summarize


class NativeEvaluatorTests(unittest.TestCase):
    """Synthetic artifacts exercise aggregation, not benchmark performance."""

    def test_numeric_score_is_preserved_and_missing_task_counts(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            task = {'task_id': 'numeric', 'dataset': 'unit', 'answer_schema': {
                'type': 'object', 'properties': {'value': {'type': 'number'}}, 'required': ['value']}}
            path = root / 'task.json'
            path.write_text(json.dumps({'task_id': 'numeric', 'status': 'completed',
                                       'answer': {'value': 0.375}, 'elapsed_seconds': 2.0}))
            factory = Mock(return_value=SimpleNamespace(evaluate=lambda answer: answer['value']))
            record = score_task(task, path, factory, {}, {}, root)
            self.assertEqual(record['score'], 0.375)
            self.assertIsNone(record['correct'])
            self.assertIsNone(record['tool_seconds'])
            missing = score_task(task, root / 'absent.json', factory, {}, {}, root)
            summary = summarize([record, missing])
            self.assertEqual(summary['declared_tasks'], 2)
            self.assertEqual(summary['mean_native_score'], 0.1875)
            self.assertIsNone(summary['accuracy'])
            self.assertEqual(missing['score'], 0.0)
            factory.assert_called_once()

    def test_no_answer_is_not_success_and_does_not_call_native_evaluator(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            task = {'task_id': 'no-answer', 'dataset': 'unit', 'answer_schema': {'type': 'object'}}
            path = root / 'task.json'
            path.write_text(json.dumps({'task_id': task['task_id'], 'status': 'completed',
                'answer': None, 'elapsed_seconds': 3.0, 'termination': 'turn_budget'}))
            factory = Mock()
            record = score_task(task, path, factory, {}, {}, root)
            self.assertFalse(record['correct'])
            self.assertFalse(record['valid_answer'])
            self.assertEqual(record['termination'], 'turn_budget')
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
