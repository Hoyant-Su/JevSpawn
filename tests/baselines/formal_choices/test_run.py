import asyncio
import json
from pathlib import Path
from threading import Lock
import time
import unittest

from baselines.formal_choices.run import blocks, completed, evaluate, execute_block, flat_task, load_protocol, save
from baselines.official_react.adapter import MathEnvironment


ROOT = Path(__file__).resolve().parents[3]


class CPUFixtureService:
    def __init__(self):
        self.records = []
        self.lock = Lock()

    def complete(self, messages, max_tokens, temperature, n, stop, task_id):
        with self.lock:
            self.records.append({'task_ids': [task_id], 'batch_size': 1, 'output_tokens': [1],
                                 'truncated': [False], 'peak_allocated_bytes': 0,
                                 'decode': [{'inter_token_seconds': []}]})
        return ['fixture response']


def fixture_task(index):
    return {'task_id': f'cpu_fixture/{index}', 'state': 'CPU interface fixture.',
            'fields': {'q0': {'question': 'Fixture choice', 'options': [
                {'id': choice, 'description': choice} for choice in 'ABCD']}}}


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT.parents[1] / 'runtime/research_v1/formal_runner_cpu' / str(time.time_ns())
        cls.directory.mkdir(parents=True)

    def protocol(self, directory, interface='sync_fields'):
        tasks = [fixture_task(0), fixture_task(1)]
        labels = directory / 'labels.jsonl'
        labels.write_text(''.join(json.dumps({'task_id': t['task_id'], 'labels': {'q0': 'A'}}) + '\n'
                                  for t in tasks))
        return {'tasks': tasks, 'native': {'seed': 0}, 'prompts': {},
                'settings': {'dataset': 'cpu_fixture', 'block_size': 8, 'interface': interface,
                             'adapter_config': {}, 'labels': str(labels), 'budget': 'CPU fixtures only'}}

    def test_resume_boundary_and_failure_denominator(self):
        directory = self.directory / 'resume'
        directory.mkdir()
        protocol = self.protocol(directory)
        save(directory / 'protocol.json', protocol)
        block_dir = directory / 'block-0000'
        (block_dir / 'attempt-0000').mkdir(parents=True)
        save(block_dir / 'complete.json.partial', {'incomplete': True})
        self.assertIsNone(completed(block_dir, ['cpu_fixture/0', 'cpu_fixture/1'], 1))
        with self.assertRaisesRegex(AssertionError, 'Incomplete block'):
            evaluate(protocol, directory)

        def solve(task, complete, config, prompts):
            complete([{'role': 'user', 'content': task['state']}], 16, 0)
            if task['task_id'].endswith('/1'):
                raise ValueError('Deliberate adapter failure in CPU fixture')
            return {'task_id': task['task_id'], 'answer': 'A'}

        block = execute_block(CPUFixtureService(), solve, protocol['tasks'], protocol, block_dir, 1)
        self.assertEqual(block['attempt'], 'attempt-0001')
        self.assertEqual(len(block['results'][1]['requests']), 1)
        self.assertEqual(block['results'][1]['status'], 'failed')
        self.assertEqual(completed(block_dir, block['task_ids'], 1), block)
        with self.assertRaises(AssertionError):
            completed(block_dir, block['task_ids'], 2)
        result = evaluate(protocol, directory)
        self.assertEqual((result['tasks'], result['correct'], result['valid']), (2, 1, 1))
        self.assertEqual(result['adapter_exceptions'], 1)

    def test_async_flat_interface(self):
        directory = self.directory / 'async'
        directory.mkdir()
        protocol = self.protocol(directory, 'async_flat')

        async def solve(task, complete, config):
            self.assertEqual(set(task), {'task_id', 'state', 'question', 'options'})
            await asyncio.to_thread(complete, [{'role': 'user', 'content': task['options']}], 16, 0)
            return {'task_id': task['task_id'], 'answer': 'D'}

        result = execute_block(CPUFixtureService(), solve, protocol['tasks'], protocol, directory / 'block', 1)
        self.assertEqual([r['answer'] for r in result['results']], ['D', 'D'])

    def test_real_dataset_configs_and_variable_choices(self):
        for config in (ROOT / 'configs/baselines/formal_choices/configs').glob('*.json'):
            protocol = load_protocol(config)
            sizes = list(map(len, blocks(protocol)))
            self.assertEqual(sum(sizes), protocol['settings']['task_count'])
            self.assertEqual(sizes[-1], 6 if protocol['settings']['dataset'] == 'aqua' else 8)
            self.assertEqual(len(protocol['warmup']),
                             5 if protocol['settings']['dataset'] == 'medxpertqa_text' else 8)
            if protocol['settings']['adapter_module'] == 'baselines.official_react.adapter':
                for task in protocol['tasks']:
                    rendered = MathEnvironment(task, protocol['prompts'],
                                               protocol['settings']['adapter_config']['tool_limits']).reset()
                    flat = flat_task(task)
                    for option in task['fields']['q0']['options']:
                        self.assertIn(option['id'] + '. ' + option['description'], rendered)
                        self.assertIn(option['id'] + ': ' + option['description'], flat['options'])


if __name__ == '__main__':
    unittest.main()
