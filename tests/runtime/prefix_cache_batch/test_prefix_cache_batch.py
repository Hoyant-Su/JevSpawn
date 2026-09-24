from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

from baselines.common.parallel_service import ParallelPrefixCache
from jev_spawn.runtime.prefix_cache import PrefixCache


SETTINGS = json.loads(Path('tests/runtime/prefix_cache_batch/fixtures.json').read_text())


class RecordedCommands:
    """CPU transport fixture for the real leader/follower cache-plan method."""
    def __init__(self, leader, messages):
        self.is_leader, self.messages = leader, messages

    def leader_value(self, value):
        if self.is_leader:
            self.messages.append(deepcopy(value))
            return value
        return self.messages.popleft()


def cache_key(tokens):
    return (tuple(tokens),)


class PrefixCacheBatchTest(unittest.TestCase):

    def test_parent_reuse_requires_an_exact_complete_cached_prefix(self):
        cases = SETTINGS['parent_cases']
        cache = PrefixCache(SETTINGS['capacity'])
        calls = []
        states, _ = cache.get_many(cases['stored'], lambda sequences: self.compute(sequences, calls))
        short, long = cases['stored']
        short_state, long_state = states
        tokens, value = cache.longest_parent(cases['extension'], cases['minimum_length'])
        self.assertEqual(tokens, long)
        self.assertIs(value, long_state)
        tokens, value = cache.longest_parent(cases['divergence'], cases['minimum_length'])
        self.assertEqual(tokens, short)
        self.assertIs(value, short_state)
        self.assertIsNone(cache.longest_parent(cases['unrelated'], cases['minimum_length']))
        self.assertIsNone(cache.longest_parent(cases['extension'], cases['beyond_length']))

    def compute(self, requested, calls):
        calls.append(deepcopy(requested))
        return [torch.tensor(tokens, dtype=getattr(torch, SETTINGS['dtype']), device=SETTINGS['device'])
                for tokens in requested]

    def run_step(self, cache, step):
        prefixes = SETTINGS['prefixes']
        inputs = [prefixes[name] for name in step['requests']]
        old = {key: (value, value.clone()) for key, value in cache.entries.items()}
        calls = []
        values, hits = cache.get_many(inputs, lambda requested: self.compute(requested, calls))
        self.assertEqual(hits, step['hits'])
        self.assertEqual([value.tolist() for value in values], inputs)
        expected_calls = [[prefixes[name] for name in step['missing']]] if step['missing'] else []
        self.assertEqual(calls, expected_calls)
        self.assertEqual(list(cache.entries), [cache_key(prefixes[name]) for name in step['resident']])
        by_key = {}
        for tokens, value, hit in zip(inputs, values, hits, strict=True):
            key = cache_key(tokens)
            if key in by_key:
                self.assertIs(value, by_key[key])
            by_key[key] = value
            if hit:
                self.assertIs(value, old[key][0])
        for value, snapshot in old.values():
            self.assertTrue(torch.equal(value, snapshot))
        return values

    def test_serial_mixed_hits_duplicates_and_evicted_hit_outputs(self):
        cache = PrefixCache(SETTINGS['capacity'])
        history = []
        for step in SETTINGS['steps']:
            values = self.run_step(cache, step)
            history.extend((value, value.clone()) for value in values)
        for value, snapshot in history:
            self.assertTrue(torch.equal(value, snapshot))

    def test_parallel_resident_plan_and_mixed_cache_batches(self):
        messages = deque()
        leader = ParallelPrefixCache(PrefixCache(SETTINGS['capacity']), RecordedCommands(True, messages))
        follower = ParallelPrefixCache(PrefixCache(SETTINGS['capacity']), RecordedCommands(False, messages))
        for step in SETTINGS['steps']:
            leader_values = self.run_step(leader, step)
            follower_values = self.run_step(follower, step)
            self.assertFalse(messages)
            for left, right in zip(leader_values, follower_values, strict=True):
                self.assertTrue(torch.equal(left, right))
                self.assertNotEqual(left.data_ptr(), right.data_ptr())
        keys = [cache_key(SETTINGS['prefixes'][name]) for name in SETTINGS['plan_requests']]
        self.assertEqual(leader.resident_plan(keys), SETTINGS['plan_hits'])
        self.assertEqual(follower.resident_plan(keys), SETTINGS['plan_hits'])
        self.assertFalse(messages)

    def test_parallel_membership_divergence_is_rejected(self):
        messages = deque()
        leader = ParallelPrefixCache(PrefixCache(SETTINGS['capacity']), RecordedCommands(True, messages))
        follower = ParallelPrefixCache(PrefixCache(SETTINGS['capacity']), RecordedCommands(False, messages))
        leader_step, *_ = SETTINGS['steps']
        for cache, names in [(leader, leader_step['resident']), (follower, SETTINGS['divergent_follower_resident'])]:
            for name in names:
                tokens = SETTINGS['prefixes'][name]
                cache.entries[cache_key(tokens)] = torch.tensor(tokens, device=SETTINGS['device'])
        keys = [cache_key(SETTINGS['prefixes'][name]) for name in SETTINGS['plan_requests']]
        leader.resident_plan(keys)
        with self.assertRaises(AssertionError):
            follower.resident_plan(keys)


if __name__ == '__main__':
    program = unittest.main(exit=False)
    result = {'tests_run': program.result.testsRun, 'passed': program.result.wasSuccessful(),
              'scope': 'CPU tests of actual cache methods using configured tensor payloads and a recorded command channel; no model outputs, GPU execution or real distributed transport.',
              'fixtures': 'tests/runtime/prefix_cache_batch/fixtures.json'}
    Path(SETTINGS['output']).write_text(json.dumps(result, indent=2) + '\n')
    assert program.result.wasSuccessful()
