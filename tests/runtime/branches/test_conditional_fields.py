from copy import deepcopy
from itertools import count
import json
from pathlib import Path
import unittest

import torch

from jev_spawn.algo.composition import rank_extensions
from jev_spawn.runtime.branches import Branch, merge_extensions, required_fields, spawn_scored
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']['rollout']['execution']


class ConditionalScores:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.pending, self.requests = {}, []

    def initial_scores(self):
        return torch.zeros(1, dtype=torch.float64)

    def decide(self, requests, *, task_id):
        self.requests.extend(deepcopy(requests))
        outputs = []
        for request in requests:
            _, field = json.loads(request['id'])
            distribution = torch.tensor(self.probabilities[field], dtype=torch.float64)
            self.pending[task_id, request['id']] = distribution
            choice = int(distribution.argmax())
            outputs.append({'id': request['id'], 'choice': request['options'][choice]['id']})
        return outputs

    def extend(self, identities, scores, *, task_id, width):
        distributions = [self.pending.pop((task_id, identity)) for identity in identities]
        return rank_extensions(distributions, scores, width)


class RecordingEnvironment:
    def __init__(self):
        self.actions, self.tool_timings = [], []

    def fork(self):
        return deepcopy(self)

    def observe(self, tool, arguments):
        self.actions.append((tool, deepcopy(arguments)))
        self.tool_timings.append({'tool': tool})
        return {'received': deepcopy(arguments)}


def schema():
    return compile_declaration({'fields': [
        {'id': 'object', 'question': 'object', 'values': ['a', 'b']},
        {'id': 'target', 'question': 'target', 'values': ['x', 'y']},
        {'id': 'operator', 'question': 'operator', 'values': ['(inspect ${object})', '(combine ${object} ${target})']}],
        'action': {'tool': 'execute', 'arguments': {'action': '${operator}'}}, 'answer': None}, SETTINGS)


class ConditionalFieldTests(unittest.TestCase):
    def test_node_identity_is_independent_of_selected_action_ranking(self):
        fixture = json.loads((ROOT / 'configs/tests/runtime/branch_identity.json').read_text())
        mappings = {}
        for width in (fixture['full_width'], fixture['limited_width']):
            mappings[width] = []
            for probabilities in fixture['cases']:
                service = ConditionalScores(probabilities)
                execution = QueryExecution(fixture['query'], service, fixture['task_id'], SETTINGS)
                nodes = (fixture['node_format'].format(index=index)
                         for index in count(fixture['initial_node_index']))
                children = spawn_scored(
                    {fixture['root_id']: Branch(execution, RecordingEnvironment(), schema())},
                    width, fixture['host_workers'], nodes)
                mappings[width].append({identity: child.environment.actions[-1]
                                        for identity, child in children.items()})
        full_reference, full_permuted = mappings[fixture['full_width']]
        limited_reference, limited_permuted = mappings[fixture['limited_width']]
        self.assertEqual(full_reference, full_permuted)
        self.assertNotEqual(limited_reference, limited_permuted)

    def test_inactive_argument_has_no_request_no_factor_and_no_duplicate_child(self):
        service = ConditionalScores({'operator': [.6, .4], 'object': [.5, .5], 'target': [.75, .25]})
        execution = QueryExecution('Unit test of conditional beam algebra.', service, 'task', SETTINGS)
        environment = RecordingEnvironment()
        children = spawn_scored({'root': Branch(execution, environment, schema())}, 6, 2,
                                (f'node-{index}' for index in count()))
        actual = {}
        for child in children.values():
            command = child.environment.actions[-1][1]['action']
            actual[command] = child.execution.trace[-1]['conditional_log_probability']
            if command.startswith('(inspect'):
                self.assertNotIn('target', child.execution.trace[-1]['selected_values'])
        expected = {'(inspect a)': .3, '(inspect b)': .3, '(combine a x)': .15,
                    '(combine b x)': .15, '(combine a y)': .05, '(combine b y)': .05}
        self.assertEqual(len(children), len(actual))
        self.assertEqual(actual.keys(), expected.keys())
        for command, probability in expected.items():
            self.assertAlmostEqual(actual[command], torch.tensor(probability, dtype=torch.float64).log().item())
        target_requests = [request for request in service.requests if json.loads(request['id'])[1] == 'target']
        self.assertEqual(len(target_requests), 2)
        self.assertTrue(all(json.loads(request['state'])['bound_fields']['operator'].startswith('(combine')
                            for request in target_requests))
        self.assertEqual(environment.actions, [])
        self.assertEqual(service.pending, {})

    def test_parents_keep_distinct_declarations_with_shared_field_identifiers(self):
        first = {'fields': [{'id': 'f0', 'question': 'direction', 'values': ['north', 'south']}],
                 'action': {'tool': 'move', 'arguments': {'direction': '${f0}'}}, 'answer': None}
        second = {'fields': [{'id': 'f0', 'question': 'number', 'values': [3, 7]},
                             {'id': 'f1', 'question': 'volume', 'values': ['quiet', 'loud']}],
                  'action': {'tool': 'signal', 'arguments': {'number': '${f0}', 'volume': '${f1}'}}, 'answer': None}
        service = ConditionalScores({'f0': [.6, .4], 'f1': [.7, .3]})
        parents = {owner: Branch(QueryExecution('Shared task.', service, 'task', SETTINGS),
                                RecordingEnvironment(), declaration)
                   for owner, declaration in [('first', first), ('second', second)]}
        children = spawn_scored(parents, 2, 2, (f'n{index}' for index in count()))
        self.assertEqual(len(children), 4)
        for child in children.values():
            tool, arguments = child.environment.actions[-1]
            if tool == 'move':
                self.assertIs(child.declaration, first)
                self.assertIn(arguments['direction'], ['north', 'south'])
                self.assertNotIn('f1', child.execution.trace[-1]['selected_values'])
            else:
                self.assertIs(child.declaration, second)
                self.assertIn(arguments['number'], [3, 7])
                self.assertIn(arguments['volume'], ['quiet', 'loud'])
        self.assertEqual(len([request for request in service.requests
                              if json.loads(request['id'])[1] == 'f0']), 2)

    def test_nested_selector_reachability_uses_bound_variant(self):
        execution = QueryExecution('Nested selectors.', None, 'task', SETTINGS)
        fields = {'outer': {'values': ['fixed', '${inner}']},
                  'inner': {'values': ['${left}', '${right}']},
                  'left': {'values': ['a', 'b']}, 'right': {'values': ['x', 'y']}}
        execution.bound_fields['outer'] = 'fixed'
        self.assertEqual(required_fields(execution, '${outer}', fields), set())
        execution.bound_fields['outer'] = '${inner}'
        execution.bound_fields['inner'] = '${right}'
        self.assertEqual(required_fields(execution, '${outer}', fields), {'right'})

    def test_carried_scores_compete_in_stable_parent_order(self):
        candidates, scores = merge_extensions([None, None, None], [1, 2], [0, 1], [0, 0],
            torch.tensor([.5, .5], dtype=torch.float64), torch.tensor([.5, .8, .8], dtype=torch.float64), 2)
        self.assertEqual(candidates, [(0, None), (1, 0)])
        torch.testing.assert_close(scores, torch.tensor([.5, .5], dtype=torch.float64))

    def test_singleton_binding_adds_no_model_factor(self):
        declaration = {'fields': [
            {'id': 'object', 'question': 'object', 'values': ['only']},
            {'id': 'operator', 'question': 'operator', 'values': ['use ${object}', 'wait']}],
            'action': {'tool': 'execute', 'arguments': {'action': '${operator}'}}, 'answer': None}
        service = ConditionalScores({'operator': [.6, .4]})
        execution = QueryExecution('Singleton domain.', service, 'task', SETTINGS)
        children = spawn_scored({'root': Branch(execution, RecordingEnvironment(), compile_declaration(declaration, SETTINGS))},
                                2, 1, iter(['n0', 'n1']))
        self.assertEqual([child.environment.actions[-1][1]['action'] for child in children.values()], ['use only', 'wait'])
        self.assertEqual(len(service.requests), 1)
        torch.testing.assert_close(torch.tensor([child.execution.trace[-1]['conditional_log_probability']
                                                for child in children.values()], dtype=torch.float64).exp(),
                                   torch.tensor([.6, .4], dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
