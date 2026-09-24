from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.rollout.branching import revise, solve
from jev_spawn.runtime.branches import Branch
from jev_spawn.runtime.query_execution import QueryExecution


ROOT = Path(__file__).resolve().parents[3]


class OnlineDeclarationTests(unittest.TestCase):
    def setUp(self):
        method = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']
        self.settings = {**method['rollout'], 'terminal_answer': method['terminal_answer']}
        self.prompts = load_prompt(self.settings['prompts'])
        self.service = Mock()
        self.service.backend.answer_labels = tuple(range(256))
        self.query = 'A test device accepts left or right. Observe its feedback before continuing.'
        self.schema = {'fields': [{'id': 'move', 'question': 'direction', 'values': ['left', 'right']}],
            'action': {'tool': 'execute', 'arguments': {'action': '${move}'}}, 'answer': None}
        self.environment = self.native()
        self.trace = {}

    def native(self, done=False, answer=None):
        return SimpleNamespace(done=done, answer=answer, tool_timings=[], observe=Mock(),
            display_tool_interface=Mock(return_value={'execute': {'input_schema': {'type': 'object'}}}),
            display_answer_schema=Mock(return_value={'type': 'object', 'properties': {
                'answer': {'type': 'string'}}, 'required': ['answer']}))

    def child(self, parent, identity, done=False, answer=None):
        child = parent.execution.fork()
        event = {'id': identity, 'parent_feedback': [],
            'action': {'tool': 'execute', 'arguments': {'action': 'left'}}, 'value': {'done': done}}
        child.latest_feedback = [event]
        child.tool_observations.append(event)
        child.trace = [{'tool_timings': []}, {'selected_values': {'move': 'left'}}]
        return Branch(child, self.native(done, answer), parent.declaration)

    def choices(self, pairs):
        self.service.decide.side_effect = [value for branches, operation in pairs for value in (
            [{'ranked_option_ids': branches, 'choice': branches[0]}],
            [{'choice': operation, 'ranked_option_ids': [operation]}])]

    def run_solver(self, rounds):
        return solve(self.query, task_id='task', service=self.service, complete=Mock(),
            settings=self.settings, prompts=self.prompts, budget={'max_turns': rounds,
                'max_new_tokens': 2048, 'temperature': 0.0}, trace=self.trace, environment=self.environment)

    def constructor(self):
        instance = Mock()
        instance.calls = []
        instance.program = deepcopy(self.schema)
        instance.build_action.return_value = deepcopy(self.schema)
        return instance

    def test_empty_slot_compiles_before_any_spawn_and_preserves_context(self):
        self.choices([(['root'], 'revise'), (['done'], 'submit')])
        builder = self.constructor()
        seen = []
        def spawn(parents, *args):
            parent = parents['root']
            seen.append(parent)
            return {'done': self.child(parent, 'done', True, {'answer': 'reached'})}
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder) as factory, \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn) as expand:
            result = self.run_solver(1)
        self.assertEqual(result['answer'], {'answer': 'reached'})
        self.assertEqual(factory.call_args.args[0], self.query)
        self.assertEqual([o['id'] for o in self.trace['rounds'][0]['operation_request']['options']], ['revise'])
        self.assertIn(self.query, seen[0].execution.query)
        self.assertEqual(seen[0].declaration['action'], self.schema['action'])
        expand.assert_called_once()
        self.assertIn('done', self.trace['rounds'][0]['children'])
        self.assertEqual(self.trace['rounds'][-1]['remaining_control_rounds'], 0)

    def test_compile_error_returns_to_model_next_turn_without_hidden_retry(self):
        self.choices([(['root'], 'revise'), (['root'], 'revise'), (['done'], 'submit')])
        builder = self.constructor()
        builder.build_action.side_effect = [ValueError('Undefined signature role: destination'), deepcopy(self.schema)]
        def spawn(parents, *args):
            return {'done': self.child(parents['root'], 'done', True, {'answer': 'reached'})}
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder) as factory, \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn) as expand:
            self.run_solver(2)
        self.assertEqual(factory.call_count, 2)
        self.assertFalse(self.trace['rounds'][0]['revision']['observation']['accepted'])
        self.assertIn('Undefined signature role', self.trace['rounds'][1]['operation_request']['state'])
        execution = factory.call_args_list[1].kwargs['feedback']['execution']
        feedback = execution['declaration_feedback_records'][execution['declaration_feedback_ids'][0]]
        self.assertIn('destination', feedback['error'])
        expand.assert_called_once()

    def test_revision_preserves_actual_state_and_is_visible_to_descendants(self):
        self.choices([(['root'], 'revise'), (['live'], 'revise'), (['done'], 'submit')])
        builder = self.constructor()
        changed = deepcopy(self.schema)
        changed['fields'][0]['values'] = ['left', 'right', 'wait']
        builder.build_action.side_effect = [deepcopy(self.schema), changed]
        parents_seen = []
        action_requests = []
        def spawn(parents, *args):
            parent = next(iter(parents.values()))
            parents_seen.append(parent)
            _, requests, _ = parent.execution.prepare(parent.declaration['fields'])
            action_requests.append(requests[0])
            identity = 'live' if len(parents_seen) == 1 else 'done'
            children = {identity: self.child(parent, identity, identity == 'done', {'answer': 'reached'})}
            if identity == 'live':
                children['sibling'] = self.child(parent, 'sibling')
            return children
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder) as factory, \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn):
            self.run_solver(2)
        self.assertEqual(parents_seen[1].execution.tool_observations[0]['id'], 'live')
        self.assertEqual(parents_seen[1].declaration['fields'][0]['values'], ['left', 'right', 'wait'])
        self.assertEqual(parents_seen[1].execution.query, self.query)
        self.assertIn('wait', parents_seen[1].execution.active_declaration['fields'][0]['values'])

        revision_state = factory.call_args.kwargs['feedback']['execution']
        self.assertEqual([event['id'] for event in revision_state['execution_events']], ['live'])
        current = self.trace['rounds'][1]
        frontier_events = [json.loads(line)['id']
                           for line in current['frontier_request']['history'].splitlines()]
        self.assertCountEqual(frontier_events, ['live', 'sibling'])
        operation_state = json.loads(current['operation_request']['state'])['state']
        self.assertEqual(operation_state['current_path'], ['live'])
        self.assertEqual(current['operation_request']['history'], current['frontier_request']['history'])
        feedback = parents_seen[1].execution.latest_feedback
        rendered = json.dumps(feedback, **self.settings['execution']['serialization'])
        self.assertIn(rendered, current['operation_request']['question'])
        self.assertEqual([event['id'] for event in feedback], ['live'])
        self.assertNotIn('sibling', current['operation_request']['question'])
        self.assertEqual(action_requests[1]['history'], current['frontier_request']['history'])
        self.assertEqual(action_requests[1]['context'], self.query)
        action_state = json.loads(action_requests[1]['state'])
        self.assertEqual(action_state['current_path'], ['live'])
        self.assertEqual(action_state['active_declaration'], parents_seen[1].declaration)

    def test_failed_revision_keeps_valid_slot_and_spawns_in_same_turn(self):
        self.choices([(['root'], 'revise'), (['live'], 'revise'), (['done'], 'submit')])
        builder = self.constructor()
        builder.build_action.side_effect = [deepcopy(self.schema), ValueError('Invalid revised slot')]
        declarations = []
        def spawn(parents, *args):
            parent = next(iter(parents.values()))
            declarations.append(parent.declaration)
            identity = 'live' if len(declarations) == 1 else 'done'
            return {identity: self.child(parent, identity, identity == 'done', {'answer': 'reached'})}
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder), \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn):
            result = self.run_solver(2)
        self.assertEqual(result['answer'], {'answer': 'reached'})
        self.assertIs(declarations[0], declarations[1])
        second = self.trace['rounds'][1]
        self.assertFalse(second['revision']['observation']['accepted'])
        self.assertIn('done', second['children'])

    def test_rejected_revision_forces_redeclaration_on_live_descendant(self):
        self.choices([(['root'], 'revise'), (['first'], 'revise'),
                      (['second'], 'revise'), (['done'], 'submit')])
        builder = self.constructor()
        builder.build_action.side_effect = [deepcopy(self.schema),
            ValueError('Invalid revised slot'), deepcopy(self.schema)]
        identities = iter(['first', 'second', 'done'])
        declarations = []

        def spawn(parents, *args):
            parent = next(iter(parents.values()))
            declarations.append(parent.declaration)
            identity = next(identities)
            return {identity: self.child(parent, identity, identity == 'done', {'answer': 'reached'})}

        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder), \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn):
            result = self.run_solver(3)

        self.assertEqual(result['answer'], {'answer': 'reached'})
        self.assertIs(declarations[0], declarations[1])
        correction = self.trace['rounds'][2]
        self.assertEqual([option['id'] for option in correction['operation_request']['options']], ['revise'])
        self.assertIn('Invalid revised slot', correction['operation_request']['state'])
        self.assertTrue(correction['revision']['observation']['accepted'])
        self.assertIn('done', correction['children'])

    def test_live_submit_uses_actual_selected_history_and_native_finish(self):
        self.choices([(['root'], 'revise'), (['live'], 'submit')])
        children = {}
        answer = {'answer': 'observed'}
        def spawn(parents, *args):
            child = self.child(parents['root'], 'live')
            def finish(tool, arguments):
                child.environment.done, child.environment.answer = True, arguments
                return {'done': True}
            child.environment.observe.side_effect = finish
            children['live'] = child
            children['sibling'] = self.child(parents['root'], 'sibling')
            return children
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=self.constructor()), \
             patch('jev_spawn.rollout.branching.spawn_scored', side_effect=spawn), \
             patch('jev_spawn.rollout.branching.finalize_answer', return_value=answer) as finalize:
            result = self.run_solver(2)
        self.assertEqual(result['answer'], answer)
        self.assertEqual(finalize.call_args.args[1]['current_path'], ['live'])
        self.assertEqual([event['id'] for event in finalize.call_args.args[1]['execution_events']], ['live'])
        frontier = self.trace['rounds'][1]['frontier_request']
        self.assertCountEqual([json.loads(line)['id'] for line in frontier['history'].splitlines()],
                              ['live', 'sibling'])
        children['live'].environment.observe.assert_called_once_with('finish', answer)

    def test_infrastructure_exception_is_not_reported_as_model_feedback(self):
        self.choices([(['root'], 'revise')])
        builder = self.constructor()
        builder.build_action.side_effect = RuntimeError('device unavailable')
        with patch('jev_spawn.rollout.branching.DeclarationBuilder', return_value=builder), \
             self.assertRaisesRegex(RuntimeError, 'device unavailable'):
            self.run_solver(2)

    def test_failed_signature_source_reaches_next_real_builder_request(self):
        invalid = 'send<{bit}>'
        corrected = 'send<{bit:range(0,2)}>'
        self.environment.display_tool_interface.return_value = {'execute': {'input_schema': {
            'type': 'object', 'properties': {'action': {'type': 'string'}}, 'required': ['action']}}}
        self.service.decide.return_value = [{'choice': '0'}]
        self.service.complete_batch.side_effect = [
            [{'text': text, 'token_ids': [42], 'finish_reason': 'stop'}] for text in [invalid, corrected]]
        branch = Branch(QueryExecution(self.query, self.service, 'task', self.settings['execution']),
                        self.environment, None)
        budget = {'max_new_tokens': 2048, 'temperature': 0.0}
        first, second = {}, {}
        revise(branch, self.query, 'task', self.service, self.settings, self.prompts, budget, first)
        self.assertFalse(first['observation']['accepted'])
        self.assertEqual(first['observation']['source_signatures'], [invalid])
        self.assertIsNone(branch.declaration)
        revise(branch, self.query, 'task', self.service, self.settings, self.prompts, budget, second)
        messages, = self.service.complete_batch.call_args.args[:1]
        self.assertIn(invalid, messages[0][1]['content'])
        execution = json.loads(self.service.decide.call_args.args[0][0]['state'])['feedback']['execution']
        feedback = execution['declaration_feedback_records'][execution['declaration_feedback_ids'][0]]
        self.assertIn(first['observation']['error'], feedback['error'])
        self.assertTrue(second['observation']['accepted'])
        self.assertEqual(second['observation']['source_signatures'], [corrected])
        self.assertEqual(branch.declaration['action']['arguments']['action'], 'send<${f0}>')


if __name__ == '__main__':
    unittest.main()
