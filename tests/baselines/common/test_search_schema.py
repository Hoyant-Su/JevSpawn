from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import unittest

from baselines.common.environment import TaskEnvironment
from baselines.common.schema_environment import SchemaObservationEnvironment
from baselines.tool_agents.tools import ActionError
from methods.evidence_flow.environment import EvidenceEnvironment
from jev_spawn.infra.prompts import load_prompt


FIXTURE = json.loads(Path('tests/baselines/common/fixtures/search_schema_v1.json').read_text())


class SearchSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol = json.loads(Path(FIXTURE['protocol']).read_text())
        cls.settings = protocol['tools']
        cls.task = next(task for task in protocol['tasks'] if task['task_id'] == FIXTURE['task_id'])
        cls.schema = load_prompt(FIXTURE['schema'])
        failures = json.loads(Path(FIXTURE['failures']).read_text())
        history = next(row['messages'] for row in failures if row['task_id'] == FIXTURE['task_id'])
        cls.calls = [json.loads(message['content']) for message in history if message['role'] == 'assistant']
        cls.corpus = EvidenceEnvironment(**cls.settings['evidence'])

    def environment(self, candidate):
        options = {'read_id_schema': self.schema} if candidate else {}
        cls = SchemaObservationEnvironment if candidate else TaskEnvironment
        return cls(self.task, self.settings, FIXTURE['tool_directory'],
                   deadline=lambda: None, evidence=self.corpus, **options)

    def test_actual_results_and_cumulative_read_schema(self):
        candidate, original = self.environment(True), self.environment(False)
        self.assertIn(json.dumps(self.schema), candidate.reset())
        visible = deepcopy(candidate.input_schemas['read'])
        self.assertNotIn('enum', visible['properties']['ids']['items'])
        snapshots = []
        for call in self.calls:
            raw, _ = original.execute(call['tool'], call['arguments'])
            actual, done = candidate.execute(call['tool'], call['arguments'])
            self.assertFalse(done)
            self.assertEqual(actual, raw)
            observation = json.loads(actual)
            self.assertEqual(observation, json.loads(raw))
            self.assertEqual(candidate.input_schemas['read'], visible)
            self.assertEqual(candidate.read_validation_schema, original.input_schemas['read'])
            snapshots.append((observation, deepcopy(observation)))
        for observation, saved in snapshots:
            self.assertEqual(observation, saved)
        ids = sorted(candidate.observed_source_ids)
        self.assertEqual(candidate.observe('read', {'ids': ids}), original.observe('read', {'ids': ids}))
        self.assertEqual(candidate.actions, original.actions)
        unseen = next(identity for identity in self.corpus.ids if identity not in candidate.observed_source_ids)
        before = candidate.evidence.trace
        with self.assertRaises(ActionError):
            candidate.execute('read', {'ids': [unseen]})
        self.assertEqual(candidate.evidence.trace, before)

    def test_concurrent_search_preserves_results_and_disclosed_ids(self):
        environment = self.environment(True)
        with ThreadPoolExecutor(max_workers=len(self.calls)) as pool:
            outputs = list(pool.map(lambda call: environment.observe(call['tool'], call['arguments']), self.calls))
        discovered = {row['id'] for output in outputs for row in output}
        self.assertEqual(environment.observed_source_ids, discovered)
        self.assertEqual(environment.read_validation_schema['properties']['ids']['items']['enum'], sorted(discovered))
        self.assertNotIn('enum', environment.input_schemas['read']['properties']['ids']['items'])
        self.assertEqual(self.environment(True).observed_source_ids, set())

    def test_unobserved_read_is_still_rejected(self):
        environment = self.environment(True)
        with self.assertRaises(ActionError):
            environment.execute('read', {'ids': [FIXTURE['invalid_source_id']]})
        self.assertEqual(environment.evidence.trace['operations'], [])


if __name__ == '__main__':
    unittest.main()
