import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

import jsonschema

from baselines.common.environment import TaskEnvironment
from baselines.tool_agents.tools import ActionError
from methods.evidence_flow.environment import EvidenceEnvironment


FIXTURE = json.loads(Path('tests/baselines/common/fixtures/read_contract.json').read_text())


class ReadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads(Path(FIXTURE['settings']).read_text())
        cls.tasks = json.loads(Path(FIXTURE['protocol']).read_text())['tasks']
        cls.corpus = EvidenceEnvironment(**cls.settings['evidence'])
        batches = json.loads(Path(FIXTURE['batches']).read_text())
        batch = next(row for row in batches if FIXTURE['evidence_task'] in row['task_ids']
                     and 'texts' in row)
        cls.query = batch['texts'][batch['task_ids'].index(FIXTURE['evidence_task'])]

    def environment(self, task_id):
        task = next(row for row in self.tasks if row['task_id'] == task_id)
        return TaskEnvironment(task, self.settings, FIXTURE['tool_directory'],
                               deadline=lambda: None, evidence=self.corpus)

    def search(self, environment, query):
        result, done = environment.execute('search', {
            'query': query, 'k': self.settings['evidence']['search_limit']})
        self.assertFalse(done)
        return json.loads(result)

    def test_real_search_domain_and_exact_source_materialization(self):
        environment = self.environment(FIXTURE['evidence_task'])
        schema = environment.input_schemas['read']
        jsonschema.Draft202012Validator.check_schema(schema)
        self.assertEqual(schema['properties']['ids']['items']['enum'], [])
        results = self.search(environment, self.query)
        ids = [row['id'] for row in results]
        self.assertEqual(schema['properties']['ids']['items']['enum'], sorted(ids))
        output, done = environment.execute('read', {'ids': ids})
        expected = [{**self.corpus.units[identity], **self.corpus.metadata(self.corpus.units[identity])}
                    for identity in ids]
        self.assertFalse(done)
        self.assertEqual(json.loads(output), expected)

    def test_nonlexical_queries_return_tool_errors_without_retrieval(self):
        environment = self.environment(FIXTURE['evidence_task'])
        for query in FIXTURE['invalid_queries']:
            arguments = {'query': query, 'k': self.settings['evidence']['search_limit']}
            with self.subTest(query=query), self.assertRaises(ActionError):
                environment.execute('search', arguments)
            result = environment.observe('search', arguments)
            self.assertEqual(result['error']['type'], ActionError.__name__)
            self.assertEqual(environment.evidence.trace['operations'], [])
            self.assertEqual(environment.input_schemas['read']['properties']['ids']['items']['enum'], [])
            self.assertIsNone(environment.answer)

    def test_invalid_and_unobserved_ids_are_action_errors_before_indexing(self):
        environment = self.environment(FIXTURE['evidence_task'])
        with self.assertRaises(ActionError):
            environment.execute('read', {'ids': [FIXTURE['invalid_source_id']]})
        self.assertEqual(environment.evidence.trace['operations'], [])
        self.search(environment, self.query)
        unseen = next(identity for identity in self.corpus.ids
                      if identity not in environment.observed_source_ids)
        before = environment.evidence.trace
        for identity in [FIXTURE['invalid_source_id'], unseen]:
            with self.subTest(identity=identity), self.assertRaises(ActionError):
                environment.execute('read', {'ids': [identity]})
        observed = environment.observe('read', {'ids': [FIXTURE['invalid_source_id']]})
        self.assertEqual(observed['error']['type'], ActionError.__name__)
        self.assertEqual(environment.evidence.trace, before)
        self.assertIsNone(environment.answer)

    def test_concurrent_search_domains_union_without_episode_leakage(self):
        environment = self.environment(FIXTURE['evidence_task'])
        queries = [self.query, FIXTURE['second_query']]
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            results = list(pool.map(lambda query: self.search(environment, query), queries))
        expected = sorted({row['id'] for result in results for row in result})
        self.assertEqual(environment.input_schemas['read']['properties']['ids']['items']['enum'], expected)
        other = self.environment(FIXTURE['evidence_task'])
        self.assertEqual(other.input_schemas['read']['properties']['ids']['items']['enum'], [])

    def test_real_ranking_catalog_enforced_before_document_lookup(self):
        environment = self.environment(FIXTURE['ranking_task'])
        catalog = environment.task['source']['candidates']
        selected = catalog[:self.settings['documents_per_read']]
        result, done = environment.execute('read_documents', {
            'ids': [row['document_id'] for row in selected]})
        self.assertFalse(done)
        self.assertEqual(json.loads(result), selected)
        with self.assertRaises(ActionError):
            environment.execute('read_documents', {'ids': [FIXTURE['invalid_source_id']]})
        with self.assertRaises(ActionError):
            environment.execute('read_documents', {'ids': FIXTURE['invalid_source_id']})


if __name__ == '__main__':
    unittest.main()
