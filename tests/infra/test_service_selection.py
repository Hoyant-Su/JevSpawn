import json
from pathlib import Path
import unittest

from baselines.common.config import SharedConfig
from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_service import parallel_factory
from baselines.common.service_factory import service_factory
from baselines.latentmas.common_service import LatentMASService, role_messages
from baselines.latentmas.context_window_service import WindowedLatentMASService


class ServiceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.inference = json.loads(Path('configs/inference/shared_service.json').read_text())
        self.shared = SharedConfig.load('configs/shared_config_tp4_v2.yaml')

    def method(self, name):
        return json.loads(Path(f'configs/baselines/common/methods/{name}.json').read_text())

    def test_general_methods_share_one_numerical_service(self):
        for name in ('single', 'react', 'lats', 'llmcompiler', 'hiagent',
                     'foldagent', 'agentprune', 'dyflow', 'jevspawn'):
            with self.subTest(method=name):
                factory = service_factory(self.method(name), self.inference)
                self.assertIs(factory.func, StableGraphFiniteService)
                self.assertIs(factory.keywords['settings'], self.inference['settings'])
                self.assertTrue(issubclass(parallel_factory(factory).func, factory.func))

    def test_latentmas_dispatches_actual_latent_collaboration_on_all_ranks(self):
        method = self.method('latentmas')
        factory = service_factory(method, self.inference)
        self.assertIs(factory.func, WindowedLatentMASService)
        self.assertIs(factory.keywords['settings'], method['settings'])
        self.assertIs(factory.keywords['input_policy'], self.inference['settings']['input_window'])
        parallel = parallel_factory(factory)
        self.assertIs(parallel.func._generate_batch, LatentMASService._generate_batch)
        self.assertIs(parallel.func._validate_inputs, WindowedLatentMASService._validate_inputs)
        self.assertEqual(parallel.keywords, factory.keywords)
        self.assertTrue(Path(method['settings']['alignment_config']).is_file())

    def test_latent_roles_preserve_task_and_method_context(self):
        factory = service_factory(self.method('latentmas'), self.inference)
        messages = [{'role': 'system', 'content': factory.keywords['prompts']['system']},
                    {'role': 'user', 'content': 'Official task context and latest tool observation.'}]
        roles = role_messages(messages, factory.keywords['prompts'], self.shared.model.path)
        self.assertEqual(len(roles), len(self.method('latentmas')['settings']['role_window']['weights']))
        for role in roles:
            text = '\n'.join(message['content'] for message in role)
            self.assertIn(messages[1]['content'], text)
            self.assertIn(messages[0]['content'], text)

    def test_missing_service_selection_cannot_silently_use_plain_generation(self):
        method = self.method('latentmas')
        method.pop('inference_service')
        with self.assertRaises(KeyError):
            service_factory(method, self.inference)


if __name__ == '__main__':
    unittest.main()
