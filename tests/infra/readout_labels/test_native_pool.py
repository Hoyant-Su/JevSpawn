import json
import unittest

from transformers import AutoTokenizer
import yaml

from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.readout_labels import native_labels, validate_boundaries
from jev_spawn.schema import CONTROLLER, controller_prompts
from project_paths import ROOT


FIXTURE = json.loads((ROOT / 'tests/infra/readout_labels/fixtures.json').read_text())
SETTINGS = json.loads((ROOT / FIXTURE['label_config']).read_text())
SHARED = yaml.safe_load((ROOT / FIXTURE['shared_config']).read_text())


class NativePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            SHARED['model']['path'], local_files_only=CORE['backend']['local_files_only'])
        cls.labels, cls.ids = native_labels(cls.tokenizer, SETTINGS)

    def test_complete_pool_has_distinct_exact_round_trip_native_tokens(self):
        self.assertEqual(len(self.labels), SETTINGS['requested_count'])
        self.assertEqual(len(set(self.ids)), len(self.labels))
        for label, token in zip(self.labels, self.ids, strict=True):
            self.assertEqual(self.tokenizer.encode(label, add_special_tokens=SETTINGS['add_special_tokens']), [token])
            self.assertEqual(self.tokenizer.decode([token], clean_up_tokenization_spaces=SETTINGS['clean_up_tokenization_spaces']), label)

    def test_actual_controller_chat_template_preserves_every_label_boundary(self):
        options = [{'id': FIXTURE['option_id'].format(index=index),
                    'description': FIXTURE['description'].format(index=index)}
                   for index in range(SETTINGS['requested_count'])]
        prompts = controller_prompts(FIXTURE['states'], FIXTURE['question'], options,
                                     self.labels, CONTROLLER['output_instruction'])
        rendered = self.tokenizer.apply_chat_template(
            [[{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}]
             for prompt in prompts], tokenize=False, add_generation_prompt=CORE['backend']['add_generation_prompt'],
            enable_thinking=CORE['backend']['enable_thinking'])
        sequences = self.tokenizer(rendered, add_special_tokens=SETTINGS['add_special_tokens'])['input_ids']
        counts = [len(self.labels)] * len(rendered)
        validate_boundaries(self.tokenizer, rendered, sequences, self.labels, self.ids, counts)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            validate_boundaries(self.tokenizer, rendered, sequences, self.labels, self.ids,
                                [len(self.labels) + FIXTURE['count_increment']] * len(rendered))
        changed = [[FIXTURE['changed_prefix_token'], *sequence] for sequence in sequences]
        with self.assertRaisesRegex(ValueError, 'concatenation'):
            validate_boundaries(self.tokenizer, rendered, changed, self.labels, self.ids, counts)

    def test_insufficient_tokenizer_capacity_fails_without_clipping(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            native_labels(self.tokenizer, {**SETTINGS, **FIXTURE['impossible_pool']})


if __name__ == '__main__':
    unittest.main()
