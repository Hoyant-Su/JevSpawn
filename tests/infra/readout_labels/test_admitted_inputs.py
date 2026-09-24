from dataclasses import FrozenInstanceError
import unittest

import test_native_pool

from jev_spawn.infra.readout_labels import AdmittedPrompt, validate_admitted_boundaries
from jev_spawn.schema import CONTROLLER, controller_prompts


class ObservedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, values, **kwargs):
        self.calls.append(values)
        return self.tokenizer(values, **kwargs)


class AdmittedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_native_pool.NativePoolTests.setUpClass()
        original = test_native_pool.NativePoolTests
        cls.tokenizer, cls.labels, cls.ids = original.tokenizer, original.labels, original.ids
        fixture = test_native_pool.FIXTURE
        options = [{'id': fixture['option_id'].format(index=index),
                    'description': fixture['description'].format(index=index)}
                   for index in range(len(cls.labels))]
        prompts = controller_prompts(fixture['states'], fixture['question'], options,
                                    cls.labels, CONTROLLER['output_instruction'])
        rendered = cls.tokenizer.apply_chat_template(
            [[{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}]
             for prompt in prompts], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = cls.tokenizer(rendered, add_special_tokens=False)['input_ids']
        cls.admitted = [AdmittedPrompt(text, tuple(tokens)) for text, tokens in zip(rendered, encoded, strict=True)]
        cls.counts = [len(cls.labels)] * len(cls.admitted)

    def test_admitted_payload_is_immutable(self):
        for prompt in self.admitted:
            with self.assertRaises(FrozenInstanceError):
                prompt.rendered = prompt.rendered
            self.assertIsInstance(prompt.tokens, tuple)

    def test_exact_tail_validation_does_not_retokenize_complete_inputs(self):
        tokenizer, cache = ObservedTokenizer(self.tokenizer), set()
        validate_admitted_boundaries(tokenizer, self.admitted, self.labels, self.ids, self.counts, cache)
        complete = {prompt.rendered for prompt in self.admitted}
        self.assertTrue(tokenizer.calls)
        self.assertFalse(any(text in complete for call in tokenizer.calls for text in call))
        tokenizer.calls.clear()
        validate_admitted_boundaries(tokenizer, self.admitted, self.labels, self.ids, self.counts, cache)
        self.assertFalse(tokenizer.calls)

    def test_cached_tail_does_not_authorize_changed_candidate_tokens(self):
        cache = set()
        validate_admitted_boundaries(self.tokenizer, self.admitted, self.labels, self.ids, self.counts, cache)
        changed = [test_native_pool.FIXTURE['changed_prefix_token']] * len(self.ids)
        with self.assertRaisesRegex(ValueError, 'concatenation'):
            validate_admitted_boundaries(self.tokenizer, self.admitted, self.labels, changed, self.counts, cache)


if __name__ == '__main__':
    unittest.main()
