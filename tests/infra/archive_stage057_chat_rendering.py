from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch

from transformers import AutoTokenizer
import yaml

from jev_spawn.infra.prompts import ROOT


def render_chats(tokenizer, message_batches):
    # Archived rejected stage057 implementation; production no longer uses this renderer.
    continuations = [messages[-1]['role'] == 'assistant' for messages in message_batches]
    rendered = tokenizer.apply_chat_template(
        [messages[:-1] if continuation else messages
         for messages, continuation in zip(message_batches, continuations, strict=True)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return [text + messages[-1]['content'] if continuation else text
            for text, messages, continuation in zip(rendered, message_batches, continuations, strict=True)]


class ChatRenderingTests(TestCase):
    @classmethod
    def setUpClass(cls):
        shared = yaml.safe_load((ROOT / 'configs/shared_config_tp4_v2.yaml').read_text())
        cls.tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)

    def original_render(self, batches):
        return self.tokenizer.apply_chat_template(
            batches, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def token_ids(self, texts):
        return self.tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)['input_ids']

    def test_ordinary_rows_preserve_exact_bytes_and_token_ids(self):
        batches = [
            [{'role': 'system', 'content': 'Task rules.'}, {'role': 'user', 'content': 'First row.\n '}],
            [{'role': 'user', 'content': 'Earlier question'}, {'role': 'assistant', 'content': 'Earlier answer'},
             {'role': 'user', 'content': 'Next question: α → β\n'}],
        ]
        expected = self.original_render(batches)
        actual = render_chats(self.tokenizer, batches)
        self.assertEqual([s.encode() for s in actual], [s.encode() for s in expected])
        self.assertEqual(self.token_ids(actual), self.token_ids(expected))

    def test_assistant_prefix_preserves_trailing_whitespace(self):
        preceding = [{'role': 'system', 'content': 'Return the requested value.'},
                     {'role': 'user', 'content': 'Write the declaration.'}]
        base = self.original_render([preceding])[0]
        for content in ('```text\n', 'prefix \n\n', 'prefix\t '):
            with self.subTest(content=content):
                actual = render_chats(self.tokenizer, [preceding + [{'role': 'assistant', 'content': content}]])
                self.assertEqual(actual, [base + content])
                self.assertEqual(self.token_ids(actual), self.token_ids([base + content]))
                self.assertTrue(actual[0].endswith(content))

    def test_mixed_rows_keep_order_and_one_batched_template_call(self):
        batches = [
            [{'role': 'user', 'content': 'row-a'}, {'role': 'assistant', 'content': 'prefix-a\n'}],
            [{'role': 'user', 'content': 'row-b'}],
            [{'role': 'user', 'content': 'row-c'}, {'role': 'assistant', 'content': 'prefix-c\n '}],
        ]
        before = deepcopy(batches)
        expected = self.original_render([[row[0]] for row in batches])
        expected[0] += 'prefix-a\n'
        expected[2] += 'prefix-c\n '
        with patch.object(self.tokenizer, 'apply_chat_template', wraps=self.tokenizer.apply_chat_template) as render:
            actual = render_chats(self.tokenizer, batches)
        render.assert_called_once()
        self.assertEqual(len(render.call_args.args[0]), len(batches))
        self.assertEqual(actual, expected)
        self.assertEqual(self.token_ids(actual), self.token_ids(expected))
        self.assertEqual(batches, before)


if __name__ == '__main__':
    main()
