import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.foldagent.adapter import TokenizerInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--batches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    interface = TokenizerInterface(tokenizer)
    batches = json.loads(args.batches.read_text())
    histories = {identity: messages for batch in batches
                 for identity, messages in zip(batch['task_ids'], batch['messages'])}
    checks = []
    for identity, history in histories.items():
        for end in range(2, len(history) + 1):
            messages = history[:end]
            expected = tokenizer(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                add_special_tokens=False)['input_ids']
            actual = interface.apply_chat_template(messages, add_generation_prompt=True)
            hiagent = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            assert isinstance(actual, list) and actual == expected == hiagent
            checks.append({'task_id': identity, 'turns': end, 'tokens': len(actual)})
    args.output.write_text(json.dumps({'token_sequences_equal': True, 'checks': checks}, indent=2) + '\n')
    print(json.dumps({'histories': len(histories), 'checks': len(checks),
                      'max_tokens': max(row['tokens'] for row in checks)}))


if __name__ == '__main__':
    main()
