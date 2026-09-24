import json
from pathlib import Path

from baselines.common.errors import InputLimitError
from jev_spawn.infra.prompts import load_prompt


def truncate_prompt(tokenizer, rendered, tokens, limit, policy, prompt):
    original = len(tokens)
    metadata = dict(original_tokens=original, retained_tokens=original,
                    omitted_tokens=0, limit=limit, head_tokens=original, tail_tokens=0)
    if original <= limit:
        return rendered, tokens, metadata
    notice = prompt.format(original_tokens=original, limit=limit)
    notice_ids = tokenizer(notice, add_special_tokens=False)['input_ids']
    capacity = limit - len(notice_ids)
    if capacity <= 0:
        raise InputLimitError('The input window cannot hold its truncation notice.')
    head = int(capacity * policy['head_fraction'])
    tail = capacity - head
    retained = tokens[:head] + notice_ids + tokens[original - tail:]
    metadata.update(retained_tokens=len(retained), omitted_tokens=original - capacity,
                    head_tokens=head, tail_tokens=tail, notice_tokens=len(notice_ids))
    return tokenizer.decode(retained, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False), retained, metadata


class RejectInputWindow:
    def apply(self, rendered, tokens, limit):
        if len(tokens) > limit:
            raise InputLimitError(f'Input has {len(tokens)} tokens; limit is {limit}. Inputs are never truncated.')
        return rendered, tokens, dict(original_tokens=len(tokens), retained_tokens=len(tokens),
                                      omitted_tokens=0, limit=limit)


class InputWindow:
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.policy = json.loads(Path(path).read_text())
        assert self.policy['mode'] == 'head_tail_notice'
        assert 0 < self.policy['head_fraction'] < 1
        self.prompt = load_prompt(self.policy['prompt'])['notice']

    def apply(self, rendered, tokens, limit):
        return truncate_prompt(self.tokenizer, rendered, tokens, limit, self.policy, self.prompt)
