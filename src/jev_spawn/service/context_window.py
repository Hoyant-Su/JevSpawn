from jev_spawn.service.errors import InputLimitError


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
