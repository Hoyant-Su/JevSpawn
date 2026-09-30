from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class AdmittedPrompt:
    rendered: str
    tokens: tuple


def native_labels(tokenizer, settings):
    labels, token_ids, seen = [], [], set()
    widths = range(settings['minimum_characters'],
                   settings['maximum_characters'] + settings['character_increment'],
                   settings['character_increment'])
    for width in widths:
        candidates = [''.join(characters) for characters in product(settings['alphabet'], repeat=width)]
        encoded = tokenizer(candidates, add_special_tokens=settings['add_special_tokens'])['input_ids']
        for label, tokens in zip(candidates, encoded, strict=True):
            if len(tokens) != settings['native_token_count']:
                continue
            token, = tokens
            if token in seen or tokenizer.decode(
                    tokens, clean_up_tokenization_spaces=settings['clean_up_tokenization_spaces']) != label:
                continue
            labels.append(label)
            token_ids.append(token)
            seen.add(token)
            if len(labels) == settings['requested_count']:
                return labels, token_ids
    raise ValueError(f'Tokenizer supplies {len(labels)} distinct round-trip native labels; '
                     f'{settings["requested_count"]} are required.')


def validate_admitted_boundaries(tokenizer, admitted, labels, token_ids, counts, cache):
    assert all(isinstance(prompt, AdmittedPrompt) for prompt in admitted)
    _validate_token_tails(tokenizer, [prompt.rendered for prompt in admitted],
                          [prompt.tokens for prompt in admitted], labels, token_ids, counts, cache)


def _validate_token_tails(tokenizer, rendered, sequences, labels, token_ids, counts, cache):
    if any(count > len(labels) or count > len(token_ids) for count in counts):
        raise ValueError('Requested candidate count exceeds the verified native label pool.')
    anchors = {index: token.content for index, token in tokenizer.added_tokens_decoder.items()
               if token.special and not (token.lstrip or token.rstrip or token.normalized or token.single_word)}
    suffixes = {}
    for text, sequence, count in zip(rendered, sequences, counts, strict=True):
        anchor = next((token for token in reversed(sequence) if token in anchors), None)
        if anchor is None:
            raise ValueError('Native answer-label validation requires an atomic chat-template boundary.')
        suffix = text[text.rindex(anchors[anchor]):]
        suffixes[suffix] = max(count, suffixes.get(suffix, 0))
    pending = [(tail, tuple(labels[:count]), tuple(token_ids[:count]))
               for tail, count in suffixes.items()
               if (tail, tuple(labels[:count]), tuple(token_ids[:count])) not in cache]
    if not pending:
        return
    tails = [tail for tail, _, _ in pending]
    tail_ids = tokenizer(tails, add_special_tokens=False)['input_ids']
    # An unnormalized atomic special token separates preceding text from the answer boundary.
    joined = tokenizer([tail + label for tail, candidates, _ in pending for label in candidates],
                       add_special_tokens=False)['input_ids']
    expected = [sequence + [token] for (_, _, candidates), sequence in zip(pending, tail_ids, strict=True)
                for token in candidates]
    if joined != expected:
        raise ValueError('Native answer-label concatenation changes the complete prompt token sequence.')
    cache.update(pending)
