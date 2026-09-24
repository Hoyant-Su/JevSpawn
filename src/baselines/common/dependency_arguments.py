import re
import tokenize


def dependency_bindings(source, fields, *, literal_bindings, argument_segments, grammar, reference_pattern):
    lines = source.splitlines(keepends=True)
    normalized = []
    for segment in argument_segments(source):
        first, *remaining = segment
        named = (first.type == tokenize.NAME and remaining and remaining[0].type == tokenize.OP
                 and remaining[0].string in grammar['named_separators'])
        prefix, value = (segment[:2], segment[2:]) if named else ([], segment)
        if not value:
            raise ValueError(source)
        start, end = value[0].start, value[-1].end
        begin = sum(map(len, lines[:start[0] - 1])) + start[1]
        finish = sum(map(len, lines[:end[0] - 1])) + end[1]
        raw = source[begin:finish]
        tokens = [(token.type, token.string) for token in prefix]
        tokens += ([(tokenize.STRING, repr(raw))] if re.fullmatch(reference_pattern, raw) else
                   [(token.type, token.string) for token in value])
        normalized.append(tokenize.untokenize(tokens))
    return literal_bindings(grammar['comma'].join(normalized), fields)
