from dataclasses import dataclass

from jev_spawn.algo.structured import common_prefix


@dataclass(frozen=True)
class SemanticDomain:
    values: tuple
    prefix: tuple
    branches: tuple
    tails: tuple


def compile_domain(tokenizer, values):
    paths = tokenizer(list(values), add_special_tokens=False)['input_ids']
    boundary = common_prefix(paths)
    assert all(len(path) > boundary for path in paths)
    branches = tuple(path[boundary] for path in paths)
    assert len(set(branches)) == len(values), 'This readout requires one distinct branch per value.'
    assert tokenizer.batch_decode(paths, clean_up_tokenization_spaces=False) == list(values)
    return SemanticDomain(tuple(values), tuple(paths[0][:boundary]), branches,
                          tuple(tuple(path[boundary:]) for path in paths))
