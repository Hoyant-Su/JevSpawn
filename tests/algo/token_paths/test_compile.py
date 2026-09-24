import json
from pathlib import Path
import unittest

from jev_spawn.algo.token_paths import compile_token_paths


FIXTURES = json.loads(Path(__file__).with_name('fixtures.json').read_text())


class TokenPathsTest(unittest.TestCase):
    def test_language_and_forced_runs(self):
        for name, example in FIXTURES.items():
            if name == 'invalid':
                continue
            with self.subTest(name=name):
                compiled = compile_token_paths(example['paths'])
                tokens = compiled['token_ids']
                accepted = set()
                pending = [(compiled['root'], ())]
                while pending:
                    state, prefix = pending.pop()
                    if compiled['terminal'][state]:
                        accepted.add(prefix)
                    outgoing = [(tokens[token], target) for token, target, valid in zip(
                        compiled['edge_token_indices'][state], compiled['edge_next_states'][state],
                        compiled['edge_valid'][state], strict=True) if valid]
                    pending.extend((target, (*prefix, token)) for token, target in outgoing)
                    forced = compiled['forced_token_indices'][state][:compiled['forced_lengths'][state]]
                    target = state
                    for token in forced:
                        transitions = [(value, next_state) for value, next_state, valid in zip(
                            compiled['edge_token_indices'][target], compiled['edge_next_states'][target],
                            compiled['edge_valid'][target], strict=True) if valid]
                        (actual_token, target), = transitions
                        self.assertEqual(actual_token, token)
                    self.assertEqual(target, compiled['forced_next_states'][state])
                self.assertEqual(accepted, set(map(tuple, example['paths'])))
                self.assertEqual(tokens, sorted(set(token for path in example['paths'] for token in path)))

    def test_rejects_ambiguous_or_empty_paths(self):
        for paths in FIXTURES['invalid']:
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                compile_token_paths(paths)


if __name__ == '__main__':
    unittest.main()
