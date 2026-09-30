import numpy as np
import torch
from transformers import StopStringCriteria


def matching_positions(tokens, indices, stops):
    positions, overlaps = {}, {}
    for stop in stops:
        size = len(stop)
        prefixes = tuple(stop[:size - index] for index in range(1, size))
        suffixes = tuple(stop[-length:] for length in range(1, size))
        short = {stop[:0]: list(range(1, size))}
        for index, prefix in enumerate(prefixes, 1):
            for length in range(1, len(prefix)):
                short.setdefault(prefix[-length:], []).append(index)
        pos, ends = {}, {}
        for token, identity in zip(tokens, indices):
            valid = short.get(token, []).copy()
            if token.endswith(prefixes):
                valid.extend(index for index, prefix in enumerate(prefixes, 1) if token.endswith(prefix))
            if valid:
                pos[identity] = sorted(valid)
            end = []
            if token.startswith(suffixes):
                end = [length for length, suffix in enumerate(suffixes, 1) if token.startswith(suffix)]
            occurrence = token.find(stop)
            while occurrence >= 0:
                end.append(size)
                occurrence = token.find(stop, occurrence + 1)
            if end:
                ends[identity] = end
        positions[stop], overlaps[stop] = pos, ends
    return positions, overlaps


class IndexedStopCriteria(StopStringCriteria):
    """Build the original GPU stop-matching table using indexed string boundaries."""

    def __init__(self, stops, mode, tokens, indices):
        self.stop_strings = tuple(stops)
        self._stop_string_matching_mode = mode
        self._stop_strings_for_matching = self._get_stop_strings_for_matching(stops, mode)
        self.maximum_token_len = max(map(len, self._stop_strings_for_matching))
        self.num_stop_strings = len(stops)
        self.target_lens = torch.tensor(list(map(len, self._stop_strings_for_matching)), dtype=torch.int32)
        positions, overlaps = matching_positions(tokens, indices, self._stop_strings_for_matching)
        lengths = [len(row) for table in positions.values() for row in table.values()]
        self.max_valid_positions = max(lengths, default=1)
        self.max_valid_end_lens = max(len(row) for table in overlaps.values() for row in table.values())
        width = self.num_stop_strings * (self.max_valid_positions + self.max_valid_end_lens) + 1
        table = np.full((max(indices) + 2, width), -1, dtype=np.int32)
        for index, stop in enumerate(self._stop_strings_for_matching):
            start = self.max_valid_positions * index
            for token, values in positions[stop].items():
                table[token, start:start + len(values)] = values
            start = self.max_valid_positions * self.num_stop_strings + self.max_valid_end_lens * index
            for token, values in overlaps[stop].items():
                table[token, start:start + len(values)] = values
        table[np.asarray(indices), -1] = np.fromiter(map(len, tokens), dtype=np.int32)
        self.embedding_vec = torch.from_numpy(table)


class StopCache:
    def __init__(self, tokenizer, device, engine):
        self.tokenizer, self.device, self.engine = tokenizer, device, engine
        self.mode = StopStringCriteria._get_stop_string_matching_mode(tokenizer)
        self.lexicon = None
        self.criteria = {}

    def get(self, stops):
        if stops not in self.criteria:
            if self.engine == 'indexed':
                if self.lexicon is None:
                    self.lexicon = StopStringCriteria.clean_tokenizer_vocab(
                        self.tokenizer, stop_string_matching_mode=self.mode)
                criteria = IndexedStopCriteria(stops, self.mode, *self.lexicon)
            else:
                assert self.engine == 'hf'
                criteria = StopStringCriteria(self.tokenizer, list(stops))
            criteria.embedding_vec = criteria.embedding_vec.to(self.device)
            criteria.target_lens = criteria.target_lens.to(self.device)
            self.criteria[stops] = criteria
        return self.criteria[stops]
