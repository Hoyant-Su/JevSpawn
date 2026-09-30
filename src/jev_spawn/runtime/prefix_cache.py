from collections import OrderedDict


class PrefixCache:
    """Retain immutable native hybrid states for exact token prefixes."""

    def __init__(self, capacity):
        assert capacity > 0
        self.capacity = capacity
        self.entries = OrderedDict()

    def get(self, sequences, compute):
        key = tuple(tuple(sequence) for sequence in sequences)
        hit = key in self.entries
        if not hit:
            if len(self.entries) == self.capacity:
                self.entries.popitem(last=False)
            self.entries[key] = compute()
        self.entries.move_to_end(key)
        return self.entries[key], hit

    def clear(self):
        self.entries.clear()

    def resident_plan(self, keys):
        return [key in self.entries for key in keys]

    def longest_parent(self, sequence, minimum_length):
        tokens = tuple(sequence)
        candidates = [key for key in self.entries if len(key) == 1
                      and minimum_length <= len(key[0]) <= len(tokens)
                      and tokens[:len(key[0])] == key[0]]
        if not candidates:
            return None
        key = max(candidates, key=lambda item: len(item[0]))
        return list(key[0]), self.entries[key]

    def get_many(self, sequences, compute):
        keys = [tuple([tuple(sequence)]) for sequence in sequences]
        unique = list(dict.fromkeys(keys))
        hits = self.resident_plan(unique)
        missing = [key for key, hit in zip(unique, hits, strict=True) if not hit]
        values = {key: self.entries[key] for key, hit in zip(unique, hits, strict=True) if hit}
        if missing:
            states = compute([list(key[0]) for key in missing])
            values.update(zip(missing, states, strict=True))
        for key in unique:
            if key not in self.entries:
                if len(self.entries) == self.capacity:
                    self.entries.popitem(last=False)
                self.entries[key] = values[key]
            self.entries.move_to_end(key)
        hit_by_key = dict(zip(unique, hits, strict=True))
        return [values[key] for key in keys], [hit_by_key[key] for key in keys]
