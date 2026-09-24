from collections import OrderedDict

from transformers.cache_utils import DynamicLayer


def state_bytes(state):
    tensors = [tensor for layer in state.layers for tensor in
               ([layer.keys, layer.values] if isinstance(layer, DynamicLayer)
                else [*layer.conv_states.values(), *layer.recurrent_states.values()])]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class HistoryCache:
    """Immutable exact-token checkpoints, including branch ancestors, bounded on GPU."""

    def __init__(self, settings):
        self.settings = settings
        self.entries = OrderedDict()
        self.outputs = {}
        self.bytes = settings['initial_bytes']

    def prefix(self, sequence):
        matches = [key for key in self.entries if len(key) <= len(sequence) and tuple(sequence[:len(key)]) == key]
        return max(matches, key=len) if matches else ()

    def read(self, key):
        self.entries.move_to_end(key)
        return self.entries[key]

    def store(self, sequence, state, output):
        key = tuple(sequence)
        assert state.get_seq_length() == len(key)
        size = state_bytes(state) + (0 if output is None else output.numel() * output.element_size())
        assert size <= self.settings['max_bytes'], 'One native history checkpoint exceeds the declared GPU cache budget.'
        if key in self.entries:
            self.entries.move_to_end(key)
            if self.outputs[key] is None and output is not None:
                self.entries.pop(key)
                self.outputs.pop(key)
                self.bytes -= state_bytes(state)
            else:
                return
        while self.entries and (len(self.entries) >= self.settings['max_entries']
                                or self.bytes + size > self.settings['max_bytes']):
            previous_key, previous = self.entries.popitem(last=False)
            previous_output = self.outputs.pop(previous_key)
            self.bytes -= state_bytes(previous)
            self.bytes -= 0 if previous_output is None else previous_output.numel() * previous_output.element_size()
        self.entries[key] = state
        self.outputs[key] = output
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.outputs.clear()
        self.bytes = self.settings['initial_bytes']
