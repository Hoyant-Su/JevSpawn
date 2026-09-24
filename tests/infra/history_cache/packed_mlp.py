from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from tests.infra.history_cache.replay_checkpoint import ObservedCheckpoint, ObservedReference


def packed_forward(original, descriptor):
    def forward(hidden_states):
        return descriptor.unpack(original(descriptor.pack(hidden_states)))
    return forward


class PackedMLP:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_extension = self.extend_states
        self.extend_states = self.extend_packed

    def extend_packed(self, backend, states, tails, work):
        lengths = [len(tail) for tail in tails if tail]
        if len(set(lengths)) <= 1:
            return self.native_extension(backend, states, tails, work)
        descriptor = RaggedSuffix(lengths, backend.device)
        bindings = [(layer.mlp, layer.mlp.forward)
                    for layer in backend.model.model.language_model.layers]
        for module, original in bindings:
            module.forward = packed_forward(original, descriptor)
        result = self.native_extension(backend, states, tails, work)
        for module, original in bindings:
            module.forward = original
        return result


class PackedReference(PackedMLP, ObservedReference):
    pass


class PackedCheckpoint(PackedMLP, ObservedCheckpoint):
    pass
