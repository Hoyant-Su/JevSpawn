from jev_spawn.infra.finite_graph import RaggedFiniteGraphTail


class StableFiniteGraphTail(RaggedFiniteGraphTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings):
        buckets = tuple(settings['batch_buckets'])
        assert buckets == tuple(sorted(set(buckets))) and all(size > 0 for size in buckets)
        buckets = tuple(size for size in buckets if size <= runtime.branch_batch_size)
        assert max(buckets) == runtime.branch_batch_size
        assert runtime.graph_cache_size >= len(buckets)
        self.buckets = buckets
        self.capacity = backend.config['max_input_tokens']
        self.graph_ids = tuple(backend.answer_label_ids)
        assert len(set(self.graph_ids)) == len(self.graph_ids)
        super().__init__(backend, runtime, prefix_cache, state_copy_settings)

    def graph_layout(self, sequences, native_ids):
        assert sequences and all(sequence and len(sequence) <= self.capacity for sequence in sequences)
        assert tuple(native_ids) == self.graph_ids[:len(native_ids)]
        physical_rows = next(size for size in self.buckets if size >= len(sequences))
        return physical_rows, physical_rows, self.capacity, self.graph_ids

    def graph_inputs(self, states, last_tokens, physical_rows):
        assert len(states) == len(last_tokens) and states and len(states) <= physical_rows
        padding = physical_rows - len(states)
        return states + [states[-1]] * padding, last_tokens + [last_tokens[-1]] * padding

    def graph_outputs(self, logits, active_rows, native_ids):
        assert tuple(native_ids) == self.graph_ids[:len(native_ids)]
        return logits[:active_rows, :len(native_ids)]
