from collections import OrderedDict

import torch
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

from jev_spawn.infra.qwen35 import ragged_gdn


class CapturedChunk:
    def __init__(self, operation, inputs, options, settings, pool, stream):
        self.inputs = {name: value.clone() for name, value in inputs.items()}
        names = settings['copy_arguments']
        dtypes = dict.fromkeys(self.inputs[name].dtype for name in names)
        self.copy_groups = [([name for name in names if self.inputs[name].dtype == dtype],
                             [self.inputs[name] for name in names if self.inputs[name].dtype == dtype])
                            for dtype in dtypes]
        self.options = {**options, 'chunk_size': settings['chunk_tokens']}
        # FLA's small metadata cache may evict tensors still referenced by a captured graph.
        self.indices = prepare_chunk_indices(self.inputs['cu_seqlens'], settings['chunk_tokens'],
                                            cu_seqlens_cpu=options['cu_seqlens_cpu'])
        self.offsets = prepare_chunk_offsets(self.inputs['cu_seqlens'], settings['chunk_tokens'])
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(settings['warmup_steps']):
                operation(**self.inputs, **self.options)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.output = operation(**self.inputs, **self.options)

    def __call__(self, inputs):
        for names, destinations in self.copy_groups:
            torch._foreach_copy_(destinations, [inputs[name] for name in names], non_blocking=True)
        self.graph.replay()
        return self.output


class ChunkGraphs:
    def __init__(self, operation, settings, device):
        self.operation, self.settings = operation, settings
        self.entries = OrderedDict()
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=device)
        self.captures = settings['initial_count']
        self.replays = settings['initial_count']

    def __call__(self, q, k, v, **kwargs):
        inputs = dict(q=q, k=k, v=v, **{
            name: kwargs[name] for name in self.settings['tensor_arguments']})
        options = {name: value for name, value in kwargs.items() if name not in inputs}
        key = (tuple(kwargs['cu_seqlens_cpu'].tolist()),
               tuple((name, tuple(value.shape), value.dtype) for name, value in inputs.items()))
        if key not in self.entries:
            if len(self.entries) == self.settings['capacity']:
                self.entries.popitem(last=False)
            self.entries[key] = CapturedChunk(self.operation, inputs, options,
                self.settings, self.pool, self.stream)
            self.captures += 1
        self.entries.move_to_end(key)
        self.replays += 1
        # Both outputs are consumed on this stream before the next layer replays the graph.
        return self.entries[key](inputs)


def install(settings, device):
    graphs = ChunkGraphs(ragged_gdn.chunk_gated_delta_rule, settings, device)
    ragged_gdn.chunk_gated_delta_rule = graphs
    return graphs
