from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import time

import torch
from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask, create_recurrent_attention_mask

from jev_spawn.algo import structured


@dataclass
class BranchRequest:
    source: object
    branches: object = None

    def reorder_cache(self, branches):
        self.branches = branches


class SuffixGraph:
    def __init__(self, backend, forward, branch, inputs, settings, pool, stream):
        self.backend, self.forward = backend, forward
        self.source = branch.source
        self.branches = branch.branches.clone()
        self.cache = deepcopy(self.source)
        self.cache.reorder_cache(self.branches)
        self.key_values = [(layer, layer.keys, layer.values) for layer in self.cache.layers
                           if isinstance(layer, DynamicLayer)]
        self.ids = inputs['input_ids'].clone()
        self.positions = inputs['position_ids'].clone()
        self.masks = self.native_masks(inputs['attention_mask'])
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream(backend.device))
        with torch.cuda.stream(stream):
            for _ in range(settings['warmup_steps']):
                self.restore_references()
                self.compute()
        torch.cuda.current_stream(backend.device).wait_stream(stream)
        torch.cuda.synchronize(backend.device)
        self.restore_references()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.compute()
        torch.cuda.synchronize(backend.device)
        self.capture_seconds = time.perf_counter() - started

    def restore_references(self):
        for layer, keys, values in self.key_values:
            layer.keys, layer.values = keys, values

    def native_masks(self, mask):
        self.restore_references()
        trunk = self.backend.model.model.language_model
        kwargs = dict(config=trunk.config, inputs_embeds=trunk.embed_tokens(self.ids),
                      attention_mask=mask, past_key_values=self.cache, position_ids=self.positions)
        return {'full_attention': create_causal_mask(**kwargs),
                'linear_attention': create_recurrent_attention_mask(**kwargs)}

    def compute(self):
        for source, target in zip(self.source.layers, self.cache.layers):
            if isinstance(source, DynamicLayer):
                torch.index_select(source.keys, 0, self.branches, out=target.keys)
                torch.index_select(source.values, 0, self.branches, out=target.values)
            if isinstance(source, LinearAttentionCacheLayerMixin):
                for index in source.conv_states:
                    torch.index_select(source.conv_states[index], 0, self.branches,
                                       out=target.conv_states[index])
                    torch.index_select(source.recurrent_states[index], 0, self.branches,
                                       out=target.recurrent_states[index])
        self.output = self.forward(input_ids=self.ids, attention_mask=self.masks,
            position_ids=self.positions, past_key_values=self.cache, use_cache=True)

    def replay(self, branch, inputs):
        self.ids.copy_(inputs['input_ids'])
        self.positions.copy_(inputs['position_ids'])
        self.branches.copy_(branch.branches)
        masks = self.native_masks(inputs['attention_mask'])
        for name, mask in masks.items():
            if mask is not None:
                self.masks[name].copy_(mask)
            else:
                assert self.masks[name] is None
        self.graph.replay()
        return self.output


class SuffixGraphs:
    def __init__(self, backend, settings):
        self.backend, self.settings = backend, settings
        self.pool, self.stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)
        self.graphs = {}
        self.forward = backend.model.model.forward

    def run(self, *args, **kwargs):
        branch = kwargs['past_key_values']
        assert isinstance(branch, BranchRequest)
        key = (id(branch.source), tuple(kwargs['input_ids'].shape), tuple(kwargs['attention_mask'].shape))
        if key not in self.graphs:
            self.graphs[key] = SuffixGraph(self.backend, self.forward, branch, kwargs,
                                         self.settings, self.pool, self.stream)
        return self.graphs[key].replay(branch, kwargs)

    @contextmanager
    def installed(self):
        original = structured.deepcopy
        structured.deepcopy = BranchRequest
        self.backend.model.model.forward = self.run
        try:
            yield
        finally:
            structured.deepcopy = original
            self.backend.model.model.forward = self.forward
