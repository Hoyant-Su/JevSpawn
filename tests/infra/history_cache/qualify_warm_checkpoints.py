import argparse
import ast
from collections import OrderedDict
import inspect
import json
from pathlib import Path
import textwrap

import torch

from tests.infra.history_cache import qualify
from tests.infra.history_cache.checkpoints import CheckpointHistory
from tests.infra.native_suffix_graph.qualify import tensors


def before_workload(history, workload, frozen):
    action = workload['cache_action']
    assert action in ('seed', 'restore')
    if action == 'restore':
        history.cache.entries = OrderedDict(frozen['entries'])
        history.cache.outputs = dict(frozen['outputs'])
        history.cache.bytes = frozen['bytes']
    else:
        assert not history.cache.entries


def after_workload(history, workload, frozen, result):
    if workload['cache_action'] == 'seed':
        frozen['entries'] = OrderedDict(history.cache.entries)
        frozen['outputs'] = dict(history.cache.outputs)
        frozen['bytes'] = history.cache.bytes
        frozen['tensors'] = [tensor for state in frozen['entries'].values() for tensor in tensors(state)]
        frozen['tensors'].extend(value for value in frozen['outputs'].values() if value is not None)
        frozen['copies'] = [tensor.clone() for tensor in frozen['tensors']]
    else:
        work = result['reports']['history']['history_work']
        assert work['exact_rows'] == 0 and 0 < work['matched_prefix_tokens'] < work['logical_input_tokens']
        assert work['computed_input_tokens'] > 0
    unchanged = all(torch.equal(left, right) for left, right in zip(frozen['tensors'], frozen['copies'], strict=True))
    assert unchanged
    result.update(frozen_cache_entries=len(frozen['entries']), frozen_cache_bytes=frozen['bytes'],
                  frozen_cache_unchanged=unchanged, cache_action=workload['cache_action'],
                  measurement=workload['measurement'])


class ForwardTrace:
    def prefill(self, inputs):
        self.prefill_forward_calls = []

        def record(module, args, kwargs):
            value = kwargs['input_ids'] if kwargs.get('input_ids') is not None else kwargs['inputs_embeds']
            self.prefill_forward_calls.append({'input_shape': list(value.shape[:2]),
                'ragged_suffix': kwargs.get('ragged_suffix') is not None})

        handle = self.trunk.register_forward_pre_hook(record, with_kwargs=True)
        value = super().prefill(inputs)
        handle.remove()
        return value


class NativeDecode(ForwardTrace, qualify.ObservedDecode):
    pass


class CachedDecode(ForwardTrace, qualify.ObservedHistoryDecode):
    pass


def instrumented_run(config, history_decode):
    tree = ast.parse(textwrap.dedent(inspect.getsource(qualify.run)))
    function, = tree.body
    history_position, = [index for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'history']
    function.body.insert(history_position + 1, ast.parse('frozen = {}').body[0])
    compute, = [node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == 'compute']
    compute.body.insert(0, ast.parse('before_workload(history, workload, frozen)').body[0])
    write_position, = [index for index, node in enumerate(compute.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'write_text']
    compute.body.insert(write_position, ast.parse('after_workload(history, workload, frozen, result)').body[0])
    report, = [node.value for node in ast.walk(compute) if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == 'reports[name]']
    report.keys.append(ast.Constant(value='prefill_forward_calls'))
    report.values.append(ast.parse('decoder.prefill_forward_calls', mode='eval').body)
    ast.fix_missing_locations(tree)
    namespace = dict(qualify.run.__globals__, HistoryPrefill=CheckpointHistory,
        ObservedDecode=NativeDecode, ObservedHistoryDecode=history_decode,
        before_workload=before_workload, after_workload=after_workload)
    exec(compile(tree, inspect.getsourcefile(qualify.run), 'exec'), namespace)
    namespace['run'](config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    instrumented_run(json.loads(parser.parse_args().config.read_text()), CachedDecode)
