import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from jev_spawn.runtime.history_cache import HistoryCache
from jev_spawn.runtime.prefix_cache import PrefixCache


def run(settings):
    module = ast.parse(Path(settings['source']).read_text())
    owner, = [node for node in module.body if isinstance(node, ast.ClassDef)]
    method, = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'get_many']
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), settings['source'], 'exec'), namespace)
    states = {}
    for key in settings['keys']:
        layer = DynamicLayer()
        values = torch.tensor(key, dtype=getattr(torch, settings['dtype'])).reshape(settings['shape'])
        layer.update(values, values.clone())
        state = DynamicCache()
        state.layers = [layer]
        states[tuple(key)] = state
    tail = SimpleNamespace()
    tail.manager = SimpleNamespace(cache=HistoryCache(settings['cache']))
    tail.roots = PrefixCache(settings['root_capacity'])
    resident = settings['keys'][settings['resident_index']]
    tail.manager.cache.store(resident, states[tuple(resident)], None)
    prefixes = [settings['keys'][index] for index in settings['request_indices']]
    loaded = []

    def prefill(missing):
        loaded.extend(missing)
        return [states[tuple(key)] for key in missing]

    actual, hits = namespace['get_many'](tail, prefixes, prefill)
    assert hits == settings['expected_hits']
    assert all(state is states[tuple(key)] for key, state in zip(prefixes, actual, strict=True))
    assert list(tail.manager.cache.entries) == [tuple(settings['keys'][settings['last_cached_index']])]
    assert loaded == [key for key in settings['keys'] if key != resident]
    print(json.dumps({'passed': True, 'rows': len(prefixes), 'loaded': loaded,
                      'hits': hits, 'retained_entries': len(tail.manager.cache.entries)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
