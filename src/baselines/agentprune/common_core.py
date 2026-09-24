import ast
import asyncio
from contextvars import ContextVar
import copy
from functools import lru_cache
import json
from pathlib import Path
from threading import Lock

import torch

from baselines.agentprune.evaluation import load_checkpoint
from baselines.agentprune.upstream import load_core


RNG = ContextVar('agentprune_common_rng')
LOAD_LOCK = Lock()


def instrument(graph_class, source):
    path = Path(source) / 'AgentPrune/graph/graph.py'
    tree = ast.parse(path.read_text())
    graph = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {'arun', 'construct_spatial_connection', 'construct_temporal_connection'}
    methods = [node for node in graph.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name in names]
    for method in methods:
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'rand':
                    node.keywords.append(ast.keyword(arg='generator', value=ast.parse('RNG.get()', mode='eval').body))
            if isinstance(node, ast.ExceptHandler):
                node.body = [ast.Raise()]
            if isinstance(node, ast.If) and ast.unparse(node.test) == 'len(final_answers) == 0':
                node.body = ast.parse("raise RuntimeError('Original decision node produced no output.')").body
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *methods], type_ignores=[])
    namespace = {'torch': torch, 'asyncio': asyncio, 'RNG': RNG}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    for name in names:
        setattr(graph_class, name, namespace[name])
    return ast.unparse(module)


@lru_cache(maxsize=1)
def loaded(source, checkpoint, training_config):
    graph_class = load_core(source)
    executed = instrument(graph_class, source)
    configuration = json.loads(Path(training_config).read_text())
    graph = load_checkpoint(Path(checkpoint), configuration)
    assert configuration['training']['num_rounds'] == 1
    assert graph.decision_node.agent_name == 'FinalRefer'
    assert all(node.agent_name == 'AnalyzeAgent' and node.role != 'Wiki Searcher'
               for node in graph.nodes.values())
    provenance = {'checkpoint': checkpoint,
                  'training_config': configuration,
                  'executed_graph_methods': executed,
                  'frozen_topology': {name: getattr(graph, name).detach().tolist() for name in
                                      ['spatial_logits', 'temporal_logits', 'spatial_masks', 'temporal_masks']}}
    return graph, provenance


def task_graph(settings):
    with LOAD_LOCK:
        graph, provenance = loaded(settings['source_directory'], settings['checkpoint'], settings['training_config'])
        graph = copy.deepcopy(graph)
    for node in [*graph.nodes.values(), graph.decision_node]:
        node.clear_connections()
        node.inputs, node.outputs, node.raw_inputs = [], [], []
        node.last_memory = {'inputs': [], 'outputs': [], 'raw_inputs': []}
        node.conversation_history = []
    for node in graph.nodes.values():
        node.wiki_summary = ''
    return graph, provenance
