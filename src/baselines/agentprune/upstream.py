import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType


def load_core(source):
    source = Path(source)
    for name in ['AgentPrune', 'AgentPrune.graph', 'AgentPrune.agents', 'AgentPrune.llm',
                 'AgentPrune.prompt', 'AgentPrune.tools', 'AgentPrune.tools.search',
                 'AgentPrune.utils', 'experiments', 'dataset']:
        module = ModuleType(name)
        module.__path__ = [str(source.joinpath(*name.split('.')))]
        sys.modules[name] = module
    sys.modules['AgentPrune.graph'].Node = importlib.import_module('AgentPrune.graph.node').Node
    importlib.import_module('AgentPrune.prompt.mmlu_prompt_set')
    importlib.import_module('AgentPrune.agents.analyze_agent')
    final = source / 'AgentPrune/agents/final_decision.py'
    tree = ast.parse(final.read_text())
    tree.body = [node for node in tree.body if
                 isinstance(node, (ast.Import, ast.ImportFrom)) and
                 getattr(node, 'module', '') != 'AgentPrune.tools.coding.python_executor'
                 or isinstance(node, ast.ClassDef) and node.name == 'FinalRefer']
    module = ModuleType('AgentPrune.agents.final_decision')
    sys.modules[module.__name__] = module
    exec(compile(tree, str(final), 'exec'), module.__dict__)
    return importlib.import_module('AgentPrune.graph.graph').Graph


def graph_kwargs(source):
    path = Path(source) / 'experiments/run_mmlu.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'get_kwargs']
    tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace['get_kwargs']


def training_function(source, observer):
    path = Path(source) / 'experiments/train_mmlu.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef))
    loop = next(node for node in function.body if isinstance(node, ast.For))
    loop.iter = ast.parse('range(observer.start_iteration, num_iters)', mode='eval').body
    loop.body.insert(0, ast.parse('observer.begin(i_iter)').body[0])
    for node in loop.body:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
            if [part.id for part in node.target.elts] == ['i_record', 'record']:
                node.body.insert(1, ast.parse('observer.graphs.append(realized_graph)').body[0])
    gather_index = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'raw_results'
                                for target in node.targets))
    loop.body.insert(gather_index + 1, ast.parse('observer.check_execution()').body[0])
    loop.body.append(ast.parse('observer.end(locals())').body[0])
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'loader'
                                               for target in node.targets):
            node.value = ast.parse('observer.loader(infinite_data_loader)', mode='eval').body
    index = function.body.index(loop)
    function.body.insert(index, ast.parse('observer.restore(optimizer)').body[0])
    namespace = {'observer': observer}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace['train'], ast.unparse(tree)


def evaluation_function(source, observer):
    path = Path(source) / 'experiments/evaluate_mmlu.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef))
    function.body = [node for node in function.body if not (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'accuracy'
                                            for target in node.targets)
        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'accuracy')]
    loop = next(node for node in function.body if isinstance(node, ast.For))
    loop.body.insert(0, ast.parse('observer.begin(i_batch)').body[0])
    record_loop = next(node for node in loop.body if isinstance(node, ast.For))
    record_loop.body.insert(1, ast.parse('observer.graphs.append(realized_graph)').body[0])
    gather_index = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'raw_results'
                                for target in node.targets))
    loop.body.insert(gather_index + 1, ast.parse('observer.check_execution()').body[0])
    loop.body = [node for node in loop.body if not (
        isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
        and [part.id for part in node.target.elts] == ['raw_answer', 'record'])]
    loop.body.append(ast.parse('observer.end_evaluation(locals())').body[0])
    next(node for node in function.body if isinstance(node, ast.Return)).value = ast.parse(
        'observer.results', mode='eval').body
    namespace = {'observer': observer}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace['evaluate'], ast.unparse(tree)
