import ast
from functools import partial
import json
from pathlib import Path

from baselines.common.lats import load_core, solve_with_core
from baselines.common.policy_frames import policy_sample


class SampleBoundary(ast.NodeTransformer):
    def visit_ListComp(self, node):
        node.elt = ast.Call(func=ast.Name(id='policy_sample', ctx=ast.Load()),
                            args=[node.elt.left, node.elt.right, ast.Name(id='frame_grammar', ctx=ast.Load())],
                            keywords=[])
        return node


def load_frame_core(source, gpt, environment, *, grammar):
    core, task = load_core(source, gpt, environment)
    path = Path(source) / 'hotpot/lats.py'
    parsed = ast.parse(path.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'get_samples')
    tree = ast.fix_missing_locations(ast.Module(body=[SampleBoundary().visit(function)], type_ignores=[]))
    core.update(policy_sample=policy_sample, frame_grammar=grammar)
    exec(compile(tree, str(path), 'exec'), core)
    return core, task


def solve(task, environment, complete, settings, prompts):
    grammar = json.loads(Path(settings['frame_grammar']).read_text())
    return solve_with_core(task, environment, complete, settings, prompts,
                           partial(load_frame_core, grammar=grammar))
