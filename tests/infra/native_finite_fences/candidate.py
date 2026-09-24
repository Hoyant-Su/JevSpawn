import ast
import inspect
import textwrap


def without_timing_fences(function, expected_fences):
    function = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    fences = [node for node in ast.walk(tree)
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and ast.unparse(node.value.func) == 'torch.cuda.synchronize']
    assert len(fences) == expected_fences
    for node in fences:
        node.value = ast.copy_location(ast.Constant(value=None), node.value)
    namespace = dict(function.__globals__)
    exec(compile(tree, inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[function.__name__]
