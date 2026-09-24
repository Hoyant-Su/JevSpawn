import ast
import inspect


def support_empty_tools(function):
    tree = ast.parse(inspect.getsource(function))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FormattedValue)
               and ast.dump(node.value) == ast.dump(ast.parse('i + 2', mode='eval').body)]
    assert len(matches) == 1, 'Expected the original join-numbering expression exactly once.'
    matches[0].value = ast.copy_location(ast.parse('len(tools) + 1', mode='eval').body,
                                         matches[0].value)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[function.__name__]
