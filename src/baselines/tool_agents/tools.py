import ast
import json
import math

from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


class ActionError(ValueError):
    pass


OPERATORS = json.loads((ROOT / 'configs/baselines/tool_agents/calculator_operators.json').read_text())
BINARY = {resolve_symbol(node): resolve_symbol(function) for node, function in OPERATORS['binary'].items()}
UNARY = {resolve_symbol(node): resolve_symbol(function) for node, function in OPERATORS['unary'].items()}


def calculate(expression, limits):
    if not isinstance(expression, str) or len(expression) > limits['max_characters']:
        raise ActionError('Calculator expression must be a bounded string.')
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ActionError('Invalid calculator expression: ' + error.msg) from error
    if sum(1 for _ in ast.walk(tree)) > limits['max_nodes']:
        raise ActionError('Calculator expression exceeds the AST node budget.')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            result = UNARY[type(node.op)](visit(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > limits['max_exponent']:
                raise ActionError('Calculator exponent exceeds its declared bound.')
            result = BINARY[type(node.op)](left, right)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == 'sqrt' and len(node.args) == 1 and not node.keywords):
            argument = visit(node.args[0])
            if argument < 0:
                raise ActionError('Square root requires a nonnegative argument.')
            result = math.sqrt(argument)
        else:
            raise ActionError(f'Unsupported calculator syntax: {type(node).__name__}.')
        if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > limits['max_magnitude']:
            raise ActionError('Calculator result exceeds the real-number bounds.')
        return result

    try:
        return {'value': visit(tree.body)}
    except (ZeroDivisionError, OverflowError) as error:
        raise ActionError('Calculator arithmetic error: ' + str(error)) from error


def execute(tool, arguments, state, limits):
    if tool == 'calculator':
        if set(arguments) != {'expression'}:
            raise ActionError('Calculator requires only expression.')
        return calculate(arguments['expression'], limits)
    paragraphs = state.split('\n\n')
    if tool == 'search':
        if set(arguments) != {'query'} or not isinstance(arguments['query'], str) or not arguments['query'].strip():
            raise ActionError('Search requires a nonempty query string.')
        query = arguments['query'].casefold()
        return {'matches': [{'index': index, 'text': paragraph} for index, paragraph in enumerate(paragraphs)
                            if query in paragraph.casefold()]}
    if tool == 'read':
        if set(arguments) != {'index'} or type(arguments['index']) is not int:
            raise ActionError('Read requires an integer paragraph index.')
        index = arguments['index']
        if not 0 <= index < len(paragraphs):
            raise ActionError('Read index is outside the supplied evidence.')
        return {'index': index, 'text': paragraphs[index]}
    raise ActionError(f'Unknown tool: {tool}.')
