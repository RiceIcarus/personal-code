import ast
import math
import operator
from typing import Callable

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 500
MAX_AST_NODES = 100
MAX_POWER = 1000
MAX_INTEGER_BITS = 4096

BINOP_FUNCTIONS: dict[type[ast.operator], Callable[[object, object], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_FUNCTIONS: dict[type[ast.unaryop], Callable[[object], object]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

CONSTANTS = {'pi': math.pi, 'e': math.e}
FUNCTIONS = {
    'abs': abs,
    'cos': math.cos,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'sin': math.sin,
    'sqrt': math.sqrt,
    'tan': math.tan,
}


def _evaluate(node: ast.AST) -> int | float:
    """Evaluate one validated numeric expression node."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError('non-finite numbers are not supported')
        if isinstance(node.value, int) and node.value.bit_length() > MAX_INTEGER_BITS:
            raise ValueError('integer is too large')
        return node.value

    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in BINOP_FUNCTIONS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise ValueError(f'exponent must be between -{MAX_POWER} and {MAX_POWER}')
        result = BINOP_FUNCTIONS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_FUNCTIONS:
        result = UNARY_FUNCTIONS[type(node.op)](_evaluate(node.operand))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError('function is not supported')
        if node.func.id == 'log' and len(node.args) not in (1, 2):
            raise ValueError('log accepts one or two arguments')
        if node.func.id != 'log' and len(node.args) != 1:
            raise ValueError(f'{node.func.id} accepts one argument')
        result = function(*[_evaluate(arg) for arg in node.args])
    else:
        raise ValueError('expression contains an unsupported operation')

    if isinstance(result, int) and result.bit_length() > MAX_INTEGER_BITS:
        raise ValueError('result is too large')
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError('result is not finite')
    return result


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a basic mathematical expression.

    Supports numbers, parentheses, +, -, *, /, //, %, **, pi, e, and selected
    math functions such as sqrt, sin, cos, tan, log, log10, exp, and abs.
    """
    expression = expression.strip()
    if not expression:
        return 'Error: expression cannot be empty'
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f'Error: expression is too long (maximum {MAX_EXPRESSION_LENGTH} characters)'

    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError):
        return 'Error: invalid mathematical expression'

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        return f'Error: expression is too complex (maximum {MAX_AST_NODES} nodes)'

    try:
        result = _evaluate(tree.body)
    except (ArithmeticError, TypeError, ValueError) as error:
        return f'Error: {error}'

    return f'Expression: {expression}\nResult: {result}'
