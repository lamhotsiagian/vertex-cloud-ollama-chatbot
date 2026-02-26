import ast
import operator as op

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

class CalcError(Exception):
    pass

def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError("Only numeric constants are allowed.")
    if isinstance(node, ast.Num):  # compatibility
        return node.n

    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        fn = _ALLOWED_OPS.get(type(node.op))
        if not fn:
            raise CalcError(f"Operator not allowed: {type(node.op).__name__}")
        return fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        fn = _ALLOWED_OPS.get(type(node.op))
        if not fn:
            raise CalcError(f"Unary operator not allowed: {type(node.op).__name__}")
        return fn(operand)

    raise CalcError(f"Expression not allowed: {type(node).__name__}")

def safe_calculate(expr: str) -> float:
    expr = expr.strip()
    if not expr:
        raise CalcError("Empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"Invalid expression: {e}")
    return _eval(tree.body)
