"""Allowlisted AST-to-SymPy construction and bounded symbolic execution.

No string is passed to SymPy's code-evaluating parsers. Only manually built
symbolic objects reach computation APIs. Chat uses a terminable child process.
"""
import ast
import asyncio
import multiprocessing
import re
import threading

import sympy as sp

from app.schemas.math_solver import MathSolverResult, MathTask


class MathInputError(ValueError):
    pass


FUNCTIONS = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
             "exp": sp.exp, "log": sp.log, "abs": sp.Abs}
CONSTANTS = {"pi": sp.pi, "E": sp.E}
_CAPACITY = threading.BoundedSemaphore(2)


def normalize_expression(text: str) -> str:
    """Normalize only elementary notation; prose and unknown names stay invalid."""
    text = text.strip().replace("^", "**").replace("−", "-").replace("×", "*").replace("÷", "/")
    if len(text) > 512 or not text:
        raise MathInputError("Expression is empty or too long")
    text = re.sub(r"(?<=\d)(?=[a-zA-Z(])", "*", text)
    text = re.sub(r"(?<=\))(?=[a-zA-Z(\d])", "*", text)
    text = re.sub(r"\b([a-zA-Z])\s*\(", r"\1*(", text)
    return text


def parse_expression(text: str) -> sp.Expr:
    normalized = normalize_expression(text)
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise MathInputError("Invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise MathInputError("Expression is too complex")

    def build(node, depth=0):
        if depth > 20:
            raise MathInputError("Expression is too deeply nested")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 16 or abs(node.value) > 10**12:
                raise MathInputError("Numeric literal is too large")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            if re.fullmatch(r"[a-zA-Z]", node.id):
                return sp.Symbol(node.id, real=True)
            raise MathInputError("Unsupported symbol")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = build(node.left, depth + 1), build(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return sp.Add(left, right, evaluate=False)
            if isinstance(node.op, ast.Sub):
                return sp.Add(left, -right, evaluate=False)
            if isinstance(node.op, ast.Mult):
                return sp.Mul(left, right, evaluate=False)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise MathInputError("Division by zero")
                return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
            if isinstance(node.op, ast.Pow):
                # Reject nested/variable exponents and bound polynomial growth.
                if not isinstance(node.right, (ast.Constant, ast.UnaryOp)) or not right.is_Rational or abs(right) > 100:
                    raise MathInputError("Unsupported or excessive exponent")
                return sp.Pow(left, right, evaluate=False)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords:
                return FUNCTIONS[node.func.id](build(node.args[0], depth + 1))
        raise MathInputError("Unsupported expression syntax")

    return build(tree.body)


class MathSolver:
    """Deterministic real-domain mathematics with explicit verification status."""

    @staticmethod
    def solve(task: MathTask) -> MathSolverResult:
        try:
            task = MathTask.model_validate(task.model_dump())
            if task.intent != "math_problem":
                raise MathInputError("No mathematical task")
            expr = parse_expression(task.left if task.operation == "solve_equation" else task.expression)
            right = parse_expression(task.right) if task.operation == "solve_equation" else None
            symbols = expr.free_symbols | (right.free_symbols if right is not None else set())
            variable = sp.Symbol(task.variable, real=True) if task.variable else None
            if task.operation in {"solve_equation", "differentiate", "integrate"}:
                if variable is None:
                    if len(symbols) != 1:
                        raise MathInputError("Specify one unambiguous variable")
                    variable = next(iter(symbols))
                if symbols - {variable}:
                    raise MathInputError("Multiple variables require explicit assumptions")
            notes = []
            if task.operation == "solve_equation":
                residual = expr - right
                # Rational polynomial equations have complete, checkable root sets.
                numerator, denominator = sp.fraction(sp.together(residual))
                polynomial = sp.Poly(numerator, variable)
                if polynomial.degree() > 8:
                    raise MathInputError("Equation degree exceeds supported limit")
                roots = sp.solveset(residual, variable, domain=sp.S.Reals)
                if roots is sp.S.EmptySet:
                    result = []
                elif isinstance(roots, sp.FiniteSet):
                    for root in roots:
                        if sp.simplify(residual.subs(variable, root)) != 0 or sp.simplify(denominator.subs(variable, root)) == 0:
                            raise MathInputError("Solution could not be verified")
                    result = [str(root) for root in sorted(roots, key=sp.default_sort_key)]
                else:
                    raise MathInputError("Equation has no supported finite solution set")
                notes.append("Real-domain solutions; original expression domain restrictions apply.")
            elif task.operation == "differentiate":
                computed = sp.diff(expr, variable)
                if computed.has(sp.Derivative):
                    raise MathInputError("Derivative could not be evaluated")
                result = str(computed)
            elif task.operation == "integrate":
                computed = sp.integrate(expr, variable)
                if computed.has(sp.Integral) or sp.simplify(sp.diff(computed, variable) - expr) != 0:
                    raise MathInputError("Integral could not be verified")
                result = str(computed)
                notes.append("Indefinite integral: add an arbitrary constant C.")
            else:
                computed = sp.simplify(expr)
                if computed.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
                    raise MathInputError("Expression is undefined or non-finite")
                result = str(computed)
                if task.operation == "simplify":
                    notes.append("Equivalent on the original expression's domain; excluded values remain excluded.")
            if len(str(result)) > 4096:
                raise MathInputError("Result exceeds size limit")
            return MathSolverResult(success=True, problem_type=task.operation, expression=task.expression or f"{task.left} = {task.right}",
                                    result=result, variable=str(variable) if variable else None, notes=notes)
        except Exception as exc:
            return MathSolverResult(success=False, problem_type=task.operation,
                                    error="Calculation verification was unavailable. Please check the expression and variable.",
                                    error_type=type(exc).__name__)

    @staticmethod
    async def execute(task: MathTask, timeout_seconds: float = 5.0) -> MathSolverResult:
        return await asyncio.to_thread(_bounded_solve, task, timeout_seconds)


def _worker(connection, payload):
    try:
        connection.send(MathSolver.solve(MathTask.model_validate(payload)).model_dump())
    finally:
        connection.close()


def _bounded_solve(task: MathTask, timeout_seconds: float) -> MathSolverResult:
    failure = MathSolverResult(success=False, problem_type=task.operation,
                               error="Calculation verification was unavailable. Try a simpler expression.")
    if not _CAPACITY.acquire(blocking=False):
        return failure.model_copy(update={"error_type": "SolverBusy"})
    parent = child = process = None
    try:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(child, task.model_dump()), daemon=True)
        process.start()
        child.close()
        if parent.poll(timeout_seconds):
            return MathSolverResult.model_validate(parent.recv())
        return failure.model_copy(update={"error_type": "SolverTimeout"})
    except Exception as exc:
        return failure.model_copy(update={"error_type": type(exc).__name__})
    finally:
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        if parent is not None:
            parent.close()
        if child is not None:
            child.close()
        _CAPACITY.release()
