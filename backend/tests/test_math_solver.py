"""Real symbolic execution and security regressions for Phase 9.1."""
import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.math_solver import MathTask
from app.tools.math_solver import MathSolver, parse_expression
from app.tools.math_task_parser import MathTaskParser
from app.tools.solver_router import SolverRouter
from app.schemas.retrieval import RetrievedChunk
import uuid


@pytest.mark.parametrize("fields,expected", [
    ({"operation": "solve_equation", "left": "2*x+4", "right": "10"}, ["3"]),
    ({"operation": "solve_equation", "left": "x**2-5*x+6", "right": "0"}, ["2", "3"]),
    ({"operation": "differentiate", "expression": "x**3+2*x"}, "3*x**2 + 2"),
    ({"operation": "integrate", "expression": "x**2"}, "x**3/3"),
    ({"operation": "calculate", "expression": "sqrt(16)+2"}, "6"),
    ({"operation": "simplify", "expression": "(x**2-1)/(x-1)"}, "x + 1"),
])
def test_supported_operations(fields, expected):
    result = MathSolver.solve(MathTask(**fields))
    assert result.success, result
    assert result.result == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('whoami')", "x.__class__", "open('file')", "[x for x in [1]]",
    "lambda: 1", "x[0]", "True", "1/0", "sqrt(x, 2)", "2**1000000", "2**(2**20)",
    "10**100**100", "a" * 513, "sin(x, evaluate=False)", "sum([1,2])",
    "(x := 2)", "(1, 2)", "{1:2}", "1000000000000000000000",
])
def test_unsafe_and_excessive_input_is_rejected(expression):
    with pytest.raises(ValueError):
        parse_expression(expression)


@pytest.mark.parametrize("fields", [
    {"operation": "solve_equation", "left": "x+y", "right": "2"},
    {"operation": "differentiate", "expression": "x+y"},
    {"operation": "solve_equation", "left": "x", "right": "x"},
    {"operation": "solve_equation", "left": "sin(x)", "right": "0"},
    {"operation": "calculate", "expression": "sqrt(-1)/0"},
])
def test_unverifiable_or_ambiguous_task_returns_safe_failure(fields):
    result = MathSolver.solve(MathTask(**fields))
    assert not result.success
    assert result.result is None
    assert "verification was unavailable" in result.error
    assert "Traceback" not in result.error


def test_domain_and_integral_notes():
    simple = MathSolver.solve(MathTask(operation="simplify", expression="(x*x-1)/(x-1)"))
    assert "excluded values" in simple.notes[0]
    integral = MathSolver.solve(MathTask(operation="integrate", expression="x**2"))
    assert "constant C" in integral.notes[0]
    # A cancelled denominator must never add an excluded root.
    equation = MathSolver.solve(MathTask(operation="solve_equation", left="(x*x-1)/(x-1)", right="2"))
    assert equation.success and equation.result == []


def test_unevaluated_integral_and_solver_exception_are_safe(monkeypatch):
    import sympy as sp
    monkeypatch.setattr(sp, "integrate", lambda expr, variable: sp.Integral(expr, variable))
    result = MathSolver.solve(MathTask(operation="integrate", expression="x*x"))
    assert not result.success
    def broken(*args):
        raise RuntimeError("private calculation details")
    monkeypatch.setattr(sp, "solveset", broken)
    result = MathSolver.solve(MathTask(operation="solve_equation", left="x", right="1"))
    assert not result.success and "private" not in result.error


def test_no_real_roots_and_explicit_variable_for_constant():
    result = MathSolver.solve(MathTask(operation="solve_equation", left="x*x+1", right="0"))
    assert result.success and result.result == []
    result = MathSolver.solve(MathTask(operation="differentiate", expression="5", variable="x"))
    assert result.success and result.result == "0"


@pytest.mark.parametrize("fields", [
    {"operation": "execute_python", "expression": "x"},
    {"operation": "calculate", "expression": "x", "left": "x"},
    {"operation": "solve_equation", "left": "x", "right": "1", "expression": "x"},
    {"intent": "normal", "operation": "calculate", "expression": "2"},
    {"operation": "calculate", "expression": "2", "code": "import os"},
    {"operation": "differentiate", "expression": "x", "variable": "__import__"},
])
def test_structured_fields_are_validated(fields):
    with pytest.raises(ValidationError):
        MathTask.model_validate(fields)


@pytest.mark.parametrize("query", [
    "Giải phương trình x^2 - 4 = 0", "Tính đạo hàm của x^3", "Tính tích phân x^2",
    "Giải bài 5", "Giải phương trình này", "Solve x^2 - 4 = 0", "Calculate the derivative",
    "sqrt(16)+2", "x**2 - 4 = 0",
])
def test_bilingual_math_routing(query):
    assert SolverRouter.route(query) == "math_problem"


@pytest.mark.parametrize("query", [
    "Tóm tắt tài liệu này", "Giải thích attention mechanism", "Tóm tắt chương 2",
    "Giải thích khái niệm gradient descent", "Tài liệu này nói gì về transformers?",
    "Explain the derivative concept", "Summarize chapter 2", "What is attention?",
])
def test_normal_routing(query):
    assert SolverRouter.route(query) == "normal"


@pytest.mark.parametrize("query,operation", [
    ("Giải phương trình x^2 - 5x + 6 = 0", "solve_equation"),
    ("Tính đạo hàm của x^3 + 2x", "differentiate"),
    ("Solve 2*x + 3 = 7", "solve_equation"),
    ("differentiate x**3 + 2*x with respect to x", "differentiate"),
    ("Integrate x^2 wrt x", "integrate"),
    ("Tính nguyên hàm của x^2 theo x", "integrate"),
    ("Calculate sqrt(16)+2", "calculate"),
    ("Simplify (x**2 - 1)/(x - 1)", "simplify"),
])
def test_task_parser(query, operation):
    task = MathTaskParser.parse(query)
    assert task.operation == operation
    assert MathSolver.solve(task).success


def chunk(content, index=0, document_id=None, passed=True):
    return RetrievedChunk(chunk_id=uuid.uuid4(), document_id=document_id or uuid.uuid4(), content=content,
                          page_number_start=1, page_number_end=1, chunk_index=index, final_rank=1,
                          passed_relevance_gate=passed)


def test_numbered_exercise_and_adjacent_chunks():
    document_id = uuid.uuid4()
    chunks = [chunk("Bài 3: Giải phương trình x^2 -", document_id=document_id),
              chunk("5x + 6 = 0\nBài 4: Solve x=99", 1, document_id)]
    task = MathTaskParser.from_retrieved("Giải bài 3 trong tài liệu", chunks)
    assert task.source == "retrieved_document"
    assert MathSolver.solve(task).result == ["2", "3"]


@pytest.mark.parametrize("chunks", [
    [chunk("Bài 4: Solve x=99")],
    [chunk("Bài 3: Solve x=99", passed=False)],
    [chunk("Bài 3: Solve x=99"), chunk("Bài 3: Solve x=100")],
    [chunk("Bài 3: Solve x^2 -"), chunk("5x + 6 = 0", 1)],
    [chunk("Bài 3: Solve x^2 -", document_id=uuid.UUID(int=1)),
     chunk("5x + 6 = 0", 2, uuid.UUID(int=1))],
    [chunk("Bài 3: Tóm tắt các định nghĩa")],
])
def test_wrong_ambiguous_incomplete_or_unrelated_exercises_are_not_solved(chunks):
    with pytest.raises(ValueError):
        MathTaskParser.from_retrieved("Giải bài 3", chunks)


@pytest.mark.asyncio
async def test_real_bounded_process_execution():
    result = await MathSolver.execute(MathTask(operation="calculate", expression="sqrt(16)+2"), 10)
    assert result.success and result.result == "6"


@pytest.mark.asyncio
async def test_timeout_is_safe_and_next_request_still_works():
    result = await MathSolver.execute(MathTask(operation="calculate", expression="2+2"), 0.001)
    assert not result.success and result.error_type == "SolverTimeout"
    assert (await MathSolver.execute(MathTask(operation="calculate", expression="2+2"), 10)).result == "4"


def test_solver_source_has_no_executable_string_parsers():
    for path in (Path(__file__).parents[1] / "app" / "tools").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                assert name not in {"eval", "exec", "parse_expr", "sympify", "lambdify"}
