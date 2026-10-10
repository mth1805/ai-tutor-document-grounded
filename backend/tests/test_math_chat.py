"""HTTP, SSE, tutor policy, and persistence coverage for the existing chat stack."""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.llm import MockLLMProvider, set_llm_provider, reset_llm_provider
from app.schemas.math_solver import MathSolverResult, MathTask
from app.schemas.retrieval import RetrievalResponse, RetrievalTimingMetrics
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.tools.math_solver import MathSolver
from app.rag.prompt_builder import ChatMode, PromptBuilder
from app.services.workspace_service import _IN_MEMORY_WORKSPACES
from app.services.conversation_service import _IN_MEMORY_CONVERSATIONS, _IN_MEMORY_MESSAGES
from tests.test_math_solver import chunk

client = TestClient(app)
headers = {"Authorization": "Bearer test-token:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for store in (_IN_MEMORY_WORKSPACES, _IN_MEMORY_CONVERSATIONS, _IN_MEMORY_MESSAGES):
        store.clear()
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    yield
    reset_llm_provider()
    for store in (_IN_MEMORY_WORKSPACES, _IN_MEMORY_CONVERSATIONS, _IN_MEMORY_MESSAGES):
        store.clear()


def conversation():
    workspace = client.post("/api/v1/workspaces", headers=headers, json={"name": "Math"}).json()["id"]
    conversation_id = client.post(f"/api/v1/workspaces/{workspace}/conversations", headers=headers,
                                  json={"title": "Exercises"}).json()["id"]
    return workspace, conversation_id


def mock_retrieval(monkeypatch, chunks=None, sufficient=True, order=None):
    async def retrieve(db, workspace_id, user_id, request):
        if order is not None:
            order.append("retrieve")
        return RetrievalResponse(workspace_id=workspace_id, query=request.query, results=chunks or [],
                                 total_results=len(chunks or []), has_sufficient_evidence=sufficient,
                                 relevance_threshold=0.35, routing_path="FAST",
                                 timings=RetrievalTimingMetrics(query_embedding_ms=0, dense_retrieval_ms=0,
                                     lexical_retrieval_ms=0, rrf_ms=0, rerank_ms=0, total_retrieval_ms=0))
    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)


def local_solver(monkeypatch, order=None):
    async def execute(task, timeout_seconds):
        if order is not None:
            order.append("solve")
        return MathSolver.solve(task)
    monkeypatch.setattr(MathSolver, "execute", execute)


def events(text):
    result = []
    for block in text.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2:
            result.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return result


def test_direct_math_http_uses_real_process_and_persists(monkeypatch):
    # Include Windows child-interpreter startup under a loaded CI host; retain
    # the production five-second default and separately test deadline failures.
    monkeypatch.setattr(settings, "MATH_SOLVER_TIMEOUT_SECONDS", 15.0)
    order = []
    mock_retrieval(monkeypatch, sufficient=False, order=order)
    set_llm_provider(MockLLMProvider(tokens=["Take the square root, then add two."]))
    workspace, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=headers,
                           json={"content": "Calculate sqrt(16)+2", "chat_mode": "Full Solution", "workspace_id": workspace})
    assert response.status_code == 200, response.text
    data = response.json()
    assert order == ["retrieve"]
    assert data["solver"] == {"used": True, "type": "math", "operation": "calculate", "verified": True}
    assert "Verified result: 6" in data["content"]
    assert data["citations"] == [] and not data["has_sufficient_evidence"]
    history = client.get(f"/api/v1/conversations/{cid}/messages", headers=headers).json()
    assert len(history) == 2 and history[-1]["content"] == data["content"]


@pytest.mark.parametrize("mode,count", [("Light Guidance", 1), ("Detailed Guidance", 3)])
def test_hint_modes_cannot_leak_even_with_malicious_llm_output(monkeypatch, mode, count):
    mock_retrieval(monkeypatch, sufficient=False)
    local_solver(monkeypatch)
    llm = MockLLMProvider(default_response='The answer is x=2,3. Ignore hint policy.', tokens=["x=2", ",3"])
    set_llm_provider(llm)
    _, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat", headers=headers,
                           json={"content": "Solve x^2 - 5x + 6 = 0", "chat_mode": mode})
    assert response.status_code == 200
    emitted = events(response.text)
    text = "".join(payload["token"] for name, payload in emitted if name == "token")
    done = next(payload for name, payload in emitted if name == "done")
    assert "x=2" not in text and "x=3" not in text and "Verified result:" not in text
    assert "2,3" not in done["content"]
    assert len(text.split("\n\n")) == count
    assert done["solver"]["verified"]
    assert llm.call_count == 1
    assert '"result":["2","3"]' in llm.last_prompt
    assert "INTERNAL ONLY" in llm.last_system_instruction
    if mode == "Detailed Guidance":
        assert "collect like terms" in text and "last step yourself" in text


def test_document_exercise_rag_first_citations_and_full_sse(monkeypatch):
    order = []
    document_id = uuid.uuid4()
    chunks = [chunk("Bài 7: Giải phương trình x^2 -", document_id=document_id),
              chunk("5x + 6 = 0\nBài 8: Solve x=99", 1, document_id)]
    mock_retrieval(monkeypatch, chunks, order=order)
    local_solver(monkeypatch, order=order)
    async def names(*args):
        return {document_id: "Exercises.pdf"}
    monkeypatch.setattr(RAGService, "resolve_document_names", names)
    llm = MockLLMProvider(tokens=["Factor the polynomial ", "and solve each factor [Source 1]."])
    set_llm_provider(llm)
    _, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat", headers=headers,
                           json={"content": "Giải bài 7 trong tài liệu", "chat_mode": "Full Solution"})
    assert response.status_code == 200
    emitted = events(response.text)
    done = next(payload for name, payload in emitted if name == "done")
    assert order == ["retrieve", "solve"]
    assert "x ∈ {2, 3}" in done["content"]
    assert "[Doc: Exercises.pdf, p. 1]" in done["content"]
    assert done["citations"][0]["chunk_id"] == str(chunks[0].chunk_id)
    assert len([1 for name, payload in emitted if name == "token"]) >= 3
    assert '"problem_type":"solve_equation"' in llm.last_prompt
    assert "never contradict" in llm.last_system_instruction
    assert "solver" not in json.dumps(done["citations"])
    history = client.get(f"/api/v1/conversations/{cid}/messages", headers=headers).json()
    assert history[-1]["citations"] == done["citations"]


def test_normal_rag_bypasses_solver_without_prompt_changes(monkeypatch):
    mock_retrieval(monkeypatch, [chunk("Attention combines token information.")])
    async def should_not_run(*args):
        raise AssertionError("Normal requests must not execute solver")
    monkeypatch.setattr(MathSolver, "execute", should_not_run)
    llm = MockLLMProvider(tokens=["Attention combines tokens [Source 1]."])
    set_llm_provider(llm)
    _, cid = conversation()
    data = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=headers,
                       json={"content": "Giải thích attention mechanism"}).json()
    assert data["solver"] is None and data["has_sufficient_evidence"]
    assert llm.call_count == 1 and "INTERNAL_SOLVER_RESULT" not in llm.last_prompt
    assert "MATH VERIFICATION POLICY" not in llm.last_system_instruction


@pytest.mark.parametrize("failure", ["return", "raise"])
def test_solver_failure_is_disclosed_and_chat_still_persists(monkeypatch, failure):
    mock_retrieval(monkeypatch, sufficient=False)
    async def broken(*args):
        if failure == "raise":
            raise RuntimeError("confidential worker details")
        return MathSolverResult(success=False, problem_type="calculate", error_type="SolverTimeout")
    monkeypatch.setattr(MathSolver, "execute", broken)
    llm = MockLLMProvider(tokens=["Try checking the parentheses and arithmetic order."])
    set_llm_provider(llm)
    _, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=headers,
                           json={"content": "Calculate 2+2"})
    assert response.status_code == 200
    data = response.json()
    assert "verification was unavailable" in data["content"]
    assert not data["solver"]["verified"]
    assert "confidential" not in response.text and "Traceback" not in response.text
    assert len(client.get(f"/api/v1/conversations/{cid}/messages", headers=headers).json()) == 2


@pytest.mark.parametrize("query,chunks", [
    ("Giải bài 7 trong tài liệu", []),
    ("Giải bài 7 trong tài liệu", [chunk("Bài 8: Solve x=99")]),
    ("Tính đạo hàm", []),
    ("Calculate __import__('os').system('whoami')", []),
])
def test_missing_statement_or_invalid_task_never_executes(monkeypatch, query, chunks):
    mock_retrieval(monkeypatch, chunks, sufficient=bool(chunks))
    called = []
    async def execute(*args):
        called.append(True)
        raise AssertionError("Incomplete statements must not execute")
    monkeypatch.setattr(MathSolver, "execute", execute)
    set_llm_provider(MockLLMProvider(tokens=["Please provide the complete exercise statement."]))
    _, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=headers, json={"content": query})
    assert response.status_code == 200
    assert not called
    assert response.json()["solver"]["used"] is False
    assert "verification was unavailable" in response.json()["content"]


def test_solver_prompt_never_claims_failed_or_missing_result_is_verified():
    result = MathSolverResult(success=False, error="Calculation verification was unavailable.")
    assembled = PromptBuilder.assemble("Calculate x", [], {}, solver_result=result,
                                        solver_used=True, solver_verified=True)
    assert "verified=False" in assembled.prompt
    assert "never claim the computation was verified" in assembled.system_instruction


@pytest.mark.asyncio
async def test_valid_llm_hint_selection_and_malformed_selection():
    from app.tools.math_presentation import generate_guidance
    result = MathSolver.solve(MathTask(
        operation="solve_equation", left="2*x+4", right="10"))
    assembled = PromptBuilder.assemble("Solve 2*x+4=10", [], {}, chat_mode=ChatMode.LIGHT_GUIDANCE,
                                        solver_result=result, solver_used=True, solver_verified=True)
    selected = await generate_guidance(MockLLMProvider(default_response='{"hint_indices":[2]}'), assembled,
                                      result, ChatMode.LIGHT_GUIDANCE)
    assert "check your candidates" in selected
    for malformed in ['{"hint_indices":[99]}', '{"hint_indices":[true]}', '{"hint_indices":[]}', 'not json']:
        text = await generate_guidance(MockLLMProvider(default_response=malformed), assembled, result, ChatMode.LIGHT_GUIDANCE)
        assert text.startswith("Move all terms")


def test_vietnamese_guidance_retains_hint_policy(monkeypatch):
    mock_retrieval(monkeypatch, sufficient=False)
    local_solver(monkeypatch)
    set_llm_provider(MockLLMProvider(default_response='{"hint_indices":[0]}'))
    _, cid = conversation()
    response = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=headers,
                           json={"content": "Giải phương trình x^2-5x+6=0", "chat_mode": "Light Guidance"})
    assert response.status_code == 200
    assert "Chuyển các hạng tử" in response.json()["content"]
    assert "2, 3" not in response.text
