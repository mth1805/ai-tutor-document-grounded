"""Focused tests for fixture-grounded Phase 10 Answer Evaluation."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService
from benchmarks.answer_eval import (evaluate_answer_citations, load_answer_dataset,
                                    load_captured_answers, validate_answer_record)
from benchmarks.capture_answer_eval import capture_answer_dataset, write_outputs
from benchmarks.run_ragas_eval import (GEMINI_OPENAI_BASE_URL,
                                      GeminiRequestScheduler,
                                      create_gemini_compatible_client,
                                      metric_arguments, prepare_samples,
                                      resolve_evaluator_config)
from benchmarks.run_rag_eval import build_report
from benchmarks.request_scheduler import RequestScheduler, RetryPolicy
from app.llm.exceptions import LLMProviderError


def test_answer_dataset_has_exactly_24_verified_local_references():
    rows = load_answer_dataset()
    assert len(rows) == 24
    assert rows[0]["eval_id"] == "eval-001"
    assert rows[0]["gold_evidence_ids"] == ["fixture:machine_learning"]
    assert all(row["reference_answer"] and row["reference_source_aliases"] == row["gold_evidence_ids"]
               for row in rows)


def test_answer_record_schema_and_json_array_parsing(tmp_path):
    record = {"eval_id": "eval-001", "question": "Q", "query_type": "factual",
              "retrieved_evidence": [], "gold_evidence_ids": ["fixture:machine_learning"],
              "final_generated_answer": "A", "document_citations": [], "citation_metadata": {},
              "evidence_gate": {}, "ttft_ms": 1.0, "generation_latency_ms": 2.0,
              "total_answer_latency_ms": 3.0, "model_configuration": {"model": "gemini-3.5-flash-lite"}}
    assert validate_answer_record(record) is record
    path = tmp_path / "answers.json"
    path.write_text(json.dumps([record], ensure_ascii=False, indent=2), encoding="utf-8")
    assert load_captured_answers(path)[0]["eval_id"] == "eval-001"
    with pytest.raises(ValueError, match="missing fields"):
        validate_answer_record({"eval_id": "bad"})


def test_answer_capture_writes_single_json_array_for_ragas_and_summary(tmp_path, monkeypatch):
    import benchmarks.capture_answer_eval as capture

    records = [
        {"eval_id": "eval-001", "question": "What is AI?", "query_type": "factual",
         "generation_status": "success", "generation_error": None,
         "final_generated_answer": "AI means artificial intelligence.", "retry_count": 0,
         "retry_delays_ms": [], "transient_error_codes": [],
         "final_failure_classification": None, "retrieved_evidence": [{"alias": "fixture:ai", "content": "Artificial intelligence."}],
         "evidence_gate": {"passed": True}, "document_citations": [], "citation_metadata": {},
         "citation_metrics": {"citation_correctness": 1.0}, "ttft_ms": 15.0,
         "generation_latency_ms": 25.0, "retrieval_latency_ms": 4.0,
         "total_answer_latency_ms": 30.0, "web_search_enabled": False,
         "model_configuration": {"model": "gemini-test"}, "gold_evidence_ids": [],
         "retrieval_configuration": {"dense_top_k": 5}, "reference_answer": "R\u00e9ponse v\u00e9rifi\u00e9e",
         "ragas_contexts": []},
        {"eval_id": "eval-002", "question": "Second question", "query_type": "comparison",
         "generation_status": "generation_failed", "generation_error": {"http_status": 503},
         "final_generated_answer": "", "retry_count": 2, "retry_delays_ms": [5000, 10000],
         "transient_error_codes": [503, 503], "final_failure_classification": "server_error",
         "retrieved_evidence": [], "evidence_gate": {"passed": True}, "document_citations": [],
         "citation_metadata": {}, "citation_metrics": {"citation_correctness": None},
         "ttft_ms": None, "generation_latency_ms": None, "retrieval_latency_ms": 8.0,
         "total_answer_latency_ms": 100.0, "web_search_enabled": False,
         "model_configuration": {"model": "gemini-test"}, "gold_evidence_ids": [],
         "retrieval_configuration": {"dense_top_k": 5}, "custom_value": {"kept": True},
         "reference_answer": "R", "ragas_contexts": []},
    ]
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    capture_path = tmp_path / "captured_answer_runs.json"
    summary_path = tmp_path / "captured_answer_summary.md"

    write_outputs(records, capture_path, configured_model="gemini-test", summary_path=summary_path)

    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert isinstance(captured, list) and len(captured) == 2
    assert len({row["eval_id"] for row in captured}) == len(captured)
    assert captured == records
    assert list(captured[0])[:3] == ["eval_id", "question", "query_type"]
    assert load_captured_answers(capture_path) == records
    assert [sample["answer"] for sample in prepare_samples(captured)] == [records[0]["final_generated_answer"]]
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*_pretty.json"))
    assert "Mean generation latency (successful generations): 25.000 ms" in summary_path.read_text(encoding="utf-8")

def test_citation_correctness_and_completeness_require_fixture_provenance():
    evidence = [{"alias": "fixture:machine_learning", "chunk_id": "chunk-a", "document_id": "doc-a",
                 "page_start": 1, "page_end": 1}]
    citations = [{"chunk_id": "chunk-a", "document_id": "doc-a", "page_start": 1, "page_end": 1}]
    result = evaluate_answer_citations(citations, evidence, ["fixture:machine_learning"])
    assert result["citation_correctness"] == 1.0
    assert result["citation_completeness"] == 1.0
    wrong_source = evaluate_answer_citations([{**citations[0], "document_id": "other"}], evidence,
                                             ["fixture:machine_learning"])
    assert wrong_source["citation_correctness"] == 0.0
    assert wrong_source["citation_completeness"] == 0.0


def test_ragas_adapter_maps_captured_answer_fields_to_v04_contract():
    record = {"eval_id": "eval-001", "question": "Q", "final_generated_answer": "A",
              "reference_answer": "R", "retrieved_evidence": [{"content": "C"}],
              "citation_metadata": {}, "generation_status": "generated"}
    assert prepare_samples([record]) == [{"question": "Q", "answer": "A", "contexts": ["C"], "reference": "R"}]
    assert prepare_samples([{**record, "generation_status": "generation_failed", "final_generated_answer": ""}]) == []


def test_ragas_gemini_configuration_does_not_require_openai_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "gemini-judge-test")
    monkeypatch.setenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-gemini-key")

    config = resolve_evaluator_config()

    assert config.provider == "gemini"
    assert config.model == "gemini-judge-test"
    assert config.embedding_model == "gemini-embedding-test"
    assert config.api_key == "test-gemini-key"


def test_ragas_missing_gemini_key_has_clear_configuration_error(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required.*OPENAI_API_KEY is not used"):
        create_gemini_compatible_client(settings.GEMINI_API_KEY)


@pytest.mark.asyncio
async def test_ragas_client_uses_gemini_openai_compatible_endpoint(monkeypatch):
    import openai

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append((type(self).__name__, kwargs))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))

    class FakeAsyncOpenAI(FakeClient):
        async def close(self):
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    client = create_gemini_compatible_client("gemini-secret")

    assert isinstance(client, FakeAsyncOpenAI)
    assert created == [
        ("FakeAsyncOpenAI", {"api_key": "gemini-secret", "base_url": GEMINI_OPENAI_BASE_URL,
                             "max_retries": 0}),
    ]


def test_ragas_043_ascore_signatures_and_metric_argument_mapping():
    import inspect
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision,
                                           ContextRecall, Faithfulness)

    expected = {
        Faithfulness: ["self", "user_input", "response", "retrieved_contexts"],
        AnswerRelevancy: ["self", "user_input", "response"],
        ContextPrecision: ["self", "user_input", "reference", "retrieved_contexts"],
        ContextRecall: ["self", "user_input", "retrieved_contexts", "reference"],
    }
    for metric, names in expected.items():
        assert list(inspect.signature(metric.ascore).parameters) == names

    mapped = metric_arguments({"question": "Q", "answer": "A", "contexts": ["C"], "reference": "R"})
    assert mapped == {
        "faithfulness": {"user_input": "Q", "response": "A", "retrieved_contexts": ["C"]},
        "answer_relevance": {"user_input": "Q", "response": "A"},
        "context_precision": {"user_input": "Q", "reference": "R", "retrieved_contexts": ["C"]},
        "context_recall": {"user_input": "Q", "retrieved_contexts": ["C"], "reference": "R"},
    }


@pytest.mark.asyncio
async def test_ragas_runner_uses_supported_metric_arguments_for_one_record(monkeypatch):
    import benchmarks.run_ragas_eval as runner

    calls = {name: [] for name in ("faithfulness", "answer_relevance", "context_precision", "context_recall")}
    invocation_order = []
    allowed = {
        "faithfulness": {"user_input", "response", "retrieved_contexts"},
        "answer_relevance": {"user_input", "response"},
        "context_precision": {"user_input", "reference", "retrieved_contexts"},
        "context_recall": {"user_input", "retrieved_contexts", "reference"},
    }

    class FakeMetric:
        def __init__(self, name):
            self.name = name

        async def ascore(self, **kwargs):
            assert set(kwargs) == allowed[self.name]
            calls[self.name].append(kwargs)
            invocation_order.append(self.name)
            return SimpleNamespace(value=0.75)

    class MetricFactory:
        def __init__(self, name):
            self.name = name

        def __call__(self, **kwargs):
            return FakeMetric(self.name)

    import ragas.embeddings.base as embedding_module
    import ragas.llms as llm_module
    import ragas.metrics.collections as metrics_module

    created_in_loop = []

    class FakeAsyncClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))

        async def close(self):
            return None

    def create_client(key):
        import asyncio
        created_in_loop.append(asyncio.get_running_loop())
        return FakeAsyncClient()

    monkeypatch.setattr(runner, "create_gemini_compatible_client", create_client)
    monkeypatch.setattr(llm_module, "llm_factory", lambda model, client: (model, client))
    monkeypatch.setattr(embedding_module, "embedding_factory", lambda provider, model, client: (provider, model, client))
    for cls_name in ("Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall"):
        monkeypatch.setattr(metrics_module, cls_name, MetricFactory({
            "Faithfulness": "faithfulness", "AnswerRelevancy": "answer_relevance",
            "ContextPrecision": "context_precision", "ContextRecall": "context_recall",
        }[cls_name]))

    result = await runner.score_records(
        [{"question": "Q", "answer": "A", "contexts": ["C"], "reference": "R"}],
        "gemini-test", "test-key",
    )

    assert all(len(metric_calls) == 1 for metric_calls in calls.values())
    assert "reference" not in calls["faithfulness"][0]
    assert "reference" not in calls["answer_relevance"][0]
    assert "reference" in calls["context_precision"][0]
    assert "reference" in calls["context_recall"][0]
    assert all(item["mean"] == 0.75 for item in result["metrics"].values())
    assert result["request_observability"]["request_count"] == 0
    assert invocation_order == ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    import asyncio
    assert created_in_loop == [asyncio.get_running_loop()]


@pytest.mark.asyncio
async def test_ragas_metrics_instantiate_with_gemini_configured_async_client_without_network():
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision,
                                           ContextRecall, Faithfulness)

    client = AsyncOpenAI(api_key="test-key", base_url=GEMINI_OPENAI_BASE_URL)
    try:
        llm = llm_factory("gemini-test", client=client)
        embeddings = embedding_factory("openai", model="gemini-embedding-2-preview", client=client)
        assert embeddings.is_async is True
        metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings),
                   ContextPrecision(llm=llm), ContextRecall(llm=llm)]
        assert len(metrics) == 4
    finally:
        await client.close()


class MockGeminiHTTPError(Exception):
    def __init__(self, status_code, message="", body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def make_fake_clock():
    state = {"time": 0.0, "waits": []}

    async def sleep(seconds):
        state["waits"].append(seconds)
        state["time"] += seconds

    def monotonic():
        return state["time"]

    return state, sleep, monotonic


@pytest.mark.asyncio
async def test_ragas_scheduler_spaces_requests_sequentially():
    state, sleep, monotonic = make_fake_clock()
    scheduler = GeminiRequestScheduler(1.0, 0, sleep=sleep, monotonic=monotonic, jitter=lambda *_: 0)
    active = {"count": 0, "max": 0}

    async def request():
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        state["time"] += 0.25
        active["count"] -= 1

    call = scheduler.wrap(request)
    await asyncio.gather(call(), call(), call())
    assert active["max"] == 1
    assert state["waits"] == pytest.approx([0.75, 0.75])
    assert scheduler.request_count == 3
    assert scheduler.summary()["concurrency"] == 1


@pytest.mark.asyncio
async def test_ragas_scheduler_retries_429_then_succeeds():
    state, sleep, monotonic = make_fake_clock()
    scheduler = GeminiRequestScheduler(0, 2, 2, 10, sleep=sleep, monotonic=monotonic, jitter=lambda *_: 0)
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MockGeminiHTTPError(429, "RESOURCE_EXHAUSTED per-minute")
        return "ok"

    assert await scheduler.wrap(request)() == "ok"
    assert attempts == scheduler.request_count == 2
    assert scheduler.retry_count == 1
    assert state["waits"] == [2]
    assert scheduler.requests[0]["quota_classification"] == "rpm_quota"


@pytest.mark.asyncio
async def test_ragas_scheduler_honors_explicit_retry_delay():
    state, sleep, monotonic = make_fake_clock()
    scheduler = GeminiRequestScheduler(0, 1, 2, 10, sleep=sleep, monotonic=monotonic, jitter=lambda *_: 0)
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MockGeminiHTTPError(429, "limited", headers={"retry-after": "9"})
        return "ok"

    assert await scheduler.wrap(request)() == "ok"
    assert state["waits"] == [9]
    assert scheduler.requests[0]["total_wait_ms"] == 9000


@pytest.mark.asyncio
async def test_ragas_scheduler_stops_immediately_for_daily_quota():
    scheduler = GeminiRequestScheduler(0, 6, sleep=lambda _: asyncio.sleep(0), jitter=lambda *_: 0)
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        raise MockGeminiHTTPError(429, "RESOURCE_EXHAUSTED RequestsPerDay quota")

    with pytest.raises(RuntimeError, match="daily quota exhausted"):
        await scheduler.wrap(request)()
    assert attempts == scheduler.request_count == 1
    assert scheduler.retry_count == 0
    assert scheduler.quota_classification == "daily_quota"


@pytest.mark.asyncio
async def test_ragas_scheduler_retries_503_and_does_not_retry_400_or_403():
    state, sleep, monotonic = make_fake_clock()
    scheduler = GeminiRequestScheduler(0, 1, 3, 9, sleep=sleep, monotonic=monotonic, jitter=lambda *_: 0)
    attempts = 0

    async def unavailable_then_ok():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MockGeminiHTTPError(503, "UNAVAILABLE")
        return "ok"

    assert await scheduler.wrap(unavailable_then_ok)() == "ok"
    assert state["waits"] == [3]
    for status in (400, 403):
        client_scheduler = GeminiRequestScheduler(0, 6, sleep=sleep, monotonic=monotonic)
        calls = 0

        async def denied():
            nonlocal calls
            calls += 1
            raise MockGeminiHTTPError(status, "client error")

        with pytest.raises(MockGeminiHTTPError):
            await client_scheduler.wrap(denied)()
        assert calls == client_scheduler.request_count == 1
        assert client_scheduler.retry_count == 0


def test_ragas_skips_failed_or_insufficient_answer_records_and_rejects_duplicate_ids():
    records = [
        {"eval_id": "ok-1", "question": "Q", "final_generated_answer": "A",
         "generation_status": "success", "ragas_contexts": ["C"], "reference_answer": "R"},
        {"eval_id": "skip-1", "question": "Q2", "final_generated_answer": "",
         "generation_status": "skipped_insufficient_evidence", "ragas_contexts": [], "reference_answer": "R2"},
    ]
    assert len(prepare_samples(records)) == 1
    with pytest.raises(ValueError, match="duplicate eval_id"):
        prepare_samples([records[0], dict(records[0])])


@pytest.mark.asyncio
async def test_answer_relevancy_ascore_uses_async_gemini_compatible_embeddings(monkeypatch):
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy

    client = AsyncOpenAI(api_key="test-key", base_url=GEMINI_OPENAI_BASE_URL)
    try:
        llm = llm_factory("gemini-test", client=client)
        async def fake_agenerate(self, prompt, output_type):
            return SimpleNamespace(question="What is the answer?", noncommittal=False)

        monkeypatch.setattr(type(llm), "agenerate", fake_agenerate)

        class FakeEmbeddingsEndpoint:
            async def create(self, input, model):
                items = input if isinstance(input, list) else [input]
                rows = [SimpleNamespace(embedding=[1.0, 0.0]) for _ in items]
                return SimpleNamespace(data=rows)

        embedding_client = SimpleNamespace(embeddings=FakeEmbeddingsEndpoint())
        embeddings = embedding_factory("openai", model="gemini-embedding-2-preview",
                                       client=embedding_client)
        assert embeddings.is_async is True

        metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
        result = await metric.ascore(user_input="What is the answer?", response="A grounded answer.")
        assert result.value == pytest.approx(1.0)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_capture_uses_retrieval_and_emits_answer_record_schema(monkeypatch):
    class MockLLM:
        model_name = "gemini-3.5-flash-lite"
        thinking_level = "medium"
        max_output_tokens = 2048

        async def generate_stream(self, prompt, system_instruction=None):
            assert "<DOCUMENT_EVIDENCE>" in prompt
            yield "Sample grounded answer [Source 1]."

    records = await capture_answer_dataset(load_answer_dataset()[:1], MockEmbeddingProvider(),
                                            MockRerankerProvider(), MockLLM())
    assert len(records) == 1
    row = records[0]
    assert row["eval_id"] == "eval-001"
    assert row["model_configuration"]["thinking_level"] == "medium"
    assert row["web_search_enabled"] is False
    assert row["final_generated_answer"].startswith("Sample grounded answer")
    assert "citation_correctness" in row["citation_metrics"]
    assert row["total_answer_latency_ms"] >= row["retrieval_latency_ms"]


@pytest.mark.asyncio
async def test_capture_retries_sequentially_and_writes_one_final_record_per_eval_id():
    import asyncio

    class RetryOnceLLM:
        model_name = "gemini-3.5-flash-lite"
        thinking_level = "medium"
        max_output_tokens = 2048

        def __init__(self):
            self.calls = 0
            self.active = 0
            self.max_active = 0

        async def generate_stream(self, prompt, system_instruction=None):
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0)
                if self.calls == 1:
                    raise LLMProviderError("rate limited", details={
                        "exception_class": "ClientError", "http_status": 429,
                        "provider_status": "RESOURCE_EXHAUSTED", "provider_message": "retry",
                        "classification": "quota", "retryable": True,
                        "retry_delay_seconds": 0,
                    })
                yield "Grounded answer [Source 1]."
            finally:
                self.active -= 1

    llm = RetryOnceLLM()
    dataset = load_answer_dataset()[:2]
    records = await capture_answer_dataset(
        dataset, MockEmbeddingProvider(), MockRerankerProvider(), llm,
        retry_policy=RetryPolicy(1, 0, 0, 0), request_scheduler=RequestScheduler(0),
    )
    assert [record["eval_id"] for record in records] == [row["eval_id"] for row in dataset]
    assert len({record["eval_id"] for record in records}) == len(records) == 2
    assert llm.max_active == 1
    assert records[0]["generation_status"] == "success"
    assert records[0]["retry_count"] == 1
    assert records[0]["final_generated_answer"]
    assert records[1]["generation_status"] == "success"
    assert records[1]["retry_count"] == 0


def test_retrieval_report_prefers_capture_configuration_and_records_old_discrepancy():
    actual = {"dense_top_k": 20, "lexical_top_k": 20, "candidate_pool_size": 20,
              "rerank_top_k": 10, "routing_mode": "always_quality"}
    report = build_report([], {"q": {"retrieval_configuration": actual}},
                          {"dense_top_k": 25}, {"dense_top_k": 25, "retrieval_routing_mode": "adaptive"})
    assert report["metadata"]["configuration"] == actual
    assert report["metadata"]["configuration_reconciliation"]["previous_reported_configuration"]["dense_top_k"] == 25
