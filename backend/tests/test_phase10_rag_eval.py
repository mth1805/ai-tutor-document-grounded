"""Focused tests for deterministic Phase 10 evaluation helpers."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.web_search.mock_provider import MockWebSearchProvider
from benchmarks.citation_eval import evaluate_citations
from benchmarks.evidence_gate_eval import confusion_matrix, web_routing_metrics
from benchmarks.metrics import aggregate_retrieval, compare_reranking, latency_summary, retrieval_metrics
from benchmarks.run_rag_eval import build_report, markdown_report, markdown_summary, read_capture_json, read_jsonl
from benchmarks.capture_rag_eval import (FIXTURE_ALIAS_TO_INDEX, capture_dataset, load_dataset,
                                         write_capture)
from benchmarks.run_ragas_eval import prepare_samples
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService


def test_dataset_has_reviewable_categories_and_loads_jsonl():
    path = Path(__file__).parents[1] / "benchmarks" / "eval_dataset.jsonl"
    dataset = read_jsonl(path)
    assert len(dataset) == 30
    assert {row["query_type"] for row in dataset} == {
        "factual", "comparison", "multi_hop", "insufficient_document", "web_fallback"
    }
    assert all("question" in row and "should_use_web" in row for row in dataset)


def test_dataset_gold_alias_mapping_is_explicit_and_fixture_bounded():
    dataset = load_dataset()
    labeled = [row for row in dataset if row["gold_evidence_ids"]]
    assert len(labeled) == 24
    assert all(alias in FIXTURE_ALIAS_TO_INDEX
               for row in labeled for alias in row["gold_evidence_ids"])
    assert set(FIXTURE_ALIAS_TO_INDEX.values()) == set(range(10))


@pytest.mark.asyncio
async def test_capture_executes_dataset_and_serializes_results(tmp_path):
    dataset = load_dataset()
    web = MockWebSearchProvider()
    records = await capture_dataset(
        dataset, MockEmbeddingProvider(), MockRerankerProvider(), web
    )
    assert len(records) == len(dataset) == 30
    assert all(record["gold_label_status"] in {"mapped", "no_gold_evidence"}
               for record in records)
    assert all(record["rankings"]["dense_only"] for record in records)
    assert all(record["rankings"]["reranked"] for record in records)
    assert all("total_retrieval_ms" in record["timings"] for record in records)
    assert web.call_count == sum(record["used_web_fallback"] for record in records)

    output = tmp_path / "captured.json"
    write_capture(records, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert len(loaded) == len(records)
    assert len({record["eval_id"] for record in loaded}) == len(loaded)
    assert loaded[0]["eval_id"] == "eval-001"
    assert loaded[0]["retrieved_results"][0]["alias"].startswith("fixture:")


@pytest.mark.asyncio
async def test_capture_marks_unknown_gold_alias_without_mapping_it():
    row = dict(load_dataset()[0])
    row["gold_evidence_ids"] = ["production:some-uuid"]
    records = await capture_dataset(
        [row], MockEmbeddingProvider(), MockRerankerProvider(), MockWebSearchProvider()
    )
    assert records[0]["gold_label_status"] == "unmapped_alias"
    assert records[0]["unmapped_gold_aliases"] == ["production:some-uuid"]
    assert records[0]["gold_alias_to_fixture_index"] == {}


@pytest.mark.asyncio
async def test_capture_normalizes_dict_shaped_reranker_results(monkeypatch):
    async def retrieve_with_dict_results(*, workspace_id, user_id, **kwargs):
        chunk = next(
            chunk
            for chunks in _IN_MEMORY_CHUNKS.values()
            for chunk in chunks
            if chunk.workspace_id == workspace_id and chunk.user_id == user_id
        )
        return SimpleNamespace(
            results=[{
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "page_number_start": chunk.page_number_start,
                "page_number_end": chunk.page_number_end,
                "chunk_index": chunk.chunk_index,
                "dense_score": 0.81,
                "lexical_score": 0.4,
                "rrf_score": 0.03,
                "rerank_score": 0.8,
                "final_rank": 1,
                "passed_relevance_gate": True,
                "retrieval_sources": ["dense", "lexical"],
            }],
            has_sufficient_evidence=True,
            timings=SimpleNamespace(
                query_embedding_ms=1.0, dense_retrieval_ms=2.0,
                lexical_retrieval_ms=3.0, rrf_ms=0.1,
                rerank_ms=4.0, total_retrieval_ms=10.0,
            ),
        )

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve_with_dict_results)
    records = await capture_dataset(
        [load_dataset()[0]], MockEmbeddingProvider(), MockRerankerProvider(),
        MockWebSearchProvider(),
    )

    record = records[0]
    assert record["rankings"]["reranked"] == ["fixture:artificial_intelligence"]
    assert record["retrieved_results"][0]["rank"] == 1
    assert record["retrieved_results"][0]["rerank_score"] == 0.8
    assert record["retrieved_results"][0]["passed_relevance_gate"] is True
    assert record["evidence_sufficient"] is True
    assert record["used_web_fallback"] is False


def test_retrieval_metrics_and_unlabeled_queries():
    metrics = retrieval_metrics(["a", "b", "c", "d", "e"], ["b", "z"])
    assert metrics == {"recall@1": 0.0, "recall@5": 1.0, "recall@10": 1.0,
                       "precision@5": 0.2, "mrr@10": 0.5}
    assert retrieval_metrics([], [])["recall@1"] is None
    assert aggregate_retrieval([metrics])["mrr@10"] == 0.5


def test_reranker_before_after_comparison():
    result = compare_reranking(["noise", "gold"], ["gold", "noise"], ["gold"])
    assert result["before"]["mrr@10"] == 0.5
    assert result["after"]["mrr@10"] == 1.0


def test_evidence_gate_confusion_and_web_routing():
    gate_rows = [
        {"gold": True, "pred": True}, {"gold": True, "pred": False},
        {"gold": False, "pred": False}, {"gold": False, "pred": True},
    ]
    matrix = confusion_matrix(gate_rows, gold_key="gold", prediction_key="pred")
    assert (matrix["true_positives"], matrix["true_negatives"],
            matrix["false_positives"], matrix["false_negatives"]) == (1, 1, 1, 1)
    assert matrix["sufficient_precision"] == 0.5
    web = web_routing_metrics([
        {"should_use_web": True, "used_web_fallback": True},
        {"should_use_web": False, "used_web_fallback": False},
        {"should_use_web": False, "used_web_fallback": True},
    ])
    assert web["correct_web_triggers"] == 1
    assert web["false_web_triggers"] == 1
    assert web["routing_accuracy"] == 2 / 3


def test_citation_validation_dedup_and_completeness():
    document = {"document_id": "doc-1", "page_start": 2, "page_end": 4}
    citation = {"document_id": "doc-1", "page_start": 3, "page_end": 3, "fact_ids": ["fact-a"]}
    result = evaluate_citations(
        [citation, citation, {"url": "https://example.test/a", "title": "A", "domain": "example.test"}],
        [document], ["https://example.test/a"], ["fact-a", "fact-b"],
    )
    assert result["citation_count"] == 2
    assert result["correct_count"] == 2
    assert result["citation_correctness"] == 1.0
    assert result["citation_completeness"] == 0.5


def test_latency_aggregation_and_report_serialization():
    assert latency_summary([1, 2, 3, 4, 5])["p95"] == 5
    dataset = [{"id": "q1", "gold_evidence_ids": ["g"], "should_use_web": False,
                "gold_documents": []}]
    records = {"q1": {"rankings": {"dense_only": ["g"], "hybrid_rrf": ["g"],
                                    "reranked": ["g"]},
                       "evidence_sufficient": True, "used_web_fallback": False,
                       "timings": {"rerank_ms": 2.0, "total_retrieval_ms": 5.0}}}
    report = build_report(dataset, records, {"relevance_threshold": 0.35})
    json.loads(json.dumps(report))
    markdown = markdown_report(report)
    assert "Dense only" in markdown
    assert "Recall@5" in markdown


def test_capture_outputs_write_single_json_array_without_obsolete_copies(tmp_path):
    records = [
        {"eval_id": "eval-001", "query_id": "eval-001", "question": "\u00bfQu\u00e9 es AI?",
         "query_type": "factual", "rankings": {"dense_only": ["fixture:ai"]},
         "retrieved_results": [{"alias": "fixture:ai", "rerank_score": 0.91}],
         "retrieval_configuration": {"dense_top_k": 10},
         "timings": {"total_retrieval_ms": 12.5}, "extension": {"unchanged": True}},
        {"eval_id": "eval-002", "query_id": "eval-002", "question": "Second query",
         "query_type": "comparison", "rankings": {"dense_only": []},
         "retrieved_results": [], "retrieval_configuration": {"dense_top_k": 10},
         "timings": {"total_retrieval_ms": 0.0}},
    ]
    output = tmp_path / "captured_runs.json"
    write_capture(records, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert len({row["eval_id"] for row in loaded}) == 2
    assert loaded == records
    assert list(loaded[0])[:4] == ["eval_id", "query_id", "question", "query_type"]
    assert read_capture_json(output) == records
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*_pretty.json"))



def test_retrieval_markdown_summary_uses_actual_report_values():
    report = {"metadata": {"dataset_version": "1.0", "query_count": 30,
                           "labeled_query_count": 24, "generated_at": "2026-10-04T00:00:00Z",
                           "configuration": {"dense_top_k": 20, "lexical_top_k": 15,
                                              "rrf_k": 60, "candidate_pool_size": 25,
                                              "rerank_top_k": 5, "routing_mode": "adaptive",
                                              "reranker_model": "local-cross-encoder",
                                              "relevance_threshold": 0.35,
                                              "min_answerable_rerank_score": 0.55}},
              "baselines": {"Dense only": {"recall@1": 0.5, "recall@5": 0.75,
                                            "recall@10": 1.0, "precision@5": 0.2, "mrr@10": 0.6}},
              "latency_ms": {"query_embedding_ms": {"mean": 1, "median": 0.9, "p95": 2},
                             "total_retrieval_ms": {"mean": 10, "median": 9, "p95": 20}}}
    summary = markdown_summary(report)
    assert "Queries: 30" in summary
    assert "Labeled queries: 24" in summary
    assert "Dense Top-K: 20" in summary
    assert "| Dense only | 0.500 | 0.750 | 1.000 | 0.200 | 0.600 |" in summary
    assert "| Query embedding | 1 | 0.9 | 2 |" in summary


def test_ragas_input_adapter_is_deterministic_and_validates_contract():
    record = {"question": "What?", "answer": "This.", "contexts": ["Evidence."],
              "reference": "Expected."}
    assert prepare_samples([record]) == [record]
    try:
        prepare_samples([{"question": "incomplete"}])
    except ValueError as exc:
        assert "answer" in str(exc)
    else:
        raise AssertionError("missing answer record fields should fail validation")
