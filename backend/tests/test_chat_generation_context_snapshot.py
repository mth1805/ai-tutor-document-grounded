import copy
import uuid
from types import SimpleNamespace

import pytest

from app.services import chat_generation_context as context
from app.services.document_summary_service import DocumentSummaryService, _SummarySegment


def _snapshot(path="document", **overrides):
    source_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    payload = {
        "original_request": "Explain cell respiration",
        "normalized_mode": "Detailed Guidance",
        "workspace_id": str(uuid.uuid4()),
        "selected_document_ids": [str(uuid.uuid4())],
        "history": [{"role": "user", "content": "What is ATP?"}],
        "resolved_document_references": {"document_id": source_id, "resolution": "explicit"},
        "generation_path": path,
        "system_instruction": "Use the captured context.",
        "prompt_plan": [{"stage": "final", "prompt": "Prompt [Source 1]", "max_output_tokens": 8192}],
        "evidence": [{
            "source_index": 1, "document_id": source_id, "document_name": "Biology.pdf",
            "chunk_id": chunk_id, "page_start": 2, "page_end": 3,
            "content": "Captured evidence", "retrieval_rank": 1, "dense_score": 0.8,
            "lexical_score": 0.7, "rrf_score": 0.03, "rerank_score": 0.9,
        }],
        "routing_decisions": {"route": "QUALITY", "evidence_sufficient": True},
        "citation_sources": [{"source_index": 1, "document_id": source_id,
                              "chunk_id": chunk_id, "document_name": "Biology.pdf",
                              "page_start": 2, "page_end": 3}],
        "web_results": [],
        "solver_output": None,
        "whole_document_source_chunks": [],
        "whole_document_summary_plan": None,
    }
    payload.update(overrides)
    return payload


def _summary_fixture(chunk_count=1, *, unique_source=False):
    workspace_id = uuid.uuid4()
    document = SimpleNamespace(id=uuid.uuid4(), original_filename="Snapshot summary.pdf")
    provider = SimpleNamespace(model_name="fixture-model", thinking_level="low")
    chunks = []
    for index in range(chunk_count):
        content = (
            f"UNIQUE_SUMMARY_SOURCE_{index} " + "explanatory text and evidence. " * 12
            if unique_source else "word " * 500
        )
        chunks.append(SimpleNamespace(
            id=uuid.uuid4(), document_id=document.id, workspace_id=workspace_id,
            user_id=uuid.uuid4(), chunk_index=index, page_number_start=index + 1,
            page_number_end=index + 1, content=content,
            token_count=500,
        ))
    query = "Summarize the entire document"
    fields = context.build_whole_document_summary_snapshot_fields(chunks, document, query, provider)
    payload = _snapshot(
        "whole_document_summary", original_request=query, workspace_id=str(workspace_id),
        selected_document_ids=[str(document.id)], system_instruction=fields["system_instruction"],
        prompt_plan=fields["prompt_plan"], evidence=fields["evidence"],
        citation_sources=fields["citation_sources"],
        whole_document_source_chunks=fields["whole_document_source_chunks"],
        whole_document_summary_plan=fields["whole_document_summary_plan"],
    )
    payload["resolved_document_references"] = {"document_id": str(document.id), "resolution": "explicit"}
    return chunks, document, provider, fields, payload


def test_canonical_snapshot_hash_is_deterministic_and_key_order_independent():
    first = _snapshot()
    reordered = dict(reversed(list(first.items())))

    a = context.prepare_snapshot(first)
    b = context.prepare_snapshot(reordered)

    assert a.context_hash == b.context_hash
    assert a.canonical_bytes == b.canonical_bytes
    assert context.canonical_json_bytes(a.payload) == context.canonical_json_bytes(b.payload)


def test_snapshot_hash_mismatch_is_rejected_before_use():
    prepared = context.prepare_snapshot(_snapshot())

    context.verify_snapshot(prepared.payload, prepared.context_hash)
    changed = copy.deepcopy(prepared.payload)
    changed["prompt_plan"][0]["prompt"] += " changed"
    with pytest.raises(context.SnapshotPreparationError, match="hash mismatch"):
        context.verify_snapshot(changed, prepared.context_hash)


def test_citation_metadata_must_match_authorized_snapshot_source():
    payload = _snapshot()
    payload["citation_sources"][0]["page_start"] = 999

    with pytest.raises(context.SnapshotPreparationError, match="Citation metadata differs"):
        context.prepare_snapshot(payload)


def test_snapshot_size_limit_fails_before_acceptance(monkeypatch):
    monkeypatch.setattr(context, "MAX_CANONICAL_SNAPSHOT_BYTES", 256)

    with pytest.raises(context.SnapshotPreparationError, match="snapshot limit"):
        context.prepare_snapshot(_snapshot())


@pytest.mark.parametrize("path", ["document", "web_fallback"])
def test_generation_paths_keep_exact_prompt_evidence_and_citation_order(path):
    payload = _snapshot(
        path,
        prompt_plan=[
            {"stage": "map_1", "prompt": "first captured prompt", "max_output_tokens": 512},
            {"stage": "reduce_final", "prompt": "second captured prompt", "max_output_tokens": 2048},
        ] if path == "whole_document_summary" else [
            {"stage": "final", "prompt": "captured prompt", "max_output_tokens": 8192},
        ],
        web_results=[{"title": "Reference", "url": "https://example.test", "content": "Captured web context"}]
        if path == "web_fallback" else [],
        routing_decisions={"route": path, "evidence_sufficient": path == "document"},
    )

    prepared = context.prepare_snapshot(payload)

    assert prepared.payload["generation_path"] == path
    assert [item["prompt"] for item in prepared.payload["prompt_plan"]] == [
        item["prompt"] for item in payload["prompt_plan"]
    ]
    assert [item["source_index"] for item in prepared.payload["evidence"]] == [1]
    assert [item["source_index"] for item in prepared.payload["citation_sources"]] == [1]
    if path == "web_fallback":
        assert prepared.payload["web_results"][0]["content"] == "Captured web context"


def test_solver_result_is_preserved_as_a_prompt_input():
    payload = _snapshot(solver_output={"success": True, "result": "42", "source_chunk_ids": []})

    prepared = context.prepare_snapshot(payload)

    assert prepared.payload["solver_output"] == payload["solver_output"]


def test_whole_document_summary_keeps_ordered_sources_and_exact_call_plan():
    chunks, document, _, fields, payload = _summary_fixture(24, unique_source=True)
    prepared = context.prepare_snapshot(payload)
    encoded = context.canonical_json_bytes(prepared.payload).decode("utf-8")
    repeated_plan = context.build_whole_document_summary_snapshot_fields(
        chunks, document, payload["original_request"], SimpleNamespace(model_name="fixture-model", thinking_level="low"),
    )["whole_document_summary_plan"]

    assert [item["source_index"] for item in prepared.payload["whole_document_source_chunks"]] == list(range(1, 25))
    assert [call["source_indices"] for call in prepared.payload["prompt_plan"]] == fields["whole_document_summary_plan"]["map_groups"]
    assert all(call["prompt"] is None for call in prepared.payload["prompt_plan"])
    assert all(encoded.count(f"UNIQUE_SUMMARY_SOURCE_{index} ") == 1 for index in range(24))
    assert sum(chunk.token_count for chunk in chunks) == fields["whole_document_summary_plan"]["estimates"]["persisted_source_tokens"]
    assert repeated_plan["plan_hash"] == fields["whole_document_summary_plan"]["plan_hash"]


def test_snapshot_map_prompt_reconstructs_exactly_from_once_stored_source():
    chunks, document, _, fields, payload = _summary_fixture(22)
    prepared = context.prepare_snapshot(payload)
    sources = prepared.payload["whole_document_source_chunks"]
    for call in prepared.payload["prompt_plan"]:
        live_group = [chunks[index - 1] for index in call["source_indices"]]
        source_by_chunk = {
            chunk.id: SimpleNamespace(
                source_index=index, document_id=document.id, document_name=document.original_filename,
                chunk_id=chunk.id, page_start=chunk.page_number_start,
                page_end=chunk.page_number_end, content=chunk.content.strip(),
            )
            for index, chunk in enumerate(chunks, start=1)
        }
        live_prompt = DocumentSummaryService._document_prompt(
            payload["original_request"], document, live_group, call["group_index"],
            source_by_chunk, final=call["stage"] == "direct",
        )
        reconstructed = DocumentSummaryService.reconstruct_document_prompt(
            payload["original_request"], document.original_filename, sources,
            call["source_indices"], call["group_index"], final=call["stage"] == "direct",
        )
        assert reconstructed == live_prompt
    assert fields["whole_document_summary_plan"]["prompt_template_version"] == DocumentSummaryService.PROMPT_TEMPLATE_VERSION


def test_summary_snapshot_uses_the_live_persisted_token_adjustment():
    chunks, document, provider, fields, payload = _summary_fixture(18)
    chunks[0].token_count += 37
    fields = context.build_whole_document_summary_snapshot_fields(
        chunks, document, payload["original_request"], provider,
    )
    payload.update(fields)
    prepared = context.prepare_snapshot(payload)
    call = prepared.payload["prompt_plan"][0]
    assert call["estimated_input_tokens"] == fields["prompt_plan"][0]["estimated_input_tokens"]
    source_items = prepared.payload["whole_document_source_chunks"]
    assert max(item["token_count_adjustment"] for item in source_items) > 0


def test_recursive_reduction_prompt_reconstruction_is_deterministic(monkeypatch):
    monkeypatch.setattr(context.settings, "DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET", 4000)
    document = SimpleNamespace(original_filename="Recursive.pdf")
    query = "Summarize recursively"
    system = DocumentSummaryService._system_instruction(query)
    budget = int(context.settings.DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET * 0.75)
    initial = [
        _SummarySegment(content="detail " * 700, source_indices=(index,))
        for index in range(1, 13)
    ]

    def run_reduction():
        segments = initial
        levels = []
        while DocumentSummaryService._prompt_token_count(
            DocumentSummaryService._reduce_prompt(query, document, segments, final=True), system
        ) > budget:
            groups = DocumentSummaryService._pack_segments(
                segments, query, document, system, budget,
            )
            prompts = [DocumentSummaryService._reduce_prompt(query, document, group, final=False)
                       for group in groups]
            levels.append(prompts)
            segments = [
                _SummarySegment(
                    content="reduced note " * 450,
                    source_indices=tuple(sorted({i for item in group for i in item.source_indices})),
                ) for group in groups
            ]
        return levels, DocumentSummaryService._reduce_prompt(query, document, segments, final=True)

    first = run_reduction()
    second = run_reduction()
    assert len(first[0]) >= 2
    assert first == second


@pytest.mark.parametrize("source_tokens", [10_000, 100_000, 500_000])
def test_summary_snapshot_size_and_call_estimates_for_representative_fixtures(source_tokens):
    chunks, _, _, fields, payload = _summary_fixture(source_tokens // 500)
    validated = context.GenerationContextSnapshot.model_validate(payload)
    encoded = context.canonical_json_bytes(validated.model_dump(mode="json"))
    if len(encoded) > context.MAX_CANONICAL_SNAPSHOT_BYTES:
        with pytest.raises(context.SnapshotPreparationError, match="snapshot limit"):
            context.prepare_snapshot(payload)
    else:
        prepared = context.prepare_snapshot(payload)
        assert prepared.canonical_bytes == len(encoded)
    estimates = fields["whole_document_summary_plan"]["estimates"]
    assert estimates["persisted_source_tokens"] == sum(chunk.token_count for chunk in chunks)
    groups = fields["whole_document_summary_plan"]["map_groups"]
    assert estimates["map_calls"] == (0 if len(groups) == 1 else len(groups))
    print({
        "source_tokens": estimates["persisted_source_tokens"],
        "chunks": estimates["source_chunk_count"],
        "canonical_bytes": len(encoded),
        "map_calls": estimates["map_calls"],
        "reduction_calls_at_max_output": estimates["reduction_calls_at_max_output"],
        "total_calls_at_max_output": estimates["total_calls_at_max_output"],
        "sequential_timeout_seconds_at_estimated_calls": estimates["sequential_timeout_seconds_at_estimated_calls"],
    })


def test_history_limit_and_empty_prompt_plan_fail_without_truncation():
    with pytest.raises(context.SnapshotPreparationError):
        context.prepare_snapshot(_snapshot(history=[
            {"role": "user", "content": str(index)} for index in range(7)
        ]))

    with pytest.raises(context.SnapshotPreparationError):
        context.prepare_snapshot(_snapshot(prompt_plan=[]))


def test_generation_path_prompt_and_output_limits_follow_runtime_config(monkeypatch):
    configured_budget = context.settings.DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET
    _, _, _, _, summary_payload = _summary_fixture(1)
    monkeypatch.setattr(context.settings, "DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET", 1)
    with pytest.raises(context.SnapshotPreparationError, match="input budget"):
        context.prepare_snapshot(summary_payload)

    # Regular chat has no summary-specific input cap; only its configured
    # generation output ceiling applies.
    context.prepare_snapshot(_snapshot())

    monkeypatch.setattr(context.settings, "DOCUMENT_SUMMARY_INPUT_TOKEN_BUDGET", configured_budget)
    monkeypatch.setattr(context.settings, "CHAT_MAX_OUTPUT_TOKENS", 100)
    with pytest.raises(context.SnapshotPreparationError, match="output budget"):
        context.prepare_snapshot(_snapshot())


def test_web_result_count_uses_configured_provider_and_surface_caps(monkeypatch):
    monkeypatch.setattr(context.settings, "TAVILY_MAX_RESULTS", 2)
    monkeypatch.setattr(context.settings, "WEB_SEARCH_MAX_SOURCES", 4)
    with pytest.raises(context.SnapshotPreparationError, match="2-result limit"):
        context.prepare_snapshot(_snapshot(
            "web_fallback",
            web_results=[{"title": str(i), "url": f"https://example.test/{i}", "content": "x"}
                         for i in range(3)],
        ))


def test_request_fingerprint_includes_all_request_selection_fields():
    common = {"content": "Question", "mode": "Detailed Guidance", "workspace_id": uuid.uuid4(),
              "document_id": uuid.uuid4(), "active_document_id": None}

    original = context.request_fingerprint(**common)
    changed = context.request_fingerprint(**{**common, "active_document_id": uuid.uuid4()})

    assert original != changed


def test_request_fingerprint_matches_canonical_request_payload():
    identity = {"content": "Question", "mode": "Detailed Guidance", "workspace_id": None,
                "document_id": None, "active_document_id": None}
    fingerprint = context.request_fingerprint(
        content="Question", mode="Detailed Guidance", workspace_id=None,
        document_id=None, active_document_id=None,
    )

    import hashlib
    assert fingerprint == hashlib.sha256(context.canonical_json_bytes(identity)).hexdigest()


def test_actual_summary_snapshot_over_cap_is_rejected_before_acceptance(monkeypatch):
    _, _, _, _, payload = _summary_fixture(2)
    monkeypatch.setattr(context, "MAX_CANONICAL_SNAPSHOT_BYTES", 128)
    with pytest.raises(context.SnapshotPreparationError, match="snapshot limit"):
        context.prepare_snapshot(payload)
