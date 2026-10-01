# Retrieval-Augmented Generation (RAG) Engineering Rules

## 1. Core Grounding Hierarchy
1. **Document-First Grounding:** Uploaded documents within the active workspace are the **primary and authoritative** source of truth.
2. **Web Search as Strict Fallback:** Web search may only be executed if and only if the relevance gate confirms that uploaded document evidence is absent or insufficient for the user query.
3. **No Web Override:** Web search evidence must never overwrite, dilute, or supersede valid document evidence when documents provide the answer.
4. **Source Distinction:** Responses must clearly delineate whether information originated from an uploaded document or a web search result.

## 2. RAG Pipeline Stages & Flow Control
Every retrieval request must strictly follow this sequential pipeline:
```
User Query
   │
   ▼
[1] Hybrid Retrieval
    ├── Dense Retrieval (BGE-M3 embeddings <=> cosine distance in pgvector)
    └── Lexical Retrieval (BM25 / PostgreSQL full-text search)
   │
   ▼
[2] Reciprocal Rank Fusion (RRF)
    └── Merge dense + lexical rankings into top-K candidate pool (e.g., K=20..30)
   │
   ▼
[3] Cross-Encoder Reranking
    └── Score (query, chunk_text) pairs -> sort by rerank score -> select top-N (e.g., N=3..5)
   │
   ▼
[4] Relevance Gating
    ├── Check top rerank scores against minimum threshold (e.g., score >= threshold)
    ├── If SUFFICIENT: Pass document chunks to Generation (Document Mode)
    └── If INSUFFICIENT: Trigger Web Search Fallback -> Pass web snippets to Generation (Web Mode)
   │
   ▼
[5] Grounded Generation & Citation Attribution
    └── Stream response with verifiable citations
```

## 3. Relevance Gating Rules
- **Explicit Thresholding:** Evaluate top reranker scores against an empirically calibrated relevance threshold.
- **Context Sufficiency Check:** If no document chunks exceed the threshold or the workspace contains no indexed documents, flag context as insufficient.
- **Fail-Safe Fallback:** When falling back to web search, log the reason (e.g., `DOC_CONTEXT_INSUFFICIENT`) and notify the user via a UI status badge that external web sources are being consulted.
- **Never Synthesize from Memory Alone:** If both document retrieval and web search yield insufficient evidence, the model must explicitly state: *"I could not find sufficient information in your uploaded documents or web sources to answer this question."*

## 4. Citation Rules & Hallucination Defense
- **Real Source Anchoring:** Every inline citation must map directly to a verified chunk in the context payload:
  - Document Citation: `[Doc: {file_name}, p. {page_number}]` with an internal reference to `chunk_id`.
  - Web Citation: `[Web: {domain_or_title}]({source_url})`.
- **Zero Fabricated Citations:** Never allow the LLM to invent document names, page numbers, or external URLs.
- **Payload Verification:** Before emitting citations over the stream or in the final payload, the backend must validate all cited `chunk_id`s against the retrieved candidate set. Invalid or missing chunk IDs must be stripped or flagged.

## 5. Architectural Separation of Concerns
- **Modular Components:** The following must exist as independent, testable modules:
  - `QueryProcessor`: Query cleaning, expansion, or hyde (if applicable).
  - `DenseRetriever`: Embedding query and querying pgvector.
  - `LexicalRetriever`: BM25 index query or PostgreSQL tsvector search.
  - `RRFMerger`: Reciprocal Rank Fusion logic.
  - `Reranker`: Cross-Encoder inference and score sorting.
  - `RelevanceGate`: Threshold evaluator and fallback router.
  - `PromptBuilder`: Context injection, system instructions, and delimiter escaping.
  - `Generator`: Gemini API streaming client.
- **No Monolithic Pipelines:** Never combine retrieval, reranking, and generation into a single untestable function.

## 6. Edge Case & Failure Handling
- **Empty Workspace:** If the workspace has zero uploaded documents, gracefully skip document retrieval and proceed to fallback or prompt the user to upload materials.
- **Unindexed / In-Flight Documents:** If documents are still processing (`Queued` or `Processing`), inform the user that recent uploads are not yet searchable.
- **Parser Failures:** Chunks with empty or corrupted text must be excluded from retrieval candidates.
