---
name: rag-engineering
description: Guides hybrid retrieval (dense + BM25), Reciprocal Rank Fusion, Cross-Encoder reranking, relevance gating, grounded prompt engineering, and web search fallback.
---

# RAG Engineering Skill

## 1. Purpose
The `rag-engineering` skill provides comprehensive guidance for designing, implementing, and tuning the document-grounded RAG pipeline for **AI Tutor Assistant**. It ensures uploaded documents are strictly prioritized, enforces relevance gating, manages web search fallback, formats grounded prompts for Gemini, and generates verifiable citations.

## 2. When to Use
Use this skill when:
- Implementing or tuning the hybrid retrieval pipeline (Dense + BM25).
- Adjusting Reciprocal Rank Fusion (RRF) parameters.
- Integrating or tuning the Cross-Encoder reranker model.
- Configuring the relevance gate score threshold.
- Designing prompt templates for grounded answer generation.
- Implementing the web search fallback mechanism.
- Formatting citations and streaming responses to the frontend.

## 3. What to Inspect First
Before modifying RAG components, inspect:
1. `backend/app/services/retrieval/` — Current implementations of dense, lexical, and hybrid search.
2. `backend/app/services/reranker/` — Cross-Encoder model loading and scoring logic.
3. `backend/app/services/relevance/` — Threshold logic for document sufficiency vs. fallback.
4. `backend/app/services/generation/` — Gemini API client, system prompts, and citation extractors.
5. `backend/app/core/config.py` — RAG hyperparameters (e.g., `TOP_K_DENSE`, `TOP_K_LEXICAL`, `RERANK_TOP_N`, `RELEVANCE_THRESHOLD`).

## 4. End-to-End Pipeline Workflow
Every query must execute through these 7 logical phases:

```
[User Query]
     │
     ▼
[Phase 1: Query Processing]
     │ Normalize query, generate BGE-M3 query embedding
     ▼
[Phase 2: Hybrid Retrieval (Workspace Scoped)]
     ├── Dense Search: pgvector <=> cosine distance (top K=25)
     └── Lexical Search: PostgreSQL tsvector BM25 ranking (top K=25)
     │
     ▼
[Phase 3: Reciprocal Rank Fusion (RRF)]
     │ Combine ranked lists using score = 1 / (60 + rank); generate candidate pool (top K=30)
     ▼
[Phase 4: Cross-Encoder Reranking]
     │ Score (query, chunk_text) with Cross-Encoder; sort descending; take top N=5
     ▼
[Phase 5: Relevance Gating]
     │ Evaluate top rerank score >= RELEVANCE_THRESHOLD
     ├── [SUFFICIENT EVIDENCE]
     │       └── Assemble context from top document chunks
     └── [INSUFFICIENT EVIDENCE]
             └── Trigger Web Search Fallback -> Extract top web snippets
     │
     ▼
[Phase 6: Grounded Generation (Gemini API)]
     │ Stream response with strict instruction: cite only provided context
     ▼
[Phase 7: Citation Assembly & Verification]
     │ Emit verified citation payloads ([Doc: ...] or [Web: ...])
```

### Detailed Phase Execution

#### Phase 1: Query Processing
- Sanitize query string (strip control characters, trim whitespace).
- Generate query embedding using cached BGE-M3 model instance.

#### Phase 2: Hybrid Retrieval
- **Dense Vector Retrieval:**
  ```sql
  SELECT id, document_id, page_number, section_title, chunk_text,
         1 - (embedding <=> :query_vector) AS dense_score
  FROM document_chunks
  WHERE workspace_id = :workspace_id
  ORDER BY embedding <=> :query_vector ASC
  LIMIT 25;
  ```
- **Lexical BM25 Retrieval:**
  ```sql
  SELECT id, document_id, page_number, section_title, chunk_text,
         ts_rank_cd(tsv, plainto_tsquery('english', :query_text)) AS lexical_score
  FROM document_chunks
  WHERE workspace_id = :workspace_id AND tsv @@ plainto_tsquery('english', :query_text)
  ORDER BY lexical_score DESC
  LIMIT 25;
  ```

#### Phase 3: Reciprocal Rank Fusion (RRF)
- Merge the two ranked lists using standard RRF with smoothing constant $k=60$:
  $$RRF\_Score(d) = \sum_{m \in \{dense, lexical\}} \frac{1}{60 + \text{rank}_m(d)}$$
- Deduplicate candidates by `chunk_id` and retain top 30 candidates.

#### Phase 4: Cross-Encoder Reranking
- Pass candidate pairs `[(query, candidate.chunk_text), ...]` to Cross-Encoder.
- Sort candidates by rerank score descending and select top $N$ (default: $N=4$).

#### Phase 5: Relevance Gating
- Compare the maximum reranker score against `RELEVANCE_THRESHOLD` (e.g., score $\ge 0.40$).
- **Branch A (Sufficient Evidence):**
  - Proceed to generation using document chunks.
  - Set source mode to `DOCUMENT_GROUNDED`.
- **Branch B (Insufficient Evidence):**
  - If workspace has no documents or top rerank score is below threshold:
  - Query web search provider (e.g., Tavily or DuckDuckGo).
  - Format web search snippets with title and URL.
  - Set source mode to `WEB_FALLBACK`.

#### Phase 6: Grounded Generation & Strict Prompting
Assemble Gemini prompt enforcing strict grounding:
```
You are AI Tutor Assistant. Answer the user query strictly using the provided context below.
Rules:
1. Ground every claim in the provided context.
2. If using Document context, insert citations like [Doc: <file_name>, p. <page_number>].
3. If using Web context, insert citations like [Web: <title>](<url>).
4. Do not state facts that are not present in the context.
5. If the context does not contain enough information, clearly state that you do not have enough information.

<context>
{formatted_context}
</context>

<query>
{user_query}
</query>
```

#### Phase 7: Citation Assembly & Verification
- Parse emitted citation markers from stream or append structured citation metadata in the terminal event.
- Validate that all cited document files and page numbers match the actual chunks supplied in `<context>`.

## 5. Engineering Rules
1. **Document Primacy:** Under no circumstances may web search run if document retrieval succeeds and meets the relevance threshold.
2. **Deterministic Gating:** The relevance gate must rely on explicit scoring thresholds or a lightweight classification prompt, not an unguided LLM whim.
3. **No Unanchored Citations:** Citations without valid source chunk references must never be delivered to the client.
4. **Streaming Fidelity:** Stream tokens continuously; do not buffer the entire LLM response on the server before sending to the client.
5. **Workspace Sandboxing:** Every database retrieval query MUST include `WHERE workspace_id = :workspace_id`. Cross-workspace retrieval is a critical security bug.

## 6. Validation Checklist
- [ ] Retrieval queries are strictly scoped to the active `workspace_id`.
- [ ] Hybrid search combines both dense and lexical results via RRF.
- [ ] Cross-Encoder reranks the fused candidates before passing to LLM.
- [ ] If relevance threshold is met, web search is NOT called.
- [ ] If relevance threshold fails, web search fallback is invoked and source is tagged as `Web`.
- [ ] Document citations contain exact `file_name` and `page_number`.
- [ ] Generation response streams tokens incrementally via SSE.

## 7. Common Failure Modes & Mitigations
- **Failure:** Web search always fires, ignoring uploaded PDFs.
  - *Root Cause:* Relevance threshold set too high or reranker scoring not normalized.
  - *Mitigation:* Calibrate relevance threshold against test set; log rerank scores during staging.
- **Failure:** Hallucinated page numbers (e.g., citing page 99 in a 10-page document).
  - *Root Cause:* System prompt did not bind citation numbers to chunk metadata.
  - *Mitigation:* Pre-format chunks as `[Source ID: doc1, Page: 4]` in prompt and validate citations post-generation.
- **Failure:** Slow chat responses (> 5 seconds before first token).
  - *Root Cause:* Reranking too many candidates (e.g., 100 chunks) on CPU.
  - *Mitigation:* Limit RRF candidate pool to top 20-30 chunks before reranking.
