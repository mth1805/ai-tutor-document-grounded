---
name: rag-evaluation
description: Guides quantitative benchmarking of retrieval (Recall@K, MRR) and generation (Faithfulness, Answer Relevance) across Dense, BM25, Hybrid, and Reranked pipelines.
---

# RAG Evaluation Skill

## 1. Purpose
The `rag-evaluation` skill establishes rigorous, objective measurement standards for the **AI Tutor Assistant** RAG pipeline. It guides coding agents in executing benchmark experiments, computing retrieval metrics (Recall@K, MRR), evaluating generation metrics (Faithfulness, Answer Relevance, Context Relevance), and scientifically validating optimizations over subjective assertions.

## 2. When to Use
Use this skill when:
- Implementing Phase 12 (Evaluation Phase).
- Evaluating retrieval pipeline changes (e.g., comparing Dense vs. BM25 vs. Hybrid vs. Hybrid + Reranker).
- Tuning hyperparameters: chunk size, chunk overlap, top-K values, RRF smoothing constant $k$, reranker score thresholds.
- Measuring model hallucination rates, citation accuracy, or answer faithfulness.
- Running regression benchmarks before deploying RAG updates.

## 3. What to Inspect First
Before executing evaluations, inspect:
1. `backend/tests/evaluation/` or `evals/` — Benchmark evaluation scripts and datasets.
2. `evals/datasets/` — Gold-standard evaluation question-context-answer test sets.
3. `backend/app/services/retrieval/` & `backend/app/services/reranker/` — The exact retrieval configurations under test.
4. Database test fixtures containing reproducible test documents and chunk embeddings.

## 4. Evaluation Workflow
Follow this 5-stage benchmark workflow:

```
[Stage 1: Dataset Preparation]
      │ Curate or generate (query, ground_truth_chunk_ids, expected_answer) triples
      ▼
[Stage 2: Retrieval Experiment Execution]
      │ Run identical query set across:
      │ Config A: Dense Only (BGE-M3)
      │ Config B: BM25 Only (PostgreSQL tsvector)
      │ Config C: Hybrid (Dense + BM25 with RRF)
      │ Config D: Hybrid + Cross-Encoder Reranker
      ▼
[Stage 3: Retrieval Metrics Calculation]
      │ Compute Recall@K, Precision@K, MRR for each configuration
      ▼
[Stage 4: Generation Faithfulness Evaluation]
      │ Run LLM generation with retrieved contexts; evaluate Faithfulness & Relevance
      ▼
[Stage 5: Comparative Reporting]
      │ Produce markdown comparison table and statistical verdict
```

### Stage 1: Gold-Standard Dataset Structure
Evaluation datasets must be structured as JSONL or JSON:
```json
[
  {
    "eval_id": "eval-001",
    "workspace_id": "ws-test-physics",
    "query": "What is Gauss's law for electric fields?",
    "ground_truth_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2"],
    "ground_truth_pages": [14, 15],
    "ground_truth_answer": "Gauss's law states that the net electric flux through any closed Gaussian surface is equal to the net charge enclosed divided by the permittivity."
  }
]
```

### Stage 2 & 3: Retrieval Metrics Formulation
For each configuration, evaluate retrieved candidates at $K \in \{3, 5, 10\}$:

1. **Recall@K:**
   $$\text{Recall@K} = \frac{|\text{Retrieved Chunks}_{@K} \cap \text{Ground Truth Chunks}|}{|\text{Ground Truth Chunks}|}$$
2. **Mean Reciprocal Rank (MRR):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
   *(where $\text{rank}_i$ is the rank position of the first relevant chunk).*
3. **Precision@K:**
   $$\text{Precision@K} = \frac{|\text{Retrieved Chunks}_{@K} \cap \text{Ground Truth Chunks}|}{K}$$

### Stage 4: Generation Metrics Formulation
Evaluate LLM responses using automated evaluation harnesses (e.g., Ragas, TruLens, or calibrated LLM-as-a-judge):
- **Faithfulness (0.0 to 1.0):** Proportion of claims in the generated response that can be directly inferred from the retrieved context. (Target: $\ge 0.95$).
- **Answer Relevance (0.0 to 1.0):** Degree to which the answer directly addresses the user query. (Target: $\ge 0.90$).
- **Context Relevance (0.0 to 1.0):** Ratio of useful sentences in retrieved chunks to total sentences in retrieved chunks. (Target: $\ge 0.70$).
- **Citation Precision (0.0 to 1.0):** Percentage of cited page references that actually contain the referenced claim. (Target: $1.00$).

### Stage 5: Comparative Evaluation Report Template
Every evaluation run must output a comparative report:

| Pipeline Configuration | Recall@3 | Recall@5 | MRR | Faithfulness | TTFT (s) | Total Latency (s) |
|---|---|---|---|---|---|---|
| **Dense Only** | 0.68 | 0.76 | 0.62 | 0.88 | 1.1s | 2.4s |
| **BM25 Only** | 0.61 | 0.71 | 0.55 | 0.84 | 0.9s | 2.1s |
| **Hybrid (RRF)** | 0.78 | 0.85 | 0.72 | 0.91 | 1.2s | 2.6s |
| **Hybrid + Cross-Encoder** | **0.86** | **0.93** | **0.84** | **0.97** | 1.4s | 3.1s |

## 5. Engineering Rules
1. **Never Rely on Vibe Checks:** A pipeline change cannot be declared an improvement based on testing 1 or 2 ad-hoc queries. Use a minimum benchmark of 30+ structured evaluation queries.
2. **Deterministic Evaluation Sets:** Test questions and ground-truth chunks must be version-controlled in the repository.
3. **Isolate Latency from Accuracy:** Report retrieval accuracy alongside latency. An approach that increases Recall@5 by 1% but increases latency by 4 seconds is unacceptable.
4. **Zero Contamination:** Never train or tune thresholds on the evaluation test set; use a distinct validation set.

## 6. Validation Checklist
- [ ] Evaluation dataset contains at least 30 diverse queries covering technical terms, summaries, and exact facts.
- [ ] Each query has verified ground-truth chunk IDs and page references.
- [ ] Recall@K and MRR are computed objectively with automated scripts.
- [ ] Faithfulness is evaluated to detect hallucinations.
- [ ] Comparative table shows baseline vs. proposed configuration.

## 7. Common Failure Modes & Mitigations
- **Failure:** Overfitting chunking parameters to one specific document.
  - *Root Cause:* Benchmark only tested a single PDF.
  - *Mitigation:* Include diverse document formats: academic papers, slide decks, multi-column reports, and short DOCX files.
- **Failure:** High retrieval recall but low answer faithfulness.
  - *Root Cause:* Retrieved chunks contain relevant keywords but the prompt does not properly constrain the LLM.
  - *Mitigation:* Refine prompt instructions and lower LLM temperature for factual queries.
