<div align="center">

# AI Tutor Assistant

**A document-grounded AI learning assistant with hybrid RAG, source citations, web fallback, and verified math solving.**

<p align="center">
  <img src="assets/background.png" alt="AI Tutor Assistant — Document-Grounded RAG Architecture" width="100%" />
</p>

[**Live Demo**](https://ai-tutor-document-grounded.vercel.app) · [**Source Code**](https://github.com/mth1805/ai-tutor-document-grounded) · [**Deployment Guide**](docs/deployment.md)

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white" alt="Next.js" />
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" />
<img src="https://img.shields.io/badge/Supabase-3FCF8E?style=flat-square&logo=supabase&logoColor=black" alt="Supabase" />
<img src="https://img.shields.io/badge/pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="pgvector" />

<img src="https://img.shields.io/badge/BGE--M3-6366F1?style=flat-square" alt="BGE-M3" />
<img src="https://img.shields.io/badge/Cross--Encoder-7C3AED?style=flat-square" alt="Cross-Encoder" />
<img src="https://img.shields.io/badge/Gemini_API-8E75B2?style=flat-square&logo=googlegemini&logoColor=white" alt="Gemini API" />
<img src="https://img.shields.io/badge/SymPy-3B5526?style=flat-square" alt="SymPy" />
<img src="https://img.shields.io/badge/Docker_Compose-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker Compose" />
<img src="https://img.shields.io/badge/Modal-171717?style=flat-square" alt="Modal" />
<img src="https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white" alt="Vercel" />

</div>

## Overview

AI Tutor lets users upload study materials and ask questions in natural language. Unlike a general chatbot, it **prioritizes evidence from the user's own documents**, provides page-level citations, and turns to web search only when document evidence is insufficient. It also supports guided learning and SymPy-backed mathematical computation.

**Key capabilities:** private workspaces and chat history · PDF/DOC/DOCX/TXT/image ingestion · OCR and Word equation extraction · document-aware questions and summaries · source-linked answers · streaming responses · three tutoring modes · verified math operations.

## End-to-End Architecture

```mermaid
flowchart TD
    U[User: upload / ask a question] --> FE[Next.js UI]
    FE --> API[FastAPI + Auth / Workspace Isolation]
    API --> ING[Async ingestion: parsing, OCR, OMML, chunking]
    ING --> EMB[BGE-M3: 1024-dimensional embeddings]
    EMB --> DB[(Supabase PostgreSQL + pgvector)]
    API --> Q[Document-aware query routing]
    Q --> RET[Hybrid retrieval]
    DB --> RET
    RET --> D[Dense vector search]
    RET --> L[Lexical search: PostgreSQL FTS]
    D --> RRF[Reciprocal Rank Fusion]
    L --> RRF
    RRF --> CE[Cross-Encoder reranking]
    CE --> G{Sufficient document evidence?}
    G -->|Yes| CTX[Grounded document context]
    G -->|No| WEB[Tavily web fallback]
    Q -->|Math exercise| M[Structured task + SymPy]
    CTX --> LLM[Gemini tutor response]
    WEB --> LLM
    M --> LLM
    LLM --> OUT[Streaming answer + citations / tutoring guidance]
    OUT --> FE
```

## Technical Highlights

| Design decision | Why it matters |
|---|---|
| **BGE-M3 + pgvector HNSW** | Multilingual semantic retrieval for English/Vietnamese materials, with vector search persisted in PostgreSQL. |
| **Hybrid retrieval + RRF** | Combines semantic relevance with exact-term matching; fuses candidate rankings before reranking. |
| **Cross-Encoder reranking + relevance gate** | Scores query–chunk pairs more precisely and decides whether available document evidence is sufficient. |
| **Document-first + web fallback** | Avoids replacing uploaded material with generic web answers; falls back to Tavily when document evidence is insufficient. |
| **Citations & document reference resolution** | Connects answers to source pages and resolves references such as “this file” using authorized workspace context. |
| **SymPy-verified math** | Separates deterministic computation from LLM explanation; supports equations, derivatives, integrals, simplification, and arithmetic. |
| **Secure, asynchronous architecture** | Supabase Auth/RLS, private storage, background ingestion, streaming via SSE, and persistent workspaces/conversations. |

### Learning behavior

The same verified result can be presented in **Light Guidance** (hint), **Detailed Guidance** (guided steps), or **Full Solution** (complete explanation). DOCX OMML fractions and powers are preserved as searchable mathematical text when extracting learning materials.

## Evaluation & Deployment

- **Evaluation:** RAGAS-based retrieval/answer evaluation, citation correctness/completeness, and latency tracking. See the evaluation artifacts for dataset-specific results; do not interpret them as guarantees on unseen documents.
- **Local:** Docker Compose runs the frontend, FastAPI backend, and ingestion worker on CPU.
- **Cloud:** Next.js on Vercel; API/worker deployment via Modal; Supabase provides Auth, PostgreSQL/pgvector, and private Storage. Modal ingestion can use CPU by default or opt in to GPU.

## Run Locally

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
docker compose up --build
```

Open **http://localhost:3000**. Configure the development Supabase project and required environment variables before testing authenticated uploads. No GPU is required for local Docker.

For detailed setup, migrations, deployment, and implementation notes, see **[Deployment](docs/deployment.md)**, **[Local Docker](docs/local-docker-runtime.md)**, **[Math Solver](docs/phase9_1_verified_math_solver.md)**, and **[DOCX Equation Handling](docs/docx-equations-sidebar.md)**.
