"""Reproducible benchmark for Phase 7 end-to-end retrieval latency on CPU.

Measures stage-by-stage timings:
- query_embedding_ms
- dense_retrieval_ms
- lexical_retrieval_ms
- rrf_ms
- rerank_ms
- total_retrieval_ms

Compares retrieval with reranking vs without reranking.
Identifies the primary latency bottleneck across English and Vietnamese queries.
"""
import os
import sys
import time
import math
import uuid
import statistics
from typing import List, Dict, Any

# Ensure backend root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.chunk import DocumentChunk
from app.schemas.retrieval import RetrievalRequest
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService
from app.ml.bge_provider import BGEEmbeddingProvider
from app.ml.reranker_provider import CrossEncoderRerankerProvider


BENCHMARK_CHUNKS = [
    # English chunks
    "Artificial intelligence is a branch of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and translation.",
    "Machine learning algorithms build a mathematical model based on sample data, known as training data, to make predictions or decisions without being explicitly programmed. Supervised learning pairs inputs with desired outputs.",
    "Deep learning architectures such as convolutional neural networks and recurrent neural networks have revolutionized computer vision, natural language processing, and medical diagnosis.",
    "The transformer architecture relies on self-attention mechanisms that compute representations of an input sequence without sequence-aligned recurrent units, enabling massive parallelization during training.",
    "Photosynthesis is a biological process used by green plants to convert solar light energy into chemical energy stored in glucose. The reactions occur within the chloroplasts of plant cells.",
    "Chlorophyll pigments absorb light most strongly in the blue and red wavelengths while reflecting green light. The light-dependent reactions generate ATP and NADPH for the Calvin cycle.",
    "Cellular respiration is a metabolic pathway that breaks down glucose to produce ATP, releasing carbon dioxide and water as waste products. It occurs in the mitochondria of eukaryotic cells.",
    "Newton's laws of motion describe the relationship between a body and the forces acting upon it. The first law of inertia states that velocity remains constant unless an external force acts.",
    "Relational database management systems organize data into structured tables with rows and columns. Foreign keys establish relational constraints to maintain referential integrity.",
    "Vector index algorithms like HNSW construct hierarchical graph layers for approximate nearest neighbor search, providing sub-linear query latency over high-dimensional vector spaces.",
    # Vietnamese chunks
    "Trí tuệ nhân tạo là ngành khoa học máy tính hướng tới việc tự động hóa các hành vi thông minh. AI hiện đại xử lý dữ liệu lớn để nhận dạng mẫu và hỗ trợ ra quyết định thông minh.",
    "Học sâu là nhánh của học máy sử dụng mạng nơ-ron nhiều tầng để trích xuất đặc trưng phân cấp. Kiến trúc Transformer đã thúc đẩy sự bùng nổ của các mô hình ngôn ngữ lớn.",
    "Cơ chế self-attention trong Transformer tính toán trọng số tương quan giữa các từ trong câu mà không cần xử lý tuần tự, giúp tối ưu hóa việc huấn luyện trên quy mô lớn.",
    "Quá trình quang hợp ở thực vật chuyển đổi năng lượng ánh sáng mặt trời thành năng lượng hóa học dưới dạng đường glucose, diễn ra chủ yếu ở lục lạp trong tế bào lá.",
    "Diệp lục hấp thụ mạnh ánh sáng ở vùng đỏ và xanh lam, phản xạ ánh sáng xanh lục. Chu trình Calvin sử dụng ATP và NADPH từ pha sáng để cố định khí CO2 thành chất hữu cơ.",
    "Định luật bảo toàn năng lượng phát biểu rằng năng lượng không tự sinh ra và không tự mất đi, chỉ biến đổi từ dạng này sang dạng khác trong một hệ cô lập.",
    "Định luật một Newton về quán tính khẳng định mọi vật sẽ giữ nguyên trạng thái đứng yên hoặc chuyển động thẳng đều nếu không chịu tác dụng của ngoại lực nào.",
    "Hệ quản trị cơ sở dữ liệu quan hệ tổ chức thông tin dưới dạng các bảng liên kết. Khóa chính và khóa ngoại đảm bảo tính toàn vẹn và nhất quán của dữ liệu.",
    "Chỉ mục vector cho phép tìm kiếm tương đồng ngữ nghĩa bằng độ đo khoảng cách cosine hoặc tích vô hướng, phục vụ cho các hệ thống tìm kiếm thông minh và RAG.",
    "Mô hình ngôn ngữ lớn được huấn luyện trước trên dữ liệu văn bản đồ sộ bằng kỹ thuật tự giám sát, sau đó được căn chỉnh để hỗ trợ học tập và giải đáp tri thức.",
]

BENCHMARK_QUERIES = [
    # English queries
    "What is machine learning and supervised learning?",
    "How does the transformer self-attention mechanism work?",
    "Explain the biological process of photosynthesis and chloroplasts.",
    "What are Newton's laws of motion in classical mechanics?",
    "How does vector index and HNSW graph work?",
    # Vietnamese queries
    "Trí tuệ nhân tạo và học sâu được ứng dụng như thế nào?",
    "Quá trình quang hợp ở thực vật diễn ra như thế nào?",
    "Định luật bảo toàn năng lượng phát biểu điều gì?",
    "Cơ sở dữ liệu quan hệ và khóa chính là gì?",
    "Mô hình ngôn ngữ lớn hoạt động ra sao?",
]


def run_retrieval_benchmark():
    print("=" * 75)
    print("STAGE 3: PHASE 7 RETRIEVAL LATENCY BENCHMARK (REAL MODELS ON CPU)")
    print("=" * 75)

    workspace_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    print("\n1. Initializing real cached models (singleton providers)...")
    emb_provider = BGEEmbeddingProvider()
    rerank_provider = CrossEncoderRerankerProvider()
    print("   Models loaded.")

    print("\n2. Seeding in-memory corpus with precomputed BGE-M3 embeddings...")
    vectors = emb_provider.encode_batch(BENCHMARK_CHUNKS, normalize=True, batch_size=16)

    chunks = []
    for idx, (content, vec) in enumerate(zip(BENCHMARK_CHUNKS, vectors)):
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=workspace_id,
            user_id=user_id,
            content=content,
            chunk_index=idx,
            page_number_start=1,
            page_number_end=1,
            embedding=vec,
        )
        chunks.append(chunk)

    _IN_MEMORY_CHUNKS[doc_id] = chunks
    print(f"   Corpus ready: {len(chunks)} chunks embedded and indexed.")

    # Warmup retrieval
    print("3. Warming up retrieval pipeline with 1 warmup query...")
    warmup_req = RetrievalRequest(query="warmup test", dense_top_k=10, lexical_top_k=10, rerank_top_k=3)
    import asyncio
    asyncio.run(
        RetrievalService.retrieve(
            db=None,
            workspace_id=workspace_id,
            user_id=user_id,
            request=warmup_req,
            embedding_provider=emb_provider,
            reranker_provider=rerank_provider,
        )
    )
    print("   Warmup complete.\n")

    # Benchmarking WITH Reranking
    print(f"4. Running benchmark WITH Reranker on {len(BENCHMARK_QUERIES)} queries...")
    print("-" * 75)
    print(f"{'Query ID':<10}{'Embed(ms)':<12}{'Dense(ms)':<12}{'Lex(ms)':<10}{'RRF(ms)':<10}{'Rerank(ms)':<14}{'Total(ms)':<10}")
    print("-" * 75)

    timings_with_rerank = []
    for i, q in enumerate(BENCHMARK_QUERIES, start=1):
        req = RetrievalRequest(
            query=q,
            dense_top_k=15,
            lexical_top_k=15,
            candidate_pool_size=20,
            rerank_top_k=5,
        )
        resp = asyncio.run(
            RetrievalService.retrieve(
                db=None,
                workspace_id=workspace_id,
                user_id=user_id,
                request=req,
                embedding_provider=emb_provider,
                reranker_provider=rerank_provider,
            )
        )
        t = resp.timings
        timings_with_rerank.append(t)
        print(
            f"Q{i:<9}{t.query_embedding_ms:<12.1f}{t.dense_retrieval_ms:<12.1f}"
            f"{t.lexical_retrieval_ms:<10.1f}{t.rrf_ms:<10.1f}{t.rerank_ms:<14.1f}{t.total_retrieval_ms:<10.1f}"
        )

    # Benchmarking WITHOUT Reranking (Candidate pool RRF only)
    print("\n5. Running benchmark WITHOUT Reranker (top 5 RRF only)...")
    timings_no_rerank = []
    for i, q in enumerate(BENCHMARK_QUERIES, start=1):
        req = RetrievalRequest(
            query=q,
            dense_top_k=15,
            lexical_top_k=15,
            candidate_pool_size=5,
            rerank_top_k=5,
        )
        # Custom run omitting rerank stage
        t0 = time.perf_counter()
        q_vec = emb_provider.encode_batch([q], normalize=True, batch_size=1)[0]
        t_embed = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        dense_cand = asyncio.run(
            RetrievalService.dense_retrieve(None, workspace_id, user_id, q_vec, 15)
        )
        t_dense = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        lex_cand = asyncio.run(
            RetrievalService.lexical_retrieve(None, workspace_id, user_id, q, 15)
        )
        t_lex = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        fused = RetrievalService.reciprocal_rank_fusion(dense_cand, lex_cand, 60, 5)
        t_rrf = (time.perf_counter() - t3) * 1000

        t_total = (time.perf_counter() - t0) * 1000
        timings_no_rerank.append({
            "embed": t_embed,
            "dense": t_dense,
            "lex": t_lex,
            "rrf": t_rrf,
            "total": t_total,
        })

    def calc_stats(vals: List[float]):
        s = sorted(vals)
        p50 = statistics.median(s)
        p95 = s[int(math.ceil(0.95 * len(s))) - 1]
        mean = statistics.mean(s)
        return mean, p50, p95

    # Summaries
    print("\n" + "=" * 75)
    print("STAGE 3 SUMMARY & LATENCY BOTTLENECK ANALYSIS")
    print("=" * 75)

    tot_with = [t.total_retrieval_ms for t in timings_with_rerank]
    emb_with = [t.query_embedding_ms for t in timings_with_rerank]
    dense_with = [t.dense_retrieval_ms for t in timings_with_rerank]
    lex_with = [t.lexical_retrieval_ms for t in timings_with_rerank]
    rrf_with = [t.rrf_ms for t in timings_with_rerank]
    rerank_with = [t.rerank_ms for t in timings_with_rerank]

    tot_no = [t["total"] for t in timings_no_rerank]

    m_tot, p50_tot, p95_tot = calc_stats(tot_with)
    m_emb, p50_emb, p95_emb = calc_stats(emb_with)
    m_dense, p50_dense, p95_dense = calc_stats(dense_with)
    m_lex, p50_lex, p95_lex = calc_stats(lex_with)
    m_rrf, p50_rrf, p95_rrf = calc_stats(rrf_with)
    m_rerank, p50_rerank, p95_rerank = calc_stats(rerank_with)

    m_no, p50_no, p95_no = calc_stats(tot_no)

    print(f"{'Pipeline Stage':<25}{'Mean (ms)':<15}{'p50 (ms)':<15}{'p95 (ms)':<15}{'% of Total':<12}")
    print("-" * 75)
    print(f"{'Query Embedding (BGE-M3)':<25}{m_emb:<15.1f}{p50_emb:<15.1f}{p95_emb:<15.1f}{m_emb/m_tot*100:<12.1f}%")
    print(f"{'Dense Retrieval':<25}{m_dense:<15.1f}{p50_dense:<15.1f}{p95_dense:<15.1f}{m_dense/m_tot*100:<12.1f}%")
    print(f"{'Lexical FTS Retrieval':<25}{m_lex:<15.1f}{p50_lex:<15.1f}{p95_lex:<15.1f}{m_lex/m_tot*100:<12.1f}%")
    print(f"{'Reciprocal Rank Fusion':<25}{m_rrf:<15.1f}{p50_rrf:<15.1f}{p95_rrf:<15.1f}{m_rrf/m_tot*100:<12.1f}%")
    print(f"{'Cross-Encoder Reranker':<25}{m_rerank:<15.1f}{p50_rerank:<15.1f}{p95_rerank:<15.1f}{m_rerank/m_tot*100:<12.1f}%")
    print("-" * 75)
    print(f"{'TOTAL (With Reranker)':<25}{m_tot:<15.1f}{p50_tot:<15.1f}{p95_tot:<15.1f}{'100.0%':<12}")
    print(f"{'TOTAL (WITHOUT Reranker)':<25}{m_no:<15.1f}{p50_no:<15.1f}{p95_no:<15.1f}{m_no/m_tot*100:<12.1f}%")
    print("-" * 75)

    print(f"\nBottleneck Breakdown:")
    print(f"1. Cross-Encoder Reranker: {m_rerank/m_tot*100:.1f}% of total retrieval time ({m_rerank:.1f} ms mean).")
    print(f"2. Query Embedding (BGE-M3): {m_emb/m_tot*100:.1f}% of total retrieval time ({m_emb:.1f} ms mean).")
    print(f"3. Dense + Lexical + RRF combined: {(m_dense+m_lex+m_rrf)/m_tot*100:.1f}% of total retrieval time.")
    print(f"\nSpeedup without Cross-Encoder: {m_tot / max(m_no, 0.001):.2f}x faster!")

    # Cleanup in-memory benchmark chunks
    _IN_MEMORY_CHUNKS.pop(doc_id, None)


if __name__ == "__main__":
    run_retrieval_benchmark()
