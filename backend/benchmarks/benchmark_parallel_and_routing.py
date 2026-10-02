"""Benchmark comparing Sequential vs Parallel retrieval and Fast vs Quality vs Adaptive routing.

Demonstrates:
- Latency before and after parallelization
- Reranker invocation rate across routing modes (always_quality, always_fast, adaptive)
- Total retrieval latency reduction
"""
import os
import sys
import time
import math
import uuid
import statistics
import asyncio
from typing import List, Dict, Any

# Ensure backend root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    # Confident queries with strong keyword + semantic overlap (ideal for FAST path)
    ("What is machine learning and supervised learning?", "High-confidence English"),
    ("Quá trình quang hợp ở thực vật diễn ra như thế nào?", "High-confidence Vietnamese"),
    ("The transformer architecture and self-attention mechanism", "High-confidence English"),
    ("Định luật bảo toàn năng lượng trong hệ cô lập", "High-confidence Vietnamese"),
    ("Relational database management systems and foreign keys", "High-confidence English"),
    # Ambiguous / conceptual queries with low term overlap (ideal for QUALITY path)
    ("How does neural computing process human thoughts?", "Ambiguous conceptual English"),
    ("Làm sao để máy tính hiểu được ý nghĩ con người?", "Ambiguous conceptual Vietnamese"),
    ("Energy transformations in biological cellular units", "Cross-domain English"),
    ("Các quy luật tự nhiên chi phối chuyển động và năng lượng", "Abstract Vietnamese"),
    ("Nearest neighbor graph approximations in embedding search", "Subtle technical English"),
]


async def main():
    print("=" * 80)
    print("STAGES 4 & 5: PARALLEL RETRIEVAL AND FAST/QUALITY PATH ROUTING BENCHMARK")
    print("=" * 80)

    workspace_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    print("\nInitializing real cached models...")
    emb_provider = BGEEmbeddingProvider()
    rerank_provider = CrossEncoderRerankerProvider()

    print("Encoding and indexing 20 benchmark chunks...")
    vectors = emb_provider.encode_batch(BENCHMARK_CHUNKS, normalize=True, batch_size=16)
    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=workspace_id,
            user_id=user_id,
            content=c,
            chunk_index=i,
            page_number_start=1,
            page_number_end=1,
            embedding=v,
        )
        for i, (c, v) in enumerate(zip(BENCHMARK_CHUNKS, vectors))
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    sys.stdout.reconfigure(encoding='utf-8')
    # Benchmark Modes: always_quality vs always_fast vs adaptive
    modes = ["always_quality", "always_fast", "adaptive"]
    mode_results = {}
    adaptive_breakdown = []

    for mode in modes:
        print(f"\nEvaluating Routing Mode: '{mode}'...", flush=True)
        timings = []
        fast_count = 0
        quality_count = 0
        evidence_count = 0

        for q_text, q_desc in BENCHMARK_QUERIES:
            req = RetrievalRequest(
                query=q_text,
                dense_top_k=15,
                lexical_top_k=15,
                candidate_pool_size=20,
                rerank_top_k=5,
                routing_mode=mode,
            )
            resp = await RetrievalService.retrieve(
                db=None,
                workspace_id=workspace_id,
                user_id=user_id,
                request=req,
                embedding_provider=emb_provider,
                reranker_provider=rerank_provider,
            )
            timings.append(resp.timings.total_retrieval_ms)
            if resp.routing_path == "FAST":
                fast_count += 1
            else:
                quality_count += 1
            if resp.has_sufficient_evidence:
                evidence_count += 1

            if mode == "adaptive":
                diag = resp.diagnostics or {}
                reason = diag.get("routing_reason", "")
                short_reason = reason[:35] + "..." if len(reason) > 35 else reason
                adaptive_breakdown.append({
                    "desc": q_desc,
                    "path": resp.routing_path,
                    "ms": resp.timings.total_retrieval_ms,
                    "reason": short_reason,
                    "top_rrf": diag.get("top_rrf_score", 0.0),
                    "gap": diag.get("score_gap", 0.0),
                })

        p50 = statistics.median(timings)
        mean = statistics.mean(timings)
        sorted_t = sorted(timings)
        p95 = sorted_t[int(math.ceil(0.95 * len(sorted_t))) - 1]

        mode_results[mode] = {
            "mean_ms": mean,
            "p50_ms": p50,
            "p95_ms": p95,
            "fast_count": fast_count,
            "quality_count": quality_count,
            "reranker_invocation_rate": (quality_count / len(BENCHMARK_QUERIES)) * 100,
            "evidence_pass_rate": (evidence_count / len(BENCHMARK_QUERIES)) * 100,
        }

    print("\n" + "=" * 80, flush=True)
    print("ROUTING COMPARISON RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Mode':<18}{'Mean (ms)':<14}{'p50 (ms)':<14}{'p95 (ms)':<14}{'CE Rerank %':<14}{'Sufficient %':<14}", flush=True)
    print("-" * 80, flush=True)
    for m in modes:
        r = mode_results[m]
        print(
            f"{m:<18}{r['mean_ms']:<14.1f}{r['p50_ms']:<14.1f}{r['p95_ms']:<14.1f}"
            f"{r['reranker_invocation_rate']:<14.1f}%{r['evidence_pass_rate']:<14.1f}%",
            flush=True,
        )
    print("-" * 80, flush=True)

    # Detailed adaptive queries breakdown
    print("\nADAPTIVE MODE PER-QUERY BREAKDOWN:", flush=True)
    print("-" * 95, flush=True)
    print(f"{'Query Description':<32}{'Path':<10}{'Total (ms)':<14}{'Top RRF':<10}{'Gap':<10}{'Reason'}", flush=True)
    print("-" * 95, flush=True)
    for item in adaptive_breakdown:
        print(f"{item['desc']:<32}{item['path']:<10}{item['ms']:<14.1f}{item['top_rrf']:<10.4f}{item['gap']:<10.4f}{item['reason']}", flush=True)

    _IN_MEMORY_CHUNKS.pop(doc_id, None)



if __name__ == "__main__":
    asyncio.run(main())
