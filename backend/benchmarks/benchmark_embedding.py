"""Reproducible benchmark for BAAI/bge-m3 embedding across batch sizes on CPU.

Measures:
- Chunks per second
- Average ms per chunk
- Median and p95 latency
- Memory usage (process RSS)

Uses cached weights and does not mutate production database or documents.
"""
import os
import sys
import time
import statistics
import tracemalloc
import ctypes
from typing import List, Dict, Any

def get_process_memory_mb() -> float:
    """Returns working set size in MB using standard ctypes Windows API without external dependencies."""
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0

# Ensure backend root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ml.bge_provider import BGEEmbeddingProvider


# 64 realistic textbook/document chunks (Vietnamese & English, ~100-200 words each)
SAMPLE_CHUNKS: List[str] = [
    # English chunks: AI, Biology, Physics, Computer Science
    (
        "Artificial intelligence is a branch of computer science focused on building smart machines "
        "capable of performing tasks that typically require human intelligence. These tasks include "
        "visual perception, speech recognition, decision-making, and translation between languages."
    ),
    (
        "Machine learning algorithms build a model based on sample data, known as training data, in "
        "order to make predictions or decisions without being explicitly programmed to do so. Supervised "
        "learning is where the algorithm is provided with labeled input-output pairs."
    ),
    (
        "Deep learning architectures such as deep neural networks, convolutional neural networks, and "
        "recurrent neural networks have been applied to fields including computer vision, speech "
        "recognition, natural language processing, and bioinformatics."
    ),
    (
        "The transformer architecture is based on self-attention mechanisms that compute representations "
        "of an input sequence without using sequence-aligned recurrent neural networks or convolutions. "
        "This allows for significantly more parallelization during training."
    ),
    (
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy "
        "into chemical energy that, through cellular respiration, can later be released to fuel the "
        "organism's metabolic activities. The reaction takes place inside chloroplasts."
    ),
    (
        "Chlorophyll absorbs light most strongly in the blue and red portions of the electromagnetic "
        "spectrum, whereas it reflects green light, giving plant leaves their characteristic green color. "
        "The light reactions produce ATP and NADPH for the Calvin cycle."
    ),
    (
        "Cellular respiration is a set of metabolic reactions and processes that take place in the cells "
        "of organisms to convert chemical energy from nutrients into adenosine triphosphate (ATP), "
        "and then release waste products including carbon dioxide and water."
    ),
    (
        "Newton's laws of motion are three basic laws of classical mechanics that describe the relationship "
        "between the motion of an object and the forces acting on it. The first law states that a body remains "
        "at rest or in motion at a constant velocity unless acted upon by a force."
    ),
    # Vietnamese chunks: Trí tuệ nhân tạo, Sinh học, Vật lý, Giáo dục
    (
        "Trí tuệ nhân tạo là một ngành thuộc lĩnh vực khoa học máy tính với mục tiêu tự động hóa các hành vi "
        "thông minh. Các hệ thống AI hiện đại có khả năng xử lý lượng dữ liệu khổng lồ để nhận diện mẫu, "
        "đưa ra dự báo và hỗ trợ quá trình ra quyết định trong kinh doanh và y tế."
    ),
    (
        "Học sâu là một tập con của học máy dựa trên các mạng nơ-ron nhân tạo với nhiều tầng biểu diễn. "
        "Các kiến trúc học sâu như CNN và Transformer đã tạo ra những bước đột phá vượt bậc trong các "
        "bài toán nhận dạng hình ảnh, thị giác máy tính và xử lý ngôn ngữ tự nhiên."
    ),
    (
        "Quá trình quang hợp ở thực vật là quá trình tổng hợp chất hữu cơ từ các chất vô cơ như nước và "
        "khí carbonic dưới tác dụng của năng lượng ánh sáng mặt trời được diệp lục hấp thụ. Khí oxy được "
        "giải phóng ra khí quyển như một sản phẩm phụ quan trọng cho sự sống."
    ),
    (
        "Định luật bảo toàn năng lượng khẳng định rằng năng lượng không tự nhiên sinh ra và cũng không tự "
        "nhiên mất đi, nó chỉ chuyển hóa từ dạng này sang dạng khác hoặc truyền từ vật này sang vật khác. "
        "Tổng năng lượng trong một hệ cô lập luôn là một hằng số."
    ),
    (
        "Hệ thống cơ sở dữ liệu quan hệ tổ chức dữ liệu thành các bảng gồm hàng và cột. Mỗi bảng có một "
        "khóa chính định danh duy nhất cho mỗi bản ghi, và các liên kết ngoại khóa giúp thiết lập mối "
        "quan hệ chặt chẽ giữa các bảng với tính toàn vẹn dữ liệu cao."
    ),
    (
        "Chỉ mục vector cho phép tìm kiếm các điểm dữ liệu tương tự nhau trong không gian nhiều chiều "
        "với độ phức tạp tính toán thấp. Thuật toán HNSW xây dựng đồ thị phân tầng giúp cân bằng tối ưu "
        "giữa tốc độ truy vấn và độ chính xác Recall."
    ),
    (
        "Mô hình ngôn ngữ lớn được huấn luyện trước trên kho ngữ liệu văn bản quy mô lớn bằng phương pháp "
        "tự giám sát. Sau đó mô hình được tinh chỉnh có hướng dẫn và căn chỉnh theo phản hồi của con người "
        "để trở thành trợ lý học tập đắc lực và an toàn."
    ),
    (
        "Thuật toán tìm kiếm lai kết hợp giữa tìm kiếm ngữ nghĩa dày đặc và tìm kiếm từ khóa kinh điển. "
        "Phương pháp Reciprocal Rank Fusion kết hợp thứ hạng từ cả hai nguồn để tạo ra danh sách ứng viên "
        "đồng thuận có độ bao phủ thông tin cao nhất."
    ),
] * 4  # 16 unique chunks * 4 = 64 realistic chunks


def run_embedding_benchmark(batch_sizes=[8, 16, 32, 64], repeats_per_batch=2) -> Dict[str, Any]:
    print("=" * 70)
    print("STAGE 2: BGE-M3 EMBEDDING BENCHMARK (CPU)")
    print("=" * 70)

    initial_rss_mb = get_process_memory_mb()
    print(f"Initial Process Memory: {initial_rss_mb:.1f} MB")

    print("\nInitializing BGEEmbeddingProvider singleton (using cached weights)...")
    t_load_start = time.perf_counter()
    provider = BGEEmbeddingProvider()
    t_load = (time.perf_counter() - t_load_start) * 1000
    print(f"Provider initialized in {t_load:.1f} ms on device: {provider.device}")

    # Warmup pass (not timed in benchmark metrics)
    print("Warming up model with 2 micro-chunks...")
    provider.encode_batch(SAMPLE_CHUNKS[:2], normalize=True, batch_size=2)
    warmup_rss_mb = get_process_memory_mb()
    print(f"Post-warmup Memory: {warmup_rss_mb:.1f} MB\n")

    num_chunks = len(SAMPLE_CHUNKS)
    results = {}

    print(f"Running benchmark on {num_chunks} representative chunks...")
    print("-" * 70)
    print(f"{'Batch Size':<12}{'Median (s)':<12}{'p95 (s)':<10}{'Chunks/s':<12}{'ms/chunk':<12}{'RAM (MB)':<10}")
    print("-" * 70)

    for bs in batch_sizes:
        durations = []
        for r in range(repeats_per_batch):
            t0 = time.perf_counter()
            vectors = provider.encode_batch(SAMPLE_CHUNKS, normalize=True, batch_size=bs)
            elapsed = time.perf_counter() - t0
            assert len(vectors) == num_chunks
            assert len(vectors[0]) == 1024
            durations.append(elapsed)

        median_dur = statistics.median(durations)
        # Approximate p95
        sorted_dur = sorted(durations)
        p95_idx = int(math.ceil(0.95 * len(sorted_dur))) - 1
        p95_dur = sorted_dur[max(0, p95_idx)]

        chunks_per_sec = num_chunks / median_dur
        ms_per_chunk = (median_dur * 1000) / num_chunks
        current_rss_mb = get_process_memory_mb()

        results[bs] = {
            "num_chunks": num_chunks,
            "durations": durations,
            "median_s": median_dur,
            "p95_s": p95_dur,
            "chunks_per_sec": chunks_per_sec,
            "ms_per_chunk": ms_per_chunk,
            "memory_mb": current_rss_mb,
        }

        print(
            f"{bs:<12}{median_dur:<12.3f}{p95_dur:<10.3f}{chunks_per_sec:<12.2f}"
            f"{ms_per_chunk:<12.2f}{current_rss_mb:<10.1f}"
        )

    print("-" * 70)

    # Find optimal batch size
    fastest_bs = max(results.keys(), key=lambda b: results[b]["chunks_per_sec"])
    print(f"\nOptimal Batch Size: {fastest_bs} ({results[fastest_bs]['chunks_per_sec']:.2f} chunks/sec)")
    return results


if __name__ == "__main__":
    import math
    run_embedding_benchmark()
