"""Fast / Quality Path V1 heuristic router for Phase 7 retrieval.

Evaluates retrieval-side signals after Reciprocal Rank Fusion (RRF) to decide
whether expensive Cross-Encoder reranking can be safely bypassed (FAST)
or is required to resolve ambiguity (QUALITY).

Signals evaluated:
1. Top candidate RRF score (indicates dual-source alignment)
2. Score gap between top-1 and top-2 candidates (margin of confidence)
3. Dual-channel consensus on the top candidate (both dense and lexical sources)
4. Agreement distribution across top candidates

Supported policies:
- always_fast: bypass Cross-Encoder reranker completely
- always_quality: always invoke Cross-Encoder reranker
- adaptive: heuristic confidence routing
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalRoutingMode(str, Enum):
    ALWAYS_FAST = "always_fast"
    ALWAYS_QUALITY = "always_quality"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingDecision:
    """Diagnostic outcome of the routing decision."""

    path: str  # "FAST" | "QUALITY"
    mode: str
    reason: str
    top_rrf_score: float
    score_gap: float
    top_has_dual_consensus: bool
    agreement_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_path": self.path,
            "routing_mode": self.mode,
            "routing_reason": self.reason,
            "top_rrf_score": round(self.top_rrf_score, 6),
            "score_gap": round(self.score_gap, 6),
            "top_has_dual_consensus": self.top_has_dual_consensus,
            "agreement_count": self.agreement_count,
        }


class RetrievalRouter:
    """Heuristic rule-based router deciding between Fast path (RRF) and Quality path (Cross-Encoder)."""

    @classmethod
    def decide(
        cls,
        candidates: List[Any],
        mode: Optional[str] = None,
        rrf_score_threshold: Optional[float] = None,
        score_gap_threshold: Optional[float] = None,
    ) -> RoutingDecision:
        """Evaluates fused candidates to produce a deterministic routing decision."""
        routing_mode = (mode or getattr(settings, "RETRIEVAL_ROUTING_MODE", "adaptive")).lower().strip()
        score_thresh = (
            rrf_score_threshold
            if rrf_score_threshold is not None
            else getattr(settings, "ROUTER_RRF_SCORE_THRESHOLD", 0.025)
        )
        gap_thresh = (
            score_gap_threshold
            if score_gap_threshold is not None
            else getattr(settings, "ROUTER_SCORE_GAP_THRESHOLD", 0.003)
        )

        # Baseline: empty candidate pool
        if not candidates:
            return RoutingDecision(
                path="FAST",
                mode=routing_mode,
                reason="Empty candidate pool; nothing to rerank",
                top_rrf_score=0.0,
                score_gap=0.0,
                top_has_dual_consensus=False,
                agreement_count=0,
            )

        # Signal 1: Top candidate RRF score
        top_cand = candidates[0]
        top_rrf = float(getattr(top_cand, "rrf_score", 0.0) or 0.0)

        # Signal 2: Top-1 vs Top-2 score gap
        if len(candidates) > 1:
            runner_up_rrf = float(getattr(candidates[1], "rrf_score", 0.0) or 0.0)
            score_gap = top_rrf - runner_up_rrf
        else:
            score_gap = top_rrf

        # Signal 3: Dual-channel consensus on the top candidate
        top_sources = set(getattr(top_cand, "retrieval_sources", set()) or set())
        top_has_dual = "dense" in top_sources and "lexical" in top_sources

        # Signal 4: Agreement count across top 3
        agreement_count = sum(
            1
            for c in candidates[:3]
            if "dense" in (getattr(c, "retrieval_sources", set()) or set())
            and "lexical" in (getattr(c, "retrieval_sources", set()) or set())
        )

        # Evaluate forced modes
        if routing_mode == RetrievalRoutingMode.ALWAYS_FAST.value:
            return RoutingDecision(
                path="FAST",
                mode=routing_mode,
                reason="Forced always_fast routing policy",
                top_rrf_score=top_rrf,
                score_gap=score_gap,
                top_has_dual_consensus=top_has_dual,
                agreement_count=agreement_count,
            )

        if routing_mode == RetrievalRoutingMode.ALWAYS_QUALITY.value:
            return RoutingDecision(
                path="QUALITY",
                mode=routing_mode,
                reason="Forced always_quality routing policy",
                top_rrf_score=top_rrf,
                score_gap=score_gap,
                top_has_dual_consensus=top_has_dual,
                agreement_count=agreement_count,
            )

        # ADAPTIVE HEURISTIC:
        # High confidence condition:
        # 1. Top candidate was retrieved by BOTH dense and lexical channels
        # 2. Top RRF score meets or exceeds threshold (e.g., >= 0.025)
        # 3. Score margin over runner-up meets or exceeds threshold (e.g., >= 0.003)
        is_confident = (
            top_has_dual
            and top_rrf >= score_thresh
            and score_gap >= gap_thresh
        )

        if is_confident:
            reason = (
                f"High confidence dual-channel consensus (RRF={top_rrf:.4f} >= {score_thresh}, "
                f"gap={score_gap:.4f} >= {gap_thresh})"
            )
            logger.info("RetrievalRouter selecting FAST path: %s", reason)
            return RoutingDecision(
                path="FAST",
                mode=routing_mode,
                reason=reason,
                top_rrf_score=top_rrf,
                score_gap=score_gap,
                top_has_dual_consensus=top_has_dual,
                agreement_count=agreement_count,
            )
        else:
            reason = (
                f"Candidate ambiguity or low consensus (RRF={top_rrf:.4f}, gap={score_gap:.4f}, "
                f"dual_consensus={top_has_dual}); escalating to Cross-Encoder"
            )
            logger.info("RetrievalRouter selecting QUALITY path: %s", reason)
            return RoutingDecision(
                path="QUALITY",
                mode=routing_mode,
                reason=reason,
                top_rrf_score=top_rrf,
                score_gap=score_gap,
                top_has_dual_consensus=top_has_dual,
                agreement_count=agreement_count,
            )
