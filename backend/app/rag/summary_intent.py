"""Bilingual summary-intent classification for document chat requests."""
import re
from dataclasses import dataclass

from app.tools.solver_router import fold


_SUMMARY_ACTION = re.compile(r"\b(?:summari[sz](?:e|ing)|sum\s+up|overview|tom\s+tat|tong\s+ket)\b")
_SUMMARY_NOUN_REQUEST = re.compile(
    r"\b(?:give|provide|write|create|make|need|want)\b.{0,32}\bsummary\b|"
    r"\bsummary\s+of\b"
)
_WHOLE_DOCUMENT = re.compile(
    r"\b(?:whole|entire|complete|full)\s+(?:uploaded\s+)?(?:document|file|pdf|book)\b|"
    r"\b(?:all|entire)\s+(?:of\s+)?(?:the\s+)?(?:document|file|pdf|book)\b|"
    r"\b(?:toan\s+bo|tat\s+ca)\s+(?:noi\s+dung\s+)?(?:cua\s+)?(?:tai\s+lieu|file|pdf)\b|"
    r"\b(?:tai\s+lieu|file|pdf)\s+nay\b"
)
_DOCUMENT_OBJECT = re.compile(
    r"\b(?:this|that|the|uploaded|selected)\s+(?:document|file|pdf|book)\b|"
    r"\b(?:tai\s+lieu|file|pdf)\s+(?:nay|do|vua\s+tai)\b"
)
_FOCUSED_SCOPE = re.compile(
    r"\b(?:section|chapter|part|paragraph|topic|subject|about|regarding|on|covered|"
    r"conclusion|introduction|methodology|methods|abstract|findings|results|appendix|"
    r"chuong|muc|phan|chu\s+de|noi\s+dung\s+ve|ve|ket\s+luan|mo\s+dau|phuong\s+phap)\b"
)
_FILENAME = re.compile(r"(?<!\w)[^\s<>\"']+\.(?:pdf|docx)\b", re.IGNORECASE)
_WHOLE_SCOPE_WORD = re.compile(r"\b(?:whole|entire|complete|full|all|toan\s+bo|tat\s+ca)\b")


@dataclass(frozen=True)
class SummaryIntent:
    is_summary: bool
    whole_document: bool = False


def classify_summary_intent(query: str) -> SummaryIntent:
    """Separate whole-document summaries from focused summaries and normal QA."""
    text = fold(query)
    if not (_SUMMARY_ACTION.search(text) or _SUMMARY_NOUN_REQUEST.search(text)):
        return SummaryIntent(is_summary=False)
    if _WHOLE_DOCUMENT.search(text):
        return SummaryIntent(is_summary=True, whole_document=True)
    if _FOCUSED_SCOPE.search(text):
        return SummaryIntent(is_summary=True, whole_document=False)
    if _DOCUMENT_OBJECT.search(text):
        return SummaryIntent(is_summary=True, whole_document=True)

    # A filename identifies the document, not the requested scope. An
    # unqualified summary of that resolved document means whole-document;
    # named topics/sections remain focused.
    has_filename = bool(_FILENAME.search(text))
    if has_filename and _WHOLE_SCOPE_WORD.search(text):
        return SummaryIntent(is_summary=True, whole_document=True)

    remainder = _SUMMARY_ACTION.sub(" ", text)
    remainder = _FILENAME.sub(" ", remainder)
    remainder = re.sub(
        r"\b(?:the|this|that|uploaded|please|can|you|me|it|all|of|entire|whole|complete|full|summary|"
        r"document|file|pdf|book|tai lieu|tep|toan bo|tat ca)\b",
        " ", remainder,
    )
    return SummaryIntent(is_summary=True, whole_document=not bool(remainder.strip(" ?.,!")))
