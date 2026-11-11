"""Narrow guard against generated copies of internal prompt section headings."""
import re


_SAFE_OUTPUT_FALLBACKS = {
    "english": "I couldn't produce a safe response to that request. Please try again.",
    "vietnamese": "Tôi chưa thể tạo câu trả lời an toàn cho yêu cầu này. Vui lòng thử lại.",
    "spanish": "No pude generar una respuesta segura a esa solicitud. Inténtalo de nuevo, por favor.",
    "french": "Je n’ai pas pu produire une réponse sûre à cette demande. Veuillez réessayer.",
    "german": "Ich konnte keine sichere Antwort auf diese Anfrage erstellen. Bitte versuche es erneut.",
    "chinese": "我无法为此请求生成安全的回答。请重试。",
    "japanese": "このリクエストに対する安全な回答を生成できませんでした。もう一度お試しください。",
    "korean": "이 요청에 대한 안전한 답변을 생성하지 못했습니다. 다시 시도해 주세요.",
    "portuguese": "Não consegui gerar uma resposta segura para esta solicitação. Tente novamente.",
    "arabic": "تعذر إنشاء إجابة آمنة لهذا الطلب. يُرجى المحاولة مرة أخرى.",
    "hindi": "मैं इस अनुरोध के लिए सुरक्षित उत्तर नहीं बना सका। कृपया फिर से प्रयास करें।",
    "italian": "Non sono riuscito a generare una risposta sicura a questa richiesta. Riprova.",
    "russian": "Не удалось подготовить безопасный ответ на этот запрос. Попробуйте ещё раз.",
}
SAFE_OUTPUT_FALLBACK = _SAFE_OUTPUT_FALLBACKS["english"]
_KNOWN_INTERNAL_HEADINGS = {
    "PEDAGOGICAL MODE",
    "INSUFFICIENT EVIDENCE",
    "UNTRUSTED DATA DEFENSE",
}


def _internal_headings(system_instruction: str) -> set[str]:
    """Extract uppercase section labels from trusted system instructions."""
    headings: set[str] = set()
    for line in system_instruction.splitlines():
        candidate = line.strip()
        candidate = re.sub(r"^\d+\.\s*", "", candidate)
        if ":" in candidate:
            candidate = candidate.split(":", 1)[0].strip()
        if len(candidate) < 3 or not candidate.isupper():
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9 /_-]*", candidate):
            headings.add(candidate)
    return headings


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _instruction_signatures(system_instruction: str) -> set[str]:
    """Collect long, exact instruction lines for high-confidence copy detection."""
    signatures: set[str] = set()
    for line in system_instruction.splitlines():
        candidate = re.sub(r"^\s*(?:\d+\.\s*|[-*]\s*)", "", line).strip()
        normalized = _normalize(candidate)
        if len(normalized) >= 72:
            signatures.add(normalized)
    return signatures


def contains_internal_prompt_heading(text: str, system_instruction: str) -> bool:
    """Detect a prompt section label emitted as its own answer line.

    Ordinary prose that discusses a mode or evidence concept is intentionally
    allowed. Requiring a whole line match avoids broad keyword filtering.
    """
    headings = _internal_headings(system_instruction) | _KNOWN_INTERNAL_HEADINGS
    if not headings:
        return False

    normalized_text = _normalize(text)
    if any(signature in normalized_text for signature in _instruction_signatures(system_instruction)):
        return True

    for line in text.splitlines():
        candidate = line.strip().strip("#* `")
        if not candidate:
            continue
        for heading in headings:
            if candidate.casefold() == heading.casefold():
                return True
            # The mode label is followed by one of the three configured mode
            # names in a leaked prompt header. Keep other labels exact-only so
            # ordinary lines such as "INSUFFICIENT EVIDENCE: ..." remain valid.
            if heading == "PEDAGOGICAL MODE" and re.fullmatch(
                r"PEDAGOGICAL MODE:\s*(?:LIGHT GUIDANCE|DETAILED GUIDANCE|FULL SOLUTION)",
                candidate,
                re.IGNORECASE,
            ):
                return True
    return False


def guard_generated_output(
    text: str,
    system_instruction: str,
    response_language: str | None = None,
) -> tuple[str, bool]:
    """Replace a response that reproduces an internal prompt heading."""
    if contains_internal_prompt_heading(text, system_instruction):
        fallback = _SAFE_OUTPUT_FALLBACKS.get(
            (response_language or "").casefold(), SAFE_OUTPUT_FALLBACK
        )
        return fallback, True
    return text, False


def chunk_validated_output(text: str, max_chars: int = 32) -> list[str]:
    """Split already-validated content for the existing SSE token contract."""
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary + 1
        pieces.append(text[start:end])
        start = end
    return pieces
