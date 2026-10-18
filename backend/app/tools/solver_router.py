"""Cheap bilingual intent routing; no model calls for ordinary questions."""
import re
import unicodedata


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text).lower().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


EXERCISE_REFERENCE = re.compile(r"\b(?:bai(?:\s+tap)?|cau|exercise|problem|question)\s*(\d+)\b")


class SolverRouter:
    @staticmethod
    def is_document_reference(query: str) -> bool:
        text = fold(query)
        return bool(EXERCISE_REFERENCE.search(text) or
                    re.search(r"\b(?:this equation|this exercise|this problem|phuong trinh nay|bai nay|in the document|trong tai lieu)\b", text))

    @staticmethod
    def route(query: str) -> str:
        text = fold(query).strip()
        if re.match(r"^(?:tom tat|summari[sz]e|giai thich|explain|what is|define)\b", text):
            return "normal"
        math_command = re.search(
            r"\b(?:solve|calculate|compute|evaluate|differentiate|integrate|simplify|"
            r"derivative|integral|giai phuong trinh|dao ham|tich phan|nguyen ham|rut gon|tinh)\b", text)
        exercise_command = re.search(r"\b(?:giai|solve|help)\b", text) and SolverRouter.is_document_reference(query)
        bare_expression = re.fullmatch(r"[\d\sA-Za-z.+*/^()=\-]+", text) and re.search(r"[+*/^=]|\d\s*-\s*\d", text)
        return "math_problem" if math_command or exercise_command or bare_expression else "normal"
