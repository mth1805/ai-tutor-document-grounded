"""Conservative extraction of elementary tasks from user text or retrieved exercises.

Unknown prose, multiple tasks and incomplete statements are rejected rather
than guessed. No LLM-generated code or unvalidated fields enter the solver.
"""
import re
import unicodedata

from app.schemas.math_solver import MathTask
from app.schemas.retrieval import RetrievedChunk
from app.tools.math_solver import MathInputError, normalize_expression, parse_expression
from app.tools.solver_router import EXERCISE_REFERENCE, fold


class MathTaskParser:
    @staticmethod
    def parse(text: str, source: str = "user_question") -> MathTask:
        text = unicodedata.normalize("NFC", text)
        normalized = fold(text).strip().rstrip(".?!")
        if re.search(r"\b(?:differentiat\w*|derivative|dao ham)\b", normalized):
            operation = "differentiate"
        elif re.search(r"\b(?:integrat\w*|integral|tich phan|nguyen ham)\b", normalized):
            operation = "integrate"
        elif re.search(r"\b(?:simplify|simplification|rut gon)\b", normalized):
            operation = "simplify"
        elif "=" in normalized or re.search(r"\b(?:solve|giai)\b", normalized):
            operation = "solve_equation"
        else:
            operation = "calculate"

        # Folded text and original have equal character counts for ordinary NFC
        # Vietnamese. Work on folded prose, retaining the expression's case.
        original = text.strip().rstrip(".?!")
        variable = None
        match = re.search(r"\s+(?:with respect to|wrt|for|theo(?: bien)?)\s+([a-z])\s*$", normalized)
        if match:
            variable = original[match.start(1):match.end(1)]
            original = original[:match.start()]
            normalized = normalized[:match.start()]
        prefix = re.match(
            r"^(?:(?:please|hay)\s+)?(?:"
            r"(?:tinh\s+)?(?:dao ham|tich phan|nguyen ham)(?:\s+cua)?|"
            r"(?:calculate|compute|find)\s+(?:the\s+)?(?:derivative|integral)(?:\s+of)?|"
            r"giai(?:\s+phuong trinh)?|solve(?:\s+(?:the\s+)?equation)?|"
            r"differentiate|integrate|simplify|rut gon|calculate|compute|evaluate|tinh"
            r")\s*:?\s*", normalized)
        if prefix:
            original = original[prefix.end():].strip()
        # A function declaration is data, not an additional equation.
        if operation in {"differentiate", "integrate"}:
            original = re.sub(r"^[a-zA-Z]\([a-zA-Z]\)\s*=\s*", "", original)
        if not original or "\n" in original or ";" in original:
            raise MathInputError("A single complete expression is required")
        fields = {"operation": operation, "source": source, "variable": variable, "confidence": 1.0}
        if operation == "solve_equation":
            if original.count("=") != 1:
                raise MathInputError("A single equation is required")
            left, right = original.split("=")
            fields.update(left=normalize_expression(left), right=normalize_expression(right))
            parse_expression(fields["left"])
            parse_expression(fields["right"])
        else:
            fields["expression"] = normalize_expression(original)
            parse_expression(fields["expression"])
        return MathTask.model_validate(fields)

    @classmethod
    def from_retrieved(cls, query: str, chunks: list[RetrievedChunk]) -> MathTask:
        """Only read gate-approved chunks, and adjacent pieces of one document.

        Numbered requests must match the exact exercise label. Other retrieved
        exercises cannot substitute for a missing requested statement.
        """
        chunks = [chunk for chunk in chunks if chunk.passed_relevance_gate]
        reference = EXERCISE_REFERENCE.search(fold(query))
        number = reference.group(1) if reference else None
        candidates = []
        by_document = {}
        for chunk in chunks:
            by_document.setdefault(chunk.document_id, []).append(chunk)
        for document_chunks in by_document.values():
            ordered = sorted(document_chunks, key=lambda chunk: chunk.chunk_index)
            groups = []
            for chunk in ordered:
                if not groups or chunk.chunk_index != groups[-1][-1].chunk_index + 1:
                    groups.append([])
                groups[-1].append(chunk)
            for group in groups:
                # Bound confidential context passed to extraction, retaining provenance
                # in the original retrieved chunk list for citation assembly.
                statement = "\n".join(unicodedata.normalize("NFC", c.content) for c in group)
                if len(statement) > 12000:
                    continue
                labels = list(EXERCISE_REFERENCE.finditer(fold(statement)))
                def selected_chunks(start, end):
                    offset = 0
                    ids = []
                    for item in group:
                        chunk_end = offset + len(unicodedata.normalize("NFC", item.content))
                        if offset < end and chunk_end > start:
                            ids.append(item.chunk_id)
                        offset = chunk_end + 1
                    return ids
                if number:
                    matched = [label for label in labels if label.group(1) == number]
                    for label in matched:
                        next_start = next((item.start() for item in labels if item.start() > label.start()), len(statement))
                        candidates.append((statement[label.end():next_start].lstrip(" .:)-\n"), selected_chunks(label.start(), next_start)))
                elif labels:
                    for index, label in enumerate(labels):
                        end = labels[index + 1].start() if index + 1 < len(labels) else len(statement)
                        candidates.append((statement[label.end():end].lstrip(" .:)-\n"), selected_chunks(label.start(), end)))
                else:
                    candidates.append((statement, [item.chunk_id for item in group]))
        tasks = {}
        for candidate, source_chunk_ids in candidates:
            # Join a statement split across adjacent retrieved chunks. Reject multiple
            # equations/operations below instead of picking an arbitrary first line.
            try:
                task = cls.parse(" ".join(candidate.split()), source="retrieved_document")
                key = task.model_dump_json()
                previous = tasks.get(key)
                ids = list(dict.fromkeys((previous.source_chunk_ids if previous else []) + source_chunk_ids))
                tasks[key] = task.model_copy(update={"source_chunk_ids": ids})
            except (ValueError, TypeError):
                continue
        if len(tasks) != 1:
            raise MathInputError("No single relevant, complete exercise was retrieved")
        return next(iter(tasks.values()))
