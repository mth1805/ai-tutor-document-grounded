"""Deterministic text cleaning and normalization for extracted document text."""
import re
import unicodedata


# Control characters to strip (keeps \t, \n, \r)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Regex to safely rejoin hyphenated words split across a newline
# Example: "infor-\nmation" -> "information"
_DEHYPHEN_RE = re.compile(r"(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)")

# Regex for 3+ consecutive newlines
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalizes and cleans raw extracted text deterministically without altering semantic content.

    Rules applied:
    1. Unicode normalization (NFKC).
    2. Normalize line endings (\\r\\n and \\r -> \\n).
    3. Remove non-printable control characters (excluding \\n, \\t).
    4. Safely rejoin words split across linebreaks with hyphens.
    5. Normalize horizontal whitespace per line (collapse redundant spaces while preserving indent).
    6. Collapse 3+ consecutive blank lines to at most 2 newlines (standard paragraph gap).
    7. Preserve headings, bullets, code blocks, and punctuation.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Line ending normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip control characters
    text = _CONTROL_CHAR_RE.sub("", text)

    # 4. De-hyphenate broken words across linebreaks
    text = _DEHYPHEN_RE.sub(r"\1\2", text)

    # 5. Normalize whitespace per line while preserving indentation
    cleaned_lines = []
    for line in text.split("\n"):
        # Detect leading whitespace (indentation)
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Count leading spaces or tabs to preserve bullet / code structure
        leading_spaces = len(line) - len(line.lstrip(" "))
        indent = " " * min(leading_spaces, 8)  # Cap indentation reasonably

        # Collapse multiple horizontal spaces in the content portion
        content = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(f"{indent}{content}")

    text = "\n".join(cleaned_lines)

    # 6. Collapse excessive blank lines (max 2 newlines between paragraphs)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)

    return text.strip()
