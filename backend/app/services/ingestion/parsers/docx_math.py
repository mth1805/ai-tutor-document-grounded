"""Ordered Word/OMML text, shared by ingestion and preview verification.

This is a text conversion, not a layout engine. Unknown constructs retain their
visible descendants and emit a content-free warning instead of disappearing.
"""
import logging
import re
from dataclasses import dataclass

from lxml import etree

logger = logging.getLogger(__name__)
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_ATOM = re.compile(r"(?:[\w.]+|\([^()]+\))\Z", re.UNICODE)
_SYMBOLS = str.maketrans({"−": "-", "–": "-", "×": "*", "⋅": "*", "÷": "/"})


@dataclass(frozen=True)
class EquationText:
    text: str
    supported: bool


def _operand(text: str) -> str:
    return text if text == "?" or _ATOM.fullmatch(text) else f"({text})"


def read_equation(element: etree._Element) -> EquationText:
    """Linearize fractions, powers, subscripts, roots, delimiters and math runs."""
    supported = True

    def warn(tag: str) -> None:
        nonlocal supported
        supported = False
        logger.warning("docx_math_fallback node=%s reason=unsupported_or_incomplete", tag)

    def children(node: etree._Element) -> str:
        return "".join(read(child) for child in node)

    def part(node: etree._Element, name: str) -> str:
        child = node.find(f"{{{MATH}}}{name}")
        return read(child).strip() if child is not None else ""

    def read(node: etree._Element) -> str:
        if not isinstance(node.tag, str):
            return ""
        tag = etree.QName(node).localname
        namespace = etree.QName(node).namespace
        if namespace == WORD:
            return (node.text or "") if tag == "t" else children(node)
        if namespace != MATH:
            warn("foreign")
            return children(node) or "[unsupported equation]"
        # Properties contain formatting, not displayed equation text.
        if tag.endswith("Pr") or tag == "ctrlPr":
            return ""
        if tag == "t":
            return (node.text or "").translate(_SYMBOLS)
        if tag in {"oMath", "oMathPara", "r", "e", "num", "den", "sup", "sub", "deg", "fName"}:
            return children(node)
        if tag == "f":
            num, den = part(node, "num"), part(node, "den")
            if not num or not den:
                warn(tag)
            return f"{_operand(num or '?')}/{_operand(den or '?')}"
        if tag in {"sSup", "sSub", "sSubSup"}:
            base = part(node, "e")
            sub, sup = part(node, "sub"), part(node, "sup")
            if not base or (tag != "sSup" and not sub) or (tag != "sSub" and not sup):
                warn(tag)
            result = _operand(base or "?")
            if tag != "sSup":
                result += "_" + _operand(sub or "?")
            if tag != "sSub":
                result += "^" + _operand(sup or "?")
            return result
        if tag == "rad":
            base, degree = part(node, "e"), part(node, "deg")
            if not base:
                warn(tag)
            return f"root({degree},{base or '?'})" if degree else f"sqrt({base or '?'})"
        if tag == "d":
            props = node.find(f"{{{MATH}}}dPr")
            def delimiter(name: str, default: str) -> str:
                item = props.find(f"{{{MATH}}}{name}") if props is not None else None
                return item.get(f"{{{MATH}}}val", default) if item is not None else default
            values = [read(child) for child in node if child.tag == f"{{{MATH}}}e"]
            return delimiter("begChr", "(") + delimiter("sepChr", "|").join(values) + delimiter("endChr", ")")
        if tag == "func":
            return f"{part(node, 'fName')}({part(node, 'e')})"
        warn(tag)
        # Separate arguments so unsupported structures do not merge numbers.
        visible = [read(child) for child in node]
        symbols = [child.get(f"{{{MATH}}}val", "") for child in node.iter(f"{{{MATH}}}chr")]
        retained = " ".join(value for value in symbols + visible if value)
        return f"[unsupported {tag}: {retained}]" if retained else "[unsupported equation]"

    text = read(element).strip()
    if not text:
        warn("empty")
        text = "[unsupported equation]"
    return EquationText(text, supported)


def ordered_text(element: etree._Element) -> str:
    """Read displayed text and inline math in XML order (also inside hyperlinks)."""
    if not isinstance(element.tag, str):
        return ""
    tag = etree.QName(element).localname
    namespace = etree.QName(element).namespace
    if namespace == MATH and tag in {"oMath", "oMathPara"}:
        return read_equation(element).text
    if namespace == WORD:
        if tag == "t":
            return element.text or ""
        if tag in {"tab", "br", "cr"}:
            return "\t" if tag == "tab" else "\n"
        if tag in {"pPr", "rPr", "del", "instrText"}:
            return ""
    return "".join(ordered_text(child) for child in element)


def document_equations(body: etree._Element) -> list[EquationText]:
    """One entry per equation; oMathPara is a wrapper, not a second equation."""
    return [read_equation(node) for node in body.iter(f"{{{MATH}}}oMath")]
