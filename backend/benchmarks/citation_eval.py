"""Deterministic comparison of output citation metadata to gold sources."""
from __future__ import annotations


def evaluate_citations(citations: list[dict], gold_documents: list[dict],
                       gold_web_urls: list[str] | None = None,
                       expected_fact_ids: list[str] | None = None) -> dict:
    docs = {(str(item["document_id"]), int(item["page_start"]), int(item["page_end"]))
            for item in gold_documents}
    web_urls = set(gold_web_urls or [])
    seen: set[tuple] = set()
    valid = 0
    malformed = 0
    covered_facts: set[str] = set()
    for citation in citations:
        try:
            if citation.get("source_type") == "web" or "url" in citation:
                url = citation["url"]
                title = citation["title"]
                domain = citation["domain"]
                if not all(isinstance(v, str) and v for v in (url, title, domain)):
                    raise ValueError("invalid web citation")
                key = ("web", url)
                is_valid = url in web_urls
            else:
                doc_id = str(citation["document_id"])
                start, end = int(citation["page_start"]), int(citation["page_end"])
                if start < 1 or end < start:
                    raise ValueError("invalid document page range")
                key = ("document", doc_id, start, end)
                is_valid = any(did == doc_id and start <= page_end and end >= page_start
                               for did, page_start, page_end in docs)
                covered_facts.update(citation.get("fact_ids", []))
            if key in seen:
                continue
            seen.add(key)
            valid += int(is_valid)
        except (KeyError, TypeError, ValueError):
            malformed += 1
    unique = len(seen)
    expected = set(expected_fact_ids or [])
    return {
        "citation_count": unique,
        "correct_count": valid,
        "malformed_count": malformed,
        "citation_correctness": valid / unique if unique else None,
        "citation_completeness": (len(expected & covered_facts) / len(expected)) if expected else None,
    }
