import type { Citation } from "./api";

/**
 * Generates a deduplication key for a citation based on:
 * document_id + page_start + page_end
 */
export function getCitationKey(citation: {
  document_id?: string;
  document_name?: string;
  page_start: number;
  page_end: number;
}): string {
  const docIdentifier = citation.document_id || citation.document_name?.toLowerCase().trim() || "unknown";
  return `${docIdentifier}:${citation.page_start}:${citation.page_end}`;
}

export function formatCitationPages(pageStart: number, pageEnd: number): string {
  return pageStart === pageEnd
    ? `p.${pageStart}`
    : `pp.${pageStart}\u2013${pageEnd}`;
}

/**
 * Deduplicates an array of Citation items strictly by:
 * document_id + page_start + page_end
 */
export function deduplicateCitations(citations: Citation[] = []): Citation[] {
  const seen = new Set<string>();
  const result: Citation[] = [];

  for (const c of citations) {
    const key = getCitationKey(c);
    if (!seen.has(key)) {
      seen.add(key);
      result.push(c);
    }
  }

  return result;
}

export interface ParsedCitationLink {
  document_id?: string;
  document_name: string;
  page_start: number;
  page_end: number;
  chunk_id?: string;
  snippet?: string;
}

/**
 * Parses a citation:// URL back into its structured citation fields.
 */
export function parseCitationUrl(url: string): ParsedCitationLink | null {
  try {
    if (!url.startsWith("citation://")) return null;
    const queryString = url.slice("citation://open?".length);
    const params = new URLSearchParams(queryString);

    const docName = params.get("name") || "Document";
    const pageStart = parseInt(params.get("pageStart") || "1", 10);
    const pageEnd = parseInt(params.get("pageEnd") || String(pageStart), 10);

    return {
      document_id: params.get("docId") || undefined,
      document_name: docName,
      page_start: isNaN(pageStart) ? 1 : pageStart,
      page_end: isNaN(pageEnd) ? pageStart : pageEnd,
      chunk_id: params.get("chunkId") || undefined,
      snippet: params.get("snippet") || undefined,
    };
  } catch {
    return null;
  }
}

/**
 * Preprocesses markdown text containing raw citation patterns like:
 * - [Doc: filename.pdf, p. 9]
 * - [Doc: filename.pdf, pp. 9-11]
 * - [Source 1]
 * 
 * 1. Reduces citation density by aggregating citations at the end of paragraphs/claim groups.
 * 2. Deduplicates repeated citations in the same paragraph/claim group.
 * 3. Formats citations as compact inline source markers: [📄 p.9] or [📄 pp.9–10].
 * 4. Tracks all inline citation keys for parent deduplication against bottom evidence cards.
 */
export function preprocessMarkdownWithCitations(
  rawText: string,
  structuredCitations: Citation[] = []
): { processedText: string; inlineCitationKeys: Set<string> } {
  if (!rawText) return { processedText: "", inlineCitationKeys: new Set() };

  const inlineCitationKeys = new Set<string>();
  let processed = rawText;

  // 1. Resolve [Source X] references to structured sources if available
  const SOURCE_GROUP_REGEX =
    /\[((?:Source|Doc)\s*[:\-]?\s*\d+(?:\s*,\s*(?:(?:Source|Doc)\s*[:\-]?\s*)?\d+)*)\]/gi;
  const SOURCE_INDEX_REGEX = /(?:Source|Doc)?\s*[:\-]?\s*(\d+)/gi;
  processed = processed.replace(SOURCE_GROUP_REGEX, (_match, sourceGroup: string) => {
    const seen = new Set<string>();
    const markers: string[] = [];
    for (const sourceMatch of Array.from(sourceGroup.matchAll(SOURCE_INDEX_REGEX))) {
      const citation = structuredCitations[Number(sourceMatch[1]) - 1];
      if (!citation) continue;
      const key = getCitationKey(citation);
      if (seen.has(key)) continue;
      seen.add(key);
      const pageLabel =
        citation.page_start === citation.page_end
          ? `p. ${citation.page_start}`
          : `pp. ${citation.page_start}-${citation.page_end}`;
      markers.push(`[Doc: ${citation.document_name}, ${pageLabel}]`);
    }
    // Unresolved or hallucinated source IDs are suppressed instead of shown raw.
    return markers.join(" ");
  });

  const DOC_PAGE_REGEX =
    /\[(?:Doc:\s*)?([^,\]]+?)(?:,\s*|\s+)pp?\.?\s*(\d+)(?:\s*[-–—]\s*(\d+))?\]/gi;

  // Helper to process a single paragraph or claim group
  const processBlock = (block: string): string => {
    // Preserve code blocks without interpreting citation-like text as sources.
    if (block.trim().startsWith("```")) {
      return block;
    }
    const isTable = block.trim().startsWith("|");

    const matches = Array.from(block.matchAll(DOC_PAGE_REGEX));
    if (matches.length === 0) return block;

    const seenKeysInBlock = new Set<string>();
    const uniqueCitationsInBlock: Array<{
      docId: string;
      chunkId: string;
      docName: string;
      finalStart: number;
      finalEnd: number;
      snippet: string;
    }> = [];

    for (const match of matches) {
      const rawFilename = match[1].trim();
      const startPage = parseInt(match[2], 10);
      const endPage = match[3] ? parseInt(match[3], 10) : startPage;

      // Find matching structured citation to preserve UUIDs and chunk IDs
      const matched =
        structuredCitations.find((c) => {
          const nameMatches =
            c.document_name.toLowerCase().includes(rawFilename.toLowerCase()) ||
            rawFilename.toLowerCase().includes(c.document_name.toLowerCase());
          return nameMatches && c.page_start === startPage;
        }) ||
        structuredCitations.find((c) => {
          return (
            c.document_name.toLowerCase().includes(rawFilename.toLowerCase()) ||
            rawFilename.toLowerCase().includes(c.document_name.toLowerCase())
          );
        });

      const docId = matched?.document_id || "";
      const chunkId = matched?.chunk_id || "";
      const docName = matched?.document_name || rawFilename;
      const finalStart = matched?.page_start || startPage;
      const finalEnd = matched?.page_end || endPage;

      const citationKey = getCitationKey({
        document_id: docId || docName,
        document_name: docName,
        page_start: finalStart,
        page_end: finalEnd,
      });

      inlineCitationKeys.add(citationKey);

      if (!seenKeysInBlock.has(citationKey)) {
        seenKeysInBlock.add(citationKey);
        uniqueCitationsInBlock.push({
          docId,
          chunkId,
          docName,
          finalStart,
          finalEnd,
          snippet: matched?.snippet || "",
        });
      }
    }

    // Strip inline citation markers from inside sentences
    let cleanedText = block.replace(DOC_PAGE_REGEX, "");
    cleanedText = cleanedText.replace(/[ \t]{2,}/g, " ");
    cleanedText = cleanedText.replace(/\s+([.,!?;:])/g, "$1");

    // Format compact citation links: [📄 p.9] or [📄 pp.9–10]
    const citationLinks = uniqueCitationsInBlock.map((c) => {
      const pageBadge =
        c.finalStart === c.finalEnd
          ? `p.${c.finalStart}`
          : `pp.${c.finalStart}–${c.finalEnd}`;
      const badgeText = `📄 ${pageBadge}`;

      const params = new URLSearchParams();
      if (c.docId) params.set("docId", c.docId);
      params.set("name", c.docName);
      params.set("pageStart", String(c.finalStart));
      params.set("pageEnd", String(c.finalEnd));
      if (c.chunkId) params.set("chunkId", c.chunkId);
      if (c.snippet) params.set("snippet", c.snippet);

      return `[${badgeText}](citation://open?${params.toString()})`;
    });

    const trimmed = cleanedText.trimEnd();
    if (isTable) {
      return citationLinks.length > 0
        ? `${trimmed}\n\n${citationLinks.join(" ")}`
        : trimmed;
    }

    return trimmed ? `${trimmed} ${citationLinks.join(" ")}` : citationLinks.join(" ");
  };

  // Split into paragraphs / claim blocks preserving markdown block separators
  const paragraphs = processed.split(/(\n\s*\n)/);
  const transformed = paragraphs.map((block) => {
    // Preserve blank line delimiters
    if (/^\n\s*\n$/.test(block)) return block;

    // Check if block contains list items (process list items line-by-line)
    const lines = block.split("\n");
    const isList = lines.some((l) => /^\s*[-*+]\s+|^\s*\d+\.\s+/.test(l));
    if (isList) {
      return lines
        .map((line) => {
          if (/^\s*[-*+]\s+|^\s*\d+\.\s+/.test(line)) {
            return processBlock(line);
          }
          return line;
        })
        .join("\n");
    }

    return processBlock(block);
  });

  return {
    processedText: transformed.join(""),
    inlineCitationKeys,
  };
}
