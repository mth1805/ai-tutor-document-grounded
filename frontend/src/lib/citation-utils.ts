import type { Citation } from "./api";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

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
    if (!url.startsWith("citation://open?")) return null;
    const queryString = url.slice("citation://open?".length);
    const params = new URLSearchParams(queryString);

    const docName = params.get("name")?.trim();
    const start = params.get("pageStart") || "";
    const end = params.get("pageEnd") || start;
    if (!docName || !/^\d+$/.test(start) || !/^\d+$/.test(end)) return null;
    const pageStart = Number(start);
    const pageEnd = Number(end);
    if (!validPages(pageStart, pageEnd)) return null;

    return {
      document_id: params.get("docId") || undefined,
      document_name: docName,
      page_start: pageStart,
      page_end: pageEnd,
      chunk_id: params.get("chunkId") || undefined,
      snippet: params.get("snippet") || undefined,
    };
  } catch {
    return null;
  }
}

function validPages(start: number, end: number): boolean {
  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start > 0 && end >= start;
}

const markdownParser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);
const normalizeName = (name: string) => name.normalize("NFC").trim().toLowerCase();

interface MarkdownNode {
  type: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: MarkdownNode[];
}

/**
 * Replace citations at their original source offsets. Parsing is used only to
 * protect code, math, links and other non-prose nodes; Markdown is never
 * serialized again, so table cells, whitespace and formatting stay intact.
 * Only adjacent duplicate markers are collapsed: separate claims keep evidence.
 */
export function preprocessMarkdownWithCitations(
  rawText: string,
  structuredCitations: Citation[] = []
): { processedText: string; inlineCitationKeys: Set<string> } {
  const inlineCitationKeys = new Set<string>();
  if (!rawText) return { processedText: "", inlineCitationKeys };

  const protectedRanges: Array<[number, number]> = [];
  const protectedTypes = new Set([
    "code", "inlineCode", "math", "inlineMath", "link", "linkReference",
    "image", "imageReference", "definition", "html",
  ]);
  const collectProtected = (node: MarkdownNode) => {
    if (protectedTypes.has(node.type)) {
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (start !== undefined && end !== undefined) protectedRanges.push([start, end]);
      return;
    }
    node.children?.forEach(collectProtected);
  };
  collectProtected(markdownParser.parse(rawText));

  // The page suffix delimits the filename, which may itself contain commas,
  // Unicode, Markdown punctuation or bracketed qualifiers.
  const markerPattern = /\[(?:Doc:\s*)?((?:[^\[\]\r\n]|\[[^\]\r\n]*\])+?)(?:,\s*|\s+)pp?\.?\s*(\d+)(?:\s*[-–—]\s*(\d+))?\]|\[((?:Source|Doc)\s*[:\-]?\s*\d+(?:\s*,\s*(?:(?:Source|Doc)\s*[:\-]?\s*)?\d+)*)\]/gi;
  let cursor = 0;
  let processedText = "";
  let adjacentKeys = new Set<string>();

  const makeLink = (citation: ParsedCitationLink, evidenceKey = getCitationKey(citation)) => {
    const key = getCitationKey(citation);
    inlineCitationKeys.add(evidenceKey);
    if (adjacentKeys.has(key)) return "";
    adjacentKeys.add(key);
    const params = new URLSearchParams();
    if (citation.document_id) params.set("docId", citation.document_id);
    params.set("name", citation.document_name);
    params.set("pageStart", String(citation.page_start));
    params.set("pageEnd", String(citation.page_end));
    if (citation.chunk_id) params.set("chunkId", citation.chunk_id);
    if (citation.snippet) params.set("snippet", citation.snippet);
    return `[📄 ${formatCitationPages(citation.page_start, citation.page_end)}](citation://open?${params})`;
  };

  for (const match of Array.from(rawText.matchAll(markerPattern))) {
    const start = match.index;
    const end = start + match[0].length;
    const escapes = rawText.slice(0, start).match(/\\+$/)?.[0].length || 0;
    if (escapes % 2) continue;
    if (protectedRanges.some(([a, b]) => start < b && end > a)) continue;
    const between = rawText.slice(cursor, start);
    if (!/^[ \t]*$/.test(between)) adjacentKeys = new Set();

    let replacement: string;
    if (match[4]) {
      const markers: string[] = [];
      for (const source of Array.from(match[4].matchAll(/\d+/g))) {
        const citation = structuredCitations[Number(source[0]) - 1];
        if (citation?.document_id && citation.chunk_id &&
            validPages(citation.page_start, citation.page_end)) {
          const link = makeLink(citation);
          if (link) markers.push(link);
        }
      }
      // Unresolved source indices cannot identify a real document/page.
      replacement = markers.join(" ");
    } else {
      const name = match[1].trim();
      const pageStart = Number(match[2]);
      const pageEnd = Number(match[3] || match[2]);
      if (!validPages(pageStart, pageEnd)) continue;
      const candidates = structuredCitations.filter(c =>
        normalizeName(c.document_name) === normalizeName(name) &&
        c.document_id && c.chunk_id && validPages(c.page_start, c.page_end) &&
        c.page_start <= pageStart && c.page_end >= pageEnd
      );
      // A filename alone cannot disambiguate two different documents.
      if (structuredCitations.length &&
          (!candidates.length || new Set(candidates.map(c => c.document_id)).size > 1)) continue;
      const matched = candidates.find(c => c.page_start === pageStart && c.page_end === pageEnd)
        || candidates[0];
      replacement = makeLink({
        document_id: matched?.document_id,
        document_name: matched?.document_name || name,
        page_start: pageStart,
        page_end: pageEnd,
        chunk_id: matched?.chunk_id,
        snippet: matched?.snippet,
      }, matched ? getCitationKey(matched) : undefined);
    }
    processedText += between + replacement;
    cursor = end;
  }

  return { processedText: processedText + rawText.slice(cursor), inlineCitationKeys };
}
