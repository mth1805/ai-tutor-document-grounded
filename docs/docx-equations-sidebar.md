# DOCX equations and sidebar controls

DOCX ingestion reads the document XML in order rather than relying on
`python-docx` paragraph/cell text, which omits Office Math (OMML). Inline
`m:oMath`, block `m:oMathPara`, and equations inside table paragraphs share the
same reader. Vietnamese surrounding text, headings, and existing logical page
provenance are retained.

Fractions use plain text (`4/25`); complex operands are grouped
(`(x+1)/(y-2)` and `(1/2)/3`). Powers use `x^2`. Math runs normalize common
arithmetic symbols. Subscripts, roots, delimiters, and simple function notation
are also readable. Unsupported structures retain their available descendants,
show an unsupported marker, and log the node category without document content.
This is a bounded conversion of common OMML constructs, not a complete Word
equation renderer or a symbolic evaluator. Equations stored only in images or
legacy embedded equation objects do not gain OMML support from this change.

The existing cleaner, chunker, persistence, embedding, and hybrid retrieval
consume this text without a new chunking strategy. Previously ingested DOCX
files need reprocessing to replace chunks and embeddings that lost equations.
The math solver and relevance/citation rules are unchanged.

Word preview first attempts the existing isolated LibreOffice PDF conversion.
For DOCX containing OMML, the converted PDF must contain extractable canonical
equation text before it is accepted. Unsupported equations, unreadable PDF text,
or a conversion failure select a fresh text preview from the original DOCX.
The original download remains unchanged. Stale persisted DOCX chunks are not
used for this fallback. Legacy DOC retains its existing chunk fallback, and
PDF/TXT/image behavior is unchanged.

This check is conservative: correctly drawn stacked fractions may not have
canonical PDF text, so a readable inline text preview can replace a visually
correct native PDF. The fallback is labeled in the viewer and does not preserve
Word pagination, fonts, or stacked mathematical layout. DOCX citations continue
to use logical page 1, as before. Microsoft Word is not required.

The sidebar uses the existing Lucide icons and color palette. A native button
collapses its width to zero and remains visible at the left edge for restoring
navigation. Hidden contents stay mounted, preserving workspace/tab/chat/upload
state. The main flex panels can shrink and expand without a reserved sidebar
column; document split/resize and mobile viewer/chat tabs keep their existing
behavior. The expanded sidebar is capped at 80% of a small viewport.

`ai-tutor-sidebar-collapsed` stores only a boolean preference in localStorage.
The server and initial client render start expanded; an effect restores the
preference after mounting. If storage is blocked, toggling still works for the
session. `aria-expanded`, `aria-controls`, labels, title, focus styling, and native
Enter/Space button activation are provided. Reduced-motion settings disable the
width transition.

Regression tests generate a real zipped DOCX fixture with the six requested
fractions, Vietnamese paragraphs, a block equation, and a table equation. They
exercise extraction, chunking, mock CPU embedding, hybrid retrieval, unsupported
nodes, native-PDF acceptance, conversion failures, missing/stacked math fallback,
private preview access, and original download integrity. Frontend tests exercise
toggle handlers and rendered elements, persistence/hydration, storage failure,
and existing mobile/split behavior. These tests do not call production services.
