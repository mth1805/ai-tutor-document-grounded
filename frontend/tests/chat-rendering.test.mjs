import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import ReactDOMServer from 'react-dom/server';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { readFileSync } from 'node:fs';

// Import citation utilities from src
import {
  getCitationKey,
  deduplicateCitations,
  preprocessMarkdownWithCitations,
  parseCitationUrl,
  formatCitationPages,
} from '../src/lib/citation-utils.ts';

// Helper to render markdown to HTML string with identical plugins as MarkdownRenderer
function renderMarkdown(content, structuredCitations = [], onOpen = null) {
  const { processedText } = preprocessMarkdownWithCitations(content, structuredCitations);

  const customUrlTransform = (url) => {
    if (url.startsWith('citation://')) return url;
    return defaultUrlTransform(url);
  };

  return ReactDOMServer.renderToString(
    React.createElement(
      ReactMarkdown,
      {
        urlTransform: customUrlTransform,
        remarkPlugins: [remarkGfm, remarkMath],
        rehypePlugins: [[rehypeKatex, { throwOnError: false }]],
        components: {
          a: ({ href, children }) => {
            if (href?.startsWith('citation://')) {
              const parsed = parseCitationUrl(href);
              const pageBadge =
                parsed?.page_start === parsed?.page_end
                  ? `p.${parsed?.page_start}`
                  : `pp.${parsed?.page_start}–${parsed?.page_end}`;
              const tooltipPage =
                parsed?.page_start === parsed?.page_end
                  ? `Page ${parsed?.page_start}`
                  : `Pages ${parsed?.page_start}–${parsed?.page_end}`;
              return React.createElement(
                'span',
                { className: 'citation-wrapper', 'data-testid': 'citation-wrapper' },
                React.createElement(
                  'button',
                  {
                    type: 'button',
                    'data-testid': 'citation-badge',
                    'data-page': pageBadge,
                    'aria-label': `${parsed?.document_name}, ${tooltipPage}`,
                    className: 'citation-marker',
                  },
                  `📄 ${pageBadge}`
                ),
                React.createElement(
                  'span',
                  { role: 'tooltip', 'data-testid': 'citation-tooltip' },
                  React.createElement('span', { 'data-testid': 'tooltip-name' }, parsed?.document_name),
                  React.createElement('span', { 'data-testid': 'tooltip-page' }, tooltipPage)
                )
              );
            }
            return React.createElement('a', { href, target: '_blank' }, children);
          },
        },
      },
      processedText
    )
  );
}

describe('1. Markdown Rendering', () => {
  it('renders headings (h1, h2, h3)', () => {
    const md = '# Title 1\n## Title 2\n### 3. Ví dụ';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<h1>Title 1</h1>'));
    assert.ok(html.includes('<h2>Title 2</h2>'));
    assert.ok(html.includes('<h3>3. Ví dụ</h3>'));
  });

  it('renders bold and italic text', () => {
    const md = 'This is **bold text** and *italic text*.';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<strong>bold text</strong>'));
    assert.ok(html.includes('<em>italic text</em>'));
  });

  it('renders bullet and numbered lists', () => {
    const md = '- Item A\n- Item B\n\n1. Step One\n2. Step Two';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<ul>'));
    assert.ok(html.includes('<li>Item A</li>'));
    assert.ok(html.includes('<ol>'));
    assert.ok(html.includes('<li>Step One</li>'));
  });

  it('renders code blocks and inline code', () => {
    const md = 'Inline `const x = 10;`\n\n```python\ndef greet():\n    return "hello"\n```';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<code>const x = 10;</code>'));
    assert.ok(html.includes('language-python'));
    assert.ok(html.includes('def greet():'));
  });

  it('renders GFM tables', () => {
    const md = '| Feature | Support |\n|---|---|\n| Markdown | Yes |\n| KaTeX | Yes |';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<table>'));
    assert.ok(html.includes('<th>Feature</th>'));
    assert.ok(html.includes('<td>Markdown</td>'));
  });

  it('keeps table citations out of cells and renders a single compact marker after the table', () => {
    const md = '| Topic | Evidence |\n|---|---|\n| Attention | [doc: lecture.pdf p.20] |\n| Heads | [doc: lecture.pdf p.20] |';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<table>'));
    assert.ok(!html.includes('[doc: lecture.pdf p.20]'));
    assert.equal((html.match(/data-testid="citation-badge"/g) || []).length, 1);
    assert.ok(html.indexOf('</table>') < html.indexOf('data-testid="citation-badge"'));
    assert.ok(html.includes('p.20'));
  });

  it('renders blockquotes', () => {
    const md = '> Grounded evidence is the primary source of truth.';
    const html = renderMarkdown(md);
    assert.ok(html.includes('<blockquote>'));
    assert.ok(html.includes('Grounded evidence is the primary source of truth.'));
  });
});

describe('2. LaTeX Rendering', () => {
  it('renders inline LaTeX math $d_{model}$', () => {
    const md = 'Kích thước **$d_{model}$ (Kích thước biểu diễn):** là 512.';
    const html = renderMarkdown(md);
    assert.ok(html.includes('class="katex"'));
    assert.ok(html.includes('d_{model}') || (html.includes('d') && html.includes('model')));
    assert.ok(html.includes('<strong>'));
  });

  it('renders display LaTeX math $$...$$', () => {
    const md = '$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$';
    const html = renderMarkdown(md);
    assert.ok(html.includes('katex'));
    assert.ok(html.includes('Attention'));
    assert.ok(html.includes('softmax'));
  });

  it('does not throw or break when rendering partial/incomplete LaTeX during streaming', () => {
    const partialMd = 'The formula is $$\\text{Attention}(Q, ';
    assert.doesNotThrow(() => {
      const html = renderMarkdown(partialMd);
      assert.ok(typeof html === 'string');
    });
  });
});

describe('3. Citation Rendering', () => {
  it('renders [Doc: filename, p. X] as a compact inline citation with tooltip', () => {
    const md = 'Transformer sử dụng cơ chế Self-Attention [Doc: B_i 6. Transformer Ver2.pdf, p. 9].';
    const structured = [
      {
        document_id: 'doc-1234',
        document_name: 'B_i 6. Transformer Ver2.pdf',
        chunk_id: 'chunk-5678',
        page_start: 9,
        page_end: 9,
        snippet: 'Transformer architecture overview',
      },
    ];

    const html = renderMarkdown(md, structured);
    // Verifies raw [Doc: ...] is removed from text
    assert.ok(!html.includes('[Doc: B_i 6. Transformer Ver2.pdf, p. 9]'));
    // Compact inline badge: 📄 p.9
    assert.ok(html.includes('data-testid="citation-badge"'));
    assert.ok(html.includes('data-page="p.9"'));
    assert.ok(html.includes('📄 p.9'));
    // Filename should NOT be directly in the button text
    const buttonHtml = html.match(/<button[^>]*>([\s\S]*?)<\/button>/)?.[1] || '';
    assert.ok(!buttonHtml.includes('B_i 6. Transformer Ver2.pdf'));
    // Filename and Page are present in the tooltip
    assert.ok(html.includes('data-testid="citation-tooltip"'));
    assert.ok(html.includes('B_i 6. Transformer Ver2.pdf'));
    assert.ok(html.includes('Page 9'));
  });

  it('renders page ranges as 📄 pp.9–12', () => {
    const md = 'Multi-Head Attention details [Doc: Transformer.pdf, pp. 9-12].';
    const html = renderMarkdown(md);
    assert.ok(html.includes('📄 pp.9–12'));
  });

  it('maps [Source 1] to structured citations as compact marker', () => {
    const md = 'Positional encodings provide order [Source 1].';
    const structured = [
      {
        document_id: 'doc-999',
        document_name: 'Attention_Is_All_You_Need.pdf',
        chunk_id: 'chunk-111',
        page_start: 4,
        page_end: 4,
      },
    ];
    const html = renderMarkdown(md, structured);
    assert.ok(html.includes('📄 p.4'));
    assert.ok(html.includes('Attention_Is_All_You_Need.pdf'));
  });

  it('maps [Source 2] to its structured citation without showing the raw marker', () => {
    const structured = [
      { document_id: 'doc-1', document_name: 'Other.pdf', chunk_id: 'c1', page_start: 2, page_end: 2 },
      { document_id: 'doc-2', document_name: 'Lecture.pdf', chunk_id: 'c2', page_start: 20, page_end: 20 },
    ];
    const html = renderMarkdown('Claim [Source 2].', structured);
    assert.ok(!html.includes('[Source 2]'));
    assert.ok(html.includes('p.20'));
    assert.ok(html.includes('Lecture.pdf'));
  });

  it('resolves grouped source IDs to compact citations and deduplicates identical pages', () => {
    const samePage = [
      { document_id: 'doc-1', document_name: 'Other.pdf', chunk_id: 'c1', page_start: 1, page_end: 1 },
      { document_id: 'doc-2', document_name: 'Lecture.pdf', chunk_id: 'c2', page_start: 20, page_end: 20 },
      { document_id: 'doc-2', document_name: 'Lecture.pdf', chunk_id: 'c3', page_start: 20, page_end: 20 },
    ];
    const html = renderMarkdown('Claim [Source 2, Source 3].', samePage);
    assert.ok(!html.includes('[Source 2, Source 3]'));
    assert.equal((html.match(/data-testid="citation-badge"/g) || []).length, 1);
    assert.ok(html.includes('p.20'));

    const differentPages = samePage.map((citation, index) =>
      index === 2 ? { ...citation, page_start: 21, page_end: 21 } : citation
    );
    const twoPageHtml = renderMarkdown('Claim [Source 2, Source 3].', differentPages);
    assert.ok(!twoPageHtml.includes('[Source 2, Source 3]'));
    assert.ok(twoPageHtml.includes('p.20'));
    assert.ok(twoPageHtml.includes('p.21'));
    assert.equal((twoPageHtml.match(/data-testid="citation-badge"/g) || []).length, 2);
  });

  it('resolves grouped source markers in tables and places compact citations after the table', () => {
    const structured = [
      { document_id: 'doc-1', document_name: 'Lecture.pdf', chunk_id: 'c1', page_start: 20, page_end: 20 },
      { document_id: 'doc-2', document_name: 'Lecture.pdf', chunk_id: 'c2', page_start: 21, page_end: 21 },
    ];
    const html = renderMarkdown(
      '| Topic | Sources |\n|---|---|\n| Attention | [Source 1, Source 2] |',
      structured
    );
    assert.ok(!html.includes('[Source 1, Source 2]'));
    assert.ok(html.indexOf('</table>') < html.indexOf('data-testid="citation-badge"'));
    assert.ok(html.includes('p.20'));
    assert.ok(html.includes('p.21'));
  });
});

describe('4. Citation Deduplication & Density Reduction', () => {
  it('deduplicates structured citations by document_id + page_start + page_end', () => {
    const duplicateList = [
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c1', page_start: 3, page_end: 3 },
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c2', page_start: 3, page_end: 3 },
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c3', page_start: 3, page_end: 3 },
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c4', page_start: 4, page_end: 4 },
      { document_id: 'doc-2', document_name: 'Notes.pdf', chunk_id: 'c5', page_start: 3, page_end: 3 },
    ];

    const unique = deduplicateCitations(duplicateList);
    assert.equal(unique.length, 3);
    assert.deepEqual(
      unique.map(c => getCitationKey(c)),
      ['doc-1:3:3', 'doc-1:4:4', 'doc-2:3:3']
    );
  });

  it('reduces citation density: shows citation at end of paragraph and reuses one citation for repeated claims', () => {
    const md =
      'Sentence A [Doc: Paper.pdf, p. 2]. Sentence B [Doc: Paper.pdf, p. 2]. Sentence C [Doc: Paper.pdf, p. 2].';
    const html = renderMarkdown(md);
    // Intermediate sentence citations stripped
    assert.ok(html.includes('Sentence A.'));
    assert.ok(html.includes('Sentence B.'));
    assert.ok(html.includes('Sentence C.'));
    // Exactly ONE compact citation rendered at end
    const badgeMatches = html.match(/data-testid="citation-badge"/g) || [];
    assert.equal(badgeMatches.length, 1);
    assert.ok(html.includes('📄 p.2'));
  });

  it('filters out citations already rendered inline from bottom evidence list', () => {
    const md = 'Transformer uses attention [Doc: Paper.pdf, p. 2].';
    const structured = [
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c1', page_start: 2, page_end: 2 },
      { document_id: 'doc-1', document_name: 'Paper.pdf', chunk_id: 'c2', page_start: 5, page_end: 5 },
    ];

    const { inlineCitationKeys } = preprocessMarkdownWithCitations(md, structured);
    const deduplicated = deduplicateCitations(structured);
    const bottomList = deduplicated.filter(c => !inlineCitationKeys.has(getCitationKey(c)));

    assert.equal(bottomList.length, 1);
    assert.equal(bottomList[0].page_start, 5);
  });
});

describe('5. Streaming Assistant Messages', () => {
  it('progressively renders streaming tokens without buffering the full response', () => {
    const streamTokens = [
      '### Kiến trúc ',
      'Transformer\n\n',
      'Mô hình bao gồm **$d_{model}$ = 512**.\n\n',
      '$$\\text{Attention}(Q, K, V) = ',
      '\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n',
      'Tham khảo: [Doc: Transformer.pdf, p. 9]',
    ];

    let accumulated = '';
    const intermediateRenders = [];

    for (const token of streamTokens) {
      accumulated += token;
      // Render at each incremental step
      const html = renderMarkdown(accumulated);
      intermediateRenders.push(html);
      assert.ok(typeof html === 'string');
      assert.ok(html.length > 0);
    }

    // First chunk renders heading
    assert.ok(intermediateRenders[0].includes('Kiến trúc'));
    // Third chunk renders inline math
    assert.ok(intermediateRenders[2].includes('katex'));
    // Fifth chunk renders display math
    assert.ok(intermediateRenders[4].includes('katex'));
    assert.ok(intermediateRenders[4].includes('Attention'));
    // Final chunk renders compact citation marker
    assert.ok(intermediateRenders[5].includes('📄 p.9'));
  });

  it('safely handles open syntax tags during active token streaming', () => {
    const openTags = [
      '**unclosed bold',
      '*unclosed italic',
      '# unclosed heading',
      '```python\ndef incomplete(',
      '$$\\frac{1}{2',
      '$d_{mod',
      '[Doc: Trans',
    ];

    for (const partial of openTags) {
      assert.doesNotThrow(() => {
        const html = renderMarkdown(partial);
        assert.ok(typeof html === 'string');
      }, `Failed on partial markdown: ${partial}`);
    }
  });
});

describe('6. Chat selection and Documents Evidence UI', () => {
  it('does not disable text selection on the workspace containing chat messages', () => {
    const splitView = readFileSync(new URL('../src/components/WorkspaceSplitView.tsx', import.meta.url), 'utf8');
    assert.match(splitView, /overflow-hidden relative bg-white/);
    assert.doesNotMatch(splitView, /overflow-hidden relative select-none bg-white/);
  });

  it('normalizes no-comma citation formats and renders evidence metadata in a focusable tooltip', () => {
    const html = renderMarkdown('Evidence [doc: lecture_notes.pdf p.9].');
    const buttonHtml = html.match(/<button[^>]*>([\s\S]*?)<\/button>/)?.[1] || '';
    assert.ok(!buttonHtml.includes('lecture_notes.pdf'));
    assert.ok(html.includes('lecture_notes.pdf'));
    assert.ok(html.includes('p.9'));
    assert.equal(formatCitationPages(9, 10), 'pp.9–10');

    const chatArea = readFileSync(new URL('../src/components/ChatArea.tsx', import.meta.url), 'utf8');
    assert.match(chatArea, /formatCitationPages\(c\.page_start, c\.page_end\)/);
    assert.match(chatArea, /role="tooltip"/);
    assert.match(chatArea, /group-hover:flex group-focus-within:flex/);
    assert.match(chatArea, /onClick=\{\(\) => handleOpenCitation\(c\)\}/);
  });
});
