import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import ts from 'typescript';
import * as citations from '../src/lib/citation-utils.ts';

// Exercise the production renderer, including its URL transform and badge handler.
const require = createRequire(import.meta.url);
const source = readFileSync(new URL('../src/components/MarkdownRenderer.tsx', import.meta.url), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true,
} }).outputText;
const module = { exports: {} };
const mocks = {
  'react-markdown': { __esModule: true, default: ReactMarkdown, defaultUrlTransform },
  'remark-gfm': remarkGfm, 'remark-math': remarkMath, 'rehype-katex': rehypeKatex,
  '@/lib/citation-utils': citations,
};
new Function('require', 'module', 'exports', code)(name => mocks[name] ?? require(name), module, module.exports);
const { MarkdownRenderer } = module.exports;
const filename = 'Bài 5. LSTM, GRU, Encoder decoder.pdf';
const evidence = [
  { document_id: 'doc-lstm', document_name: filename, chunk_id: 'c20', page_start: 20, page_end: 20, snippet: 'LSTM evidence' },
  { document_id: 'doc-lstm', document_name: filename, chunk_id: 'c37', page_start: 37, page_end: 38 },
];
const marker = (page = 'p. 20', name = filename) => `[Doc: ${name}, ${page}]`;
const render = (content, structured = evidence) => renderToStaticMarkup(
  React.createElement(MarkdownRenderer, { content, citations: structured })
);
const buttons = html => [...html.matchAll(/<button\b[^>]*aria-label="([^"]+)"[^>]*>([\s\S]*?)<\/button>/g)];
const links = (text, structured = evidence) => {
  const result = citations.preprocessMarkdownWithCitations(text, structured);
  return [...result.processedText.matchAll(/\]\((citation:\/\/[^)]+)\)/g)]
    .map(m => citations.parseCitationUrl(m[1]));
};

test('production renderer keeps single pages, ranges and multiple citations in the correct GFM cells', () => {
  for (const leadingPipe of [true, false]) {
    const prefix = leadingPipe ? '| ' : '';
    const html = render(`${prefix}Claim | Evidence |\n${prefix}--- | --- |\n${prefix}LSTM | ${marker()} ${marker('pp. 37-38')} |\n${prefix}GRU | ${marker()} |`);
    const cells = [...html.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(m => m[1]);
    assert.equal(cells.length, 4);
    assert.equal(buttons(cells[1]).length, 2);
    assert.equal(buttons(cells[3]).length, 1);
    assert.equal(buttons(html.slice(html.indexOf('</table>'))).length, 0);
    assert.doesNotMatch(html, /\[Doc:/);
    assert.match(cells[1], /pp\.37–38/);
    assert.match(cells[1], /aria-label="Bài 5\. LSTM, GRU, Encoder decoder\.pdf, Page 20"/);
    assert.match(cells[1], /role="tooltip"/);
    assert.match(cells[1], /group-focus-within:flex/);
    assert.match(cells[1], /lucide-file-text/);
  }
});

test('citations stay next to claims in paragraphs, headings and both list types', () => {
  const html = render(`# Heading ${marker()}\n\n**Claim** ${marker()}.\n\n- Bullet ${marker()}\n\n1. Step ${marker('pp. 37–38')}`);
  for (const tag of ['h1', 'p', 'li']) assert.match(html, new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?<button`));
  assert.equal(buttons(html).length, 4);
  assert.match(html, /<strong[^>]*>Claim<\/strong>/);
});

test('only adjacent duplicates collapse; distinct ranges, documents, cells and claims retain badges', () => {
  const extra = { ...evidence[0], document_id: 'doc-other', document_name: 'Khác.pdf' };
  const text = `Claim ${marker()} ${marker()} [Source 1, Source 2, Source 1]. Next ${marker()} ${marker('p. 20', 'Khác.pdf')}`;
  const html = render(text, [...evidence, extra]);
  assert.equal(buttons(html).length, 4);
  assert.equal(buttons(html).filter(m => m[1].endsWith('Page 20')).length, 3);
});

test('metadata resolves exact Unicode filenames and pages without silently substituting different evidence', () => {
  assert.deepEqual(links(marker()), [evidence[0]]);
  assert.equal(links(marker('pp. 37-38'))[0].chunk_id, 'c37');
  assert.equal(links(marker('p. 37'))[0].page_end, 37);
  for (const name of [filename.normalize('NFD'), filename.toUpperCase()]) {
    assert.equal(links(marker('p. 20', name))[0].document_id, 'doc-lstm');
  }
  for (const text of [marker('p. 99'), marker('pp. 20-38'), marker('p. 20', 'LSTM'), marker('p. 20', 'Unknown.pdf')]) {
    assert.equal(buttons(render(text)).length, 0);
    assert.equal(citations.preprocessMarkdownWithCitations(text, evidence).processedText, text);
  }
  assert.equal(buttons(render(marker(), [...evidence, { ...evidence[0], document_id: 'ambiguous' }])).length, 0);
});

test('filenames preserve commas, Unicode and Markdown punctuation in tooltips and protocol metadata', () => {
  for (const name of [filename, 'Bài_5 (LSTM), GRU & decoder!.pdf', 'Bài [5], bản mới.pdf']) {
    const structured = [{ ...evidence[0], document_name: name }];
    assert.equal(links(marker('p. 20', name), structured)[0].document_name, name);
    assert.equal(buttons(render(marker('p. 20', name), structured)).length, 1);
  }
});

test('invalid pages and malformed citation URLs never fabricate a page-one badge', () => {
  for (const page of ['p. 0', 'p. -1', 'p. abc', 'pp. 38-37', 'p. 9007199254740992']) {
    assert.equal(buttons(render(marker(page))).length, 0);
  }
  for (const url of ['citation://wrong?name=A&pageStart=1', 'citation://open?name=A',
    'citation://open?name=A&pageStart=1x', 'citation://open?name=A&pageStart=0',
    'citation://open?name=A&pageStart=2&pageEnd=1', 'citation://open?pageStart=1']) {
    assert.equal(citations.parseCitationUrl(url), null);
    assert.equal(buttons(render(`[invalid](${url})`)).length, 0);
  }
  assert.equal(buttons(render('[Source 99]')).length, 0);
});

test('streamed table citations complete in place and gain verified metadata on the final payload', () => {
  const text = `| Claim | Evidence |\n|---|---|\n| LSTM | ${marker()} ${marker('pp. 37-38')} |`;
  const firstEnd = text.indexOf(']') + 1;
  for (let i = 1; i <= text.length; i++) {
    const html = render(text.slice(0, i), []);
    assert.equal(buttons(html).length, i < firstEnd ? 0 : i < text.lastIndexOf(']') + 1 ? 1 : 2);
  }
  assert.deepEqual(links(text).map(c => c.chunk_id), ['c20', 'c37']);
  assert.equal(buttons(render(text)).length, 2);
});

test('code, math, links and escaped citation examples stay unchanged', () => {
  for (const text of [
    `\\${marker()}`,
    `\`${marker()} [Source 1]\``,
    `\`\`\`text\n${marker()}\n\n[Source 1]\n\`\`\``,
    `~~~text\n${marker()} [Source 1]\n~~~`,
    `    ${marker()} [Source 1]\n`,
    `\`\`\`\n${marker()}\n\n[Source 1]`,
    `[${marker()}](https://example.com)`,
    `[link](https://example.com "${marker()}")`,
    `$\\text{${marker()} [Source 1]}$`,
    `$$\n\\text{${marker()} [Source 1]}\n$$`,
  ]) {
    assert.equal(citations.preprocessMarkdownWithCitations(text, evidence).processedText, text);
    assert.equal(buttons(render(text)).length, 0);
  }
  const mixed = `\`${marker()}\`\n\n$E=mc^2$ [link](https://example.com) **bold** ${marker()}`;
  const html = render(mixed);
  assert.equal(buttons(html).length, 1);
  assert.match(html, /class="katex"/);
  assert.match(html, /href="https:\/\/example.com"/);
  assert.match(html, /<strong[^>]*>bold<\/strong>/);
});

test('production badge click forwards document, chunk, snippet and first page to the existing handler', () => {
  // Inspect the production anchor renderer with hooks stubbed only for useMemo.
  const testModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => name === 'react'
    ? { ...React, useMemo: fn => fn() } : mocks[name] ?? require(name), testModule, testModule.exports);
  let opened;
  const tree = testModule.exports.MarkdownRenderer({ content: marker(), citations: evidence, onOpenCitation: c => { opened = c; } });
  const markdown = tree.props.children;
  const href = citations.preprocessMarkdownWithCitations(marker(), evidence).processedText.match(/\]\((.*)\)/)[1];
  const badge = markdown.props.components.a({ href });
  const button = React.Children.toArray(badge.props.children).find(child => child.type === 'button');
  let stopped = false;
  button.props.onClick({ stopPropagation() { stopped = true; } });
  assert.equal(stopped, true);
  assert.deepEqual(opened, evidence[0]);
});
