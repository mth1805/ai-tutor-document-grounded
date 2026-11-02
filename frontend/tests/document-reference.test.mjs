import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import React from 'react';
import ts from 'typescript';
import * as citationUtils from '../src/lib/citation-utils.ts';

function compile(path, dependencies) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React,
    target: ts.ScriptTarget.ES2020, esModuleInterop: true,
  } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(dependencies, module, module.exports);
  return module.exports;
}

function find(tree, type) {
  if (!tree || typeof tree !== 'object') return;
  if (tree.type === type) return tree;
  for (const child of React.Children.toArray(tree.props?.children)) {
    const found = find(child, type);
    if (found) return found;
  }
}

test('chat submission carries only the viewer document in the current workspace', async () => {
  for (const selected of [null, { id: 'selected', workspace_id: 'workspace' },
    { id: 'stale-selection', workspace_id: 'other-workspace' }]) {
    const states = [];
    let index = 0;
    let sent;
    const react = { ...React, useState: initial => {
      const current = index++;
      if (!(current in states)) states[current] = initial;
      return [states[current], next => { states[current] = typeof next === 'function' ? next(states[current]) : next; }];
    }, useEffect() {}, useCallback: callback => callback, useRef: () => ({ current: null }) };
    const { ChatArea } = compile('../src/components/ChatArea.tsx', name => {
      if (name === 'react') return react;
      if (name === 'next/navigation') return { useRouter: () => ({}),
        useParams: () => ({ workspaceId: 'workspace', conversationId: 'conversation' }) };
      if (name === '@/lib/auth-context') return { useAuth: () => ({ user: { id: 'user' }, token: 'test-token' }) };
      if (name === '@/lib/document-context') return { useDocument: () => ({ selectedDocument: selected }) };
      if (name === '@/lib/api') return { apiClient: { streamChat: async (_cid, payload) => { sent = payload; } } };
      if (name === 'lucide-react') return new Proxy({}, { get: () => () => null });
      return {};
    });
    let tree = ChatArea();
    find(tree, 'input').props.onChange({ target: { value: 'tóm tắt file này' } });
    index = 0;
    tree = ChatArea();
    await find(tree, 'form').props.onSubmit({ preventDefault() {} });
    assert.equal(sent.content, 'tóm tắt file này');
    assert.equal(sent.workspace_id, 'workspace');
    assert.equal(sent.active_document_id, selected?.workspace_id === 'workspace' ? selected.id : undefined);
  }
});

test('streaming request serializes explicit and active document IDs with auth', async t => {
  let request;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    request = { url, ...options };
    return new Response('event: done\ndata: {"message_id":"m","content":"Which document?","citations":[]}\n\n');
  });
  const { apiClient } = compile('../src/lib/api.ts', () => ({ API_BASE_URL: 'http://localhost:8000' }));
  let done;
  await apiClient.streamChat('conversation', {
    content: 'summarize this document', workspace_id: 'workspace',
    document_id: 'attached', active_document_id: 'viewer',
  }, 'test-token', { onDone: payload => { done = payload; }, onError: error => assert.fail(error) });
  assert.equal(request.headers.Authorization, 'Bearer test-token');
  assert.equal(JSON.parse(request.body).document_id, 'attached');
  assert.equal(JSON.parse(request.body).active_document_id, 'viewer');
  assert.equal(done.content, 'Which document?');
});

test('chat citation handler opens the verified document and page, with unambiguous exact filename fallback', async () => {
  const filename = 'Bài 5. LSTM, GRU, Encoder decoder.pdf';
  const doc = { id: 'document-id', workspace_id: 'workspace', original_filename: filename };
  for (const scenario of [
    { citation: { document_id: doc.id, document_name: filename, page_start: 37, page_end: 38 }, docs: [], expected: true },
    { citation: { document_name: filename, page_start: 20, page_end: 20 }, docs: [doc], expected: true },
    { citation: { document_name: 'LSTM', page_start: 20, page_end: 20 }, docs: [doc], expected: false },
    { citation: { document_name: filename, page_start: 20, page_end: 20 }, docs: [doc, { ...doc, id: 'duplicate-name' }], expected: false },
  ]) {
    const states = [];
    let index = 0;
    let selected;
    const fetched = [];
    const MarkdownRenderer = () => null;
    const react = { ...React, useState: initial => {
      const current = index++;
      if (!(current in states)) states[current] = current === 0
        ? [{ id: 'answer', role: 'assistant', content: 'Claim', citations: [] }] : initial;
      return [states[current], next => { states[current] = next; }];
    }, useEffect() {}, useCallback: fn => fn, useRef: () => ({ current: null }) };
    const { ChatArea } = compile('../src/components/ChatArea.tsx', name => {
      if (name === 'react') return react;
      if (name === 'next/navigation') return { useRouter: () => ({}), useParams: () => ({ workspaceId: 'workspace', conversationId: 'thread' }) };
      if (name === '@/lib/auth-context') return { useAuth: () => ({ user: { id: 'user' }, token: 'test-token' }) };
      if (name === '@/lib/citation-utils') return citationUtils;
      if (name === '@/components/MarkdownRenderer') return { MarkdownRenderer };
      if (name === '@/lib/document-context') return { useDocument: () => ({ selectedDocument: null,
        setIsViewerCollapsed() {}, setSelectedDocument: (...args) => { selected = args; } }) };
      if (name === '@/lib/api') return { apiClient: {
        listDocuments: async (workspace, token) => {
          assert.equal(workspace, 'workspace'); assert.equal(token, 'test-token'); return scenario.docs;
        },
        getDocument: async (id, token) => { fetched.push(id); assert.equal(token, 'test-token'); return doc; },
      } };
      if (name === 'lucide-react') return new Proxy({}, { get: () => () => null });
      return {};
    });
    const tree = ChatArea();
    await find(tree, MarkdownRenderer).props.onOpenCitation(scenario.citation);
    assert.deepEqual(fetched, scenario.expected ? [doc.id] : []);
    assert.deepEqual(selected, scenario.expected ? [doc, scenario.citation.page_start] : undefined);
  }
});
