import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import React from 'react';
import ReactDOMServer from 'react-dom/server';
import ts from 'typescript';
import { friendlyAuthError, SIGNUP_CONFIRMATION_MESSAGE } from '../src/lib/auth-messages.ts';

const require = createRequire(import.meta.url);
function load(relative, mocks) {
  const source = readFileSync(new URL(relative, import.meta.url), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React,
    target: ts.ScriptTarget.ES2020, esModuleInterop: true,
  } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => mocks[name] ?? require(name), module, module.exports);
  return module.exports;
}
function hooks() {
  const state = [];
  let index = 0;
  return {
    react: { ...React, useState: value => {
      const key = index++;
      if (!(key in state)) state[key] = value;
      return [state[key], value => { state[key] = typeof value === 'function' ? value(state[key]) : value; }];
    }, useEffect: () => {} },
    render: component => { index = 0; return component(); },
    state,
  };
}
function find(element, predicate) {
  if (!React.isValidElement(element)) return null;
  if (predicate(element)) return element;
  for (const child of React.Children.toArray(element.props.children)) {
    const match = find(child, predicate);
    if (match) return match;
  }
  return null;
}

test('friendly errors cover unconfirmed email and existing credential failures', () => {
  const expected = 'Please confirm your email before signing in. Check your inbox for the confirmation link.';
  assert.equal(friendlyAuthError({ message: 'Email not confirmed' }), expected);
  assert.equal(friendlyAuthError({ message: 'provider details', code: 'email_not_confirmed' }), expected);
  assert.match(friendlyAuthError({ message: 'Invalid login credentials' }), /email or password is incorrect/);
  assert.doesNotMatch(friendlyAuthError({ message: 'private provider details' }), /private provider/);
});

test('signup page renders confirmation message after submitting and does not redirect', async () => {
  const harness = hooks();
  const calls = [];
  const Page = load('../src/app/(auth)/signup/page.tsx', {
    react: harness.react,
    'next/link': ({ href, children, ...props }) => React.createElement('a', { href, ...props }, children),
    '@/lib/auth-context': { useAuth: () => ({ isLoading: false, signUp: async (...args) => { calls.push(args); return {}; } }) },
    '@/lib/auth-messages': { SIGNUP_CONFIRMATION_MESSAGE },
  }).default;
  let tree = harness.render(Page);
  const email = find(tree, node => node.type === 'input' && node.props.type === 'email');
  email.props.onChange({ target: { value: 'student@example.com' } });
  const passwordInputs = [];
  const collect = node => {
    if (!React.isValidElement(node)) return;
    if (node.type === 'input' && node.props.type === 'password') passwordInputs.push(node);
    React.Children.forEach(node.props.children, collect);
  };
  collect(tree);
  for (const input of passwordInputs) input.props.onChange({ target: { value: 'password123' } });
  tree = harness.render(Page);
  await find(tree, node => node.type === 'form').props.onSubmit({ preventDefault() {} });
  tree = harness.render(Page);
  const html = ReactDOMServer.renderToStaticMarkup(tree);
  assert.ok(html.includes(SIGNUP_CONFIRMATION_MESSAGE));
  assert.ok(find(tree, node => node.props.role === 'status'));
  assert.deepEqual(calls, [['student@example.com', 'password123']]);
});

test('Supabase auth keeps no-session signup logged out and maps unconfirmed sign-in', async () => {
  const harness = hooks();
  let loginError = { message: 'Email not confirmed', code: 'email_not_confirmed' };
  const auth = {
    signUp: async () => ({ data: { user: { id: 'new-account' }, session: null }, error: null }),
    signInWithPassword: async () => ({ error: loginError }),
  };
  const { AuthProvider } = load('../src/lib/auth-context.tsx', {
    react: harness.react,
    '@/lib/supabase/client': { isSupabaseConfigured: () => true, createClient: () => ({ auth }) },
    '@/lib/auth-messages': { friendlyAuthError },
  });
  let tree = harness.render(() => AuthProvider({ children: null }));
  assert.deepEqual(await tree.props.value.signUp('student@example.com', 'password123'), {});
  tree = harness.render(() => AuthProvider({ children: null }));
  assert.equal(tree.props.value.user, null);
  assert.equal(tree.props.value.session, null);
  assert.equal(tree.props.value.token, null);
  assert.equal(tree.props.value.isLoading, false);
  assert.match((await tree.props.value.signIn('student@example.com', 'password123')).error, /confirm your email/);
  loginError = null;
  assert.deepEqual(await tree.props.value.signIn('student@example.com', 'password123'), {});
});

test('Unicode download fetches the authenticated original, never the converted preview', async () => {
  const names = ['BÀI 5.pdf', 'ĐỀ CƯƠNG.docx', 'Học máy nâng cao.pdf', 'Tài liệu NLP tiếng Việt.txt', 'Chương 1 – Tổng quan.pdf'];
  const calls = [];
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  globalThis.window = { document: {
    createElement: () => ({ click() { calls.push(this.download); }, remove() {} }),
    body: { appendChild() {} },
  } };
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'https://api.test/api/v1/documents/document-id/download');
    assert.equal(options.headers.Authorization, 'Bearer test-token');
    return new Response('original document');
  };
  try {
    const { downloadOriginalDocument } = load('../src/lib/document-download.ts', { './config': { API_BASE_URL: 'https://api.test' } });
    for (const name of names) await downloadOriginalDocument('document-id', name, 'test-token');
    assert.deepEqual(calls, names);
  } finally {
    globalThis.window = originalWindow;
    globalThis.fetch = originalFetch;
  }
});

test('product JSX contains no visible phase labels', () => {
  const files = [];
  function walk(url) {
    for (const entry of readdirSync(url, { withFileTypes: true })) {
      const child = new URL(entry.name + (entry.isDirectory() ? '/' : ''), url);
      if (entry.isDirectory()) walk(child);
      else if (entry.name.endsWith('.tsx')) files.push(child);
    }
  }
  walk(new URL('../src/', import.meta.url));
  for (const url of files) {
    const source = ts.createSourceFile(url.pathname, readFileSync(url, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    function visit(node) {
      if (ts.isJsxText(node) || ts.isStringLiteral(node)) assert.doesNotMatch(node.text, /\bphase\s*\d+\b/i, url.pathname);
      ts.forEachChild(node, visit);
    }
    visit(source);
  }
});

test('viewer renders Word PDF/text responses and preserves PDF/image loading routes', async () => {
  const originalFetch = globalThis.fetch;
  try {
    for (const [filename, originalMime, previewMime, body, tag] of [
      ['ĐỀ CƯƠNG.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/pdf', '%PDF-1.5 preview', 'iframe'],
      ['ĐỀ CƯƠNG.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'Tài liệu tiếng Việt', 'pre'],
      ['BÀI 5.pdf', 'application/pdf', 'application/pdf', '%PDF-1.5 original', 'iframe'],
      ['image.png', 'image/png', 'image/png', 'image bytes', 'img'],
    ]) {
      const harness = hooks();
      const effects = [];
      harness.react.useEffect = effect => { effects.push(effect); };
      globalThis.fetch = async (url, options) => {
        assert.equal(url, `https://api.test/api/v1/documents/document-id/${filename.endsWith('.docx') ? 'preview' : 'download'}`);
        assert.equal(options.headers.Authorization, 'Bearer test-token');
        return new Response(body, { headers: { 'Content-Type': previewMime } });
      };
      const { DocumentViewer } = load('../src/components/DocumentViewer.tsx', {
        react: harness.react,
        '@/lib/config': { API_BASE_URL: 'https://api.test' },
        '@/lib/auth-context': { useAuth: () => ({ token: 'test-token' }) },
        '@/components/DocumentManager': { DocumentManager: () => null },
        '@/lib/document-download': { downloadOriginalDocument: async () => {} },
      });
      const props = { workspaceId: 'workspace-id', document: {
        id: 'document-id', original_filename: filename, mime_type: originalMime, file_size: 100,
      }, onClose() {}, onSelectDocument() {} };
      harness.render(() => DocumentViewer(props));
      const cleanup = effects[0]();
      await new Promise(resolve => setImmediate(resolve));
      const tree = harness.render(() => DocumentViewer(props));
      const preview = find(tree, node => node.type === tag);
      assert.ok(preview, `${filename} should render ${previewMime} as ${tag}`);
      if (tag === 'pre') assert.equal(preview.props.children, body);
      else assert.match(preview.props.src, /^blob:/);
      cleanup();
    }
  } finally { globalThis.fetch = originalFetch; }
});
