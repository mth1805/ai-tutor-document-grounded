import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { resolveApiBaseUrl, validateHostedSupabase } from '../src/lib/public-config.mjs';

test('API configuration permits localhost only locally and requires valid HTTPS in production', () => {
  // Local development defaults to http://localhost:8000
  assert.equal(resolveApiBaseUrl(), 'http://localhost:8000');
  assert.equal(resolveApiBaseUrl('http://localhost:8000/'), 'http://localhost:8000');
  assert.equal(resolveApiBaseUrl(undefined, 'http://localhost:8000/'), 'http://localhost:8000');

  // Local development browser origins
  assert.equal(resolveApiBaseUrl(undefined, undefined, { browserHostname: 'localhost' }), 'http://localhost:8000');
  assert.equal(resolveApiBaseUrl('http://localhost:8000/', undefined, { browserHostname: '127.0.0.1' }), 'http://localhost:8000');

  // Local container build override (Docker)
  assert.equal(resolveApiBaseUrl('http://localhost:8000/', undefined, { production: true, allowLocal: true }), 'http://localhost:8000');

  // Production requires valid HTTPS and uses NEXT_PUBLIC_API_BASE_URL
  assert.equal(resolveApiBaseUrl('https://test-api.modal.run///', undefined, { production: true }), 'https://test-api.modal.run');
  assert.equal(resolveApiBaseUrl('https://test-api.modal.run///', undefined, { hosted: true }), 'https://test-api.modal.run');
  assert.equal(resolveApiBaseUrl('https://test-api.modal.run///', undefined, { browserHostname: 'ai-tutor.vercel.app' }), 'https://test-api.modal.run');
  assert.equal(resolveApiBaseUrl(undefined, 'https://test-api.modal.run'), 'https://test-api.modal.run');

  // Production must not silently fall back to localhost
  assert.throws(() => resolveApiBaseUrl(undefined, undefined, { production: true }), /required for production/);
  assert.throws(() => resolveApiBaseUrl('', undefined, { production: true }), /required for production/);
  assert.throws(() => resolveApiBaseUrl(undefined, undefined, { hosted: true }), /required for production/);
  assert.throws(() => resolveApiBaseUrl(undefined, undefined, { browserHostname: 'ai-tutor.vercel.app' }), /required for production/);

  // http://localhost:8000 is forbidden in production
  assert.throws(() => resolveApiBaseUrl('http://localhost:8000/', undefined, { production: true }), /allowed only in local development/);
  assert.throws(() => resolveApiBaseUrl('http://localhost:8000/', undefined, { hosted: true }));
  assert.throws(() => resolveApiBaseUrl('http://localhost:8000/', undefined, { browserHostname: 'ai-tutor.vercel.app' }), /allowed only in local development/);

  // Rejection of invalid origins in production and hosted
  for (const value of [undefined, '', 'http://localhost:8000', 'https://127.0.0.1', '/api', 'http://test-api.modal.run', 'https://user:password@test.invalid', 'https://test.invalid/api', 'https://test.invalid?key=x']) {
    assert.throws(() => resolveApiBaseUrl(value, undefined, { hosted: true }));
    assert.throws(() => resolveApiBaseUrl(value, undefined, { production: true }));
  }

  // Conflict between primary and legacy variables
  assert.throws(() => resolveApiBaseUrl('https://a.invalid', 'https://b.invalid'), /conflict/);
});

test('hosted Supabase rejects missing, placeholder and privileged credentials', () => {
  const jwt = (role) => `header.${Buffer.from(JSON.stringify({ role })).toString('base64url')}.signature`;
  validateHostedSupabase('https://test.supabase.co', jwt('anon'));
  validateHostedSupabase('https://test.supabase.co', 'sb_publishable_test');
  for (const key of [undefined, '', 'placeholder-anon-key', 'sb_secret_test', jwt('service_role'), jwt('authenticated'), 'malformed']) {
    assert.throws(() => validateHostedSupabase('https://test.supabase.co', key));
  }
  assert.throws(() => validateHostedSupabase('http://test.supabase.co', jwt('anon')));
});

// Compile the actual API module into a network-free fetch harness.
const source = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')
  .replace('import { API_BASE_URL } from "./config";', 'const API_BASE_URL = "https://test-api.modal.run";');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { apiClient } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('REST and multipart uploads send bearer tokens to the direct backend origin', async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return new Response('{}', { headers: { 'Content-Type': 'application/json' } });
  };
  try {
    await apiClient.listWorkspaces('test-token');
    await apiClient.uploadDocument('workspace', new File(['%PDF-'], 'test.pdf'), 'test-token');
    assert.equal(calls[0].url, 'https://test-api.modal.run/api/v1/workspaces');
    assert.equal(calls[1].url, 'https://test-api.modal.run/api/v1/workspaces/workspace/documents');
    for (const call of calls) assert.equal(call.options.headers.Authorization, 'Bearer test-token');
    assert.equal(calls[1].options.headers['Content-Type'], undefined);
    assert.ok(calls[1].options.body instanceof FormData);
  } finally { globalThis.fetch = original; }
});

test('SSE delivers split UTF-8 tokens incrementally and handles disconnect and stop', async () => {
  const original = globalThis.fetch;
  let controller;
  const abort = new AbortController();
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'https://test-api.modal.run/api/v1/conversations/conversation/chat');
    assert.equal(options.headers.Authorization, 'Bearer test-token');
    assert.equal(options.signal, abort.signal);
    return new Response(new ReadableStream({ start(value) { controller = value; } }));
  };
  try {
    let resolveToken;
    const tokenReceived = new Promise((resolve) => { resolveToken = resolve; });
    const tokens = [];
    const pending = apiClient.streamChat('conversation', { content: 'hello' }, 'test-token', {
      onToken: (token) => { tokens.push(token); resolveToken(); },
    }, abort.signal);
    const bytes = new TextEncoder().encode('event: token\ndata: {"token":"Việt"}\n\n');
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    await tokenReceived;
    assert.deepEqual(tokens, ['Việt']);
    controller.enqueue(new TextEncoder().encode('event: done\ndata: {"content":"Việt"}\n\n'));
    controller.close();
    await pending;
    const disconnected = apiClient.streamChat('conversation', { content: 'hello' }, 'test-token', {}, abort.signal);
    controller.close();
    await assert.rejects(disconnected, /before completion/);
    const stopped = apiClient.streamChat('conversation', { content: 'hello' }, 'test-token', {}, abort.signal);
    abort.abort();
    controller.error(new DOMException('Aborted', 'AbortError'));
    await stopped;
  } finally { globalThis.fetch = original; }
});
