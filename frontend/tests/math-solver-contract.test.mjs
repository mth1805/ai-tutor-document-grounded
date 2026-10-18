import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

function loadApi() {
  const source = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
  } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(() => ({ API_BASE_URL: 'http://localhost:8000' }), module, module.exports);
  return module.exports.apiClient;
}

test('streaming client carries optional solver metadata with existing tokens and citations', async t => {
  const solver = { used: true, type: 'math', operation: 'solve_equation', verified: true };
  const citation = { document_id: 'd', document_name: 'Exercises.pdf', chunk_id: 'c', page_start: 1, page_end: 1 };
  const done = { message_id: 'm', content: 'Verified result: x ∈ {2, 3}', citations: [citation],
    has_sufficient_evidence: true, solver };
  const wire = 'event: token\ndata: {"token":"Factor the expression."}\n\n' +
    `event: done\ndata: ${JSON.stringify(done)}\n\n`;
  const encoded = new TextEncoder().encode(wire);
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) {
      // Split JSON and UTF-8 across transport chunks.
      for (let index = 0; index < encoded.length; index += 7) controller.enqueue(encoded.slice(index, index + 7));
      controller.close();
    },
  })));
  const tokens = [];
  let final;
  await loadApi().streamChat('conversation', { content: 'Solve exercise 7' }, 'test-token', {
    onToken: token => tokens.push(token), onDone: payload => { final = payload; },
    onError: error => assert.fail(error),
  });
  assert.deepEqual(tokens, ['Factor the expression.']);
  assert.deepEqual(final, done);
  assert.equal(Object.hasOwn(final.solver, 'result'), false);
});
