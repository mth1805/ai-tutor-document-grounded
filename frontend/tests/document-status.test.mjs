import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isDocumentProcessing } from '../src/lib/document-status.ts';

test('poll queue and parsing/model startup; stop on terminal outcomes', () => {
  for (const status of ['uploaded', 'queued', 'processing']) {
    assert.equal(isDocumentProcessing({ status, embedding_status: 'pending' }), true);
  }
  for (const embedding_status of ['pending', 'processing']) {
    assert.equal(isDocumentProcessing({ status: 'processed', embedding_status }), true);
  }
  for (const embedding_status of ['completed', 'failed']) {
    assert.equal(isDocumentProcessing({ status: 'processed', embedding_status }), false);
  }
  assert.equal(isDocumentProcessing({ status: 'failed', embedding_status: 'processing' }), false);
});
