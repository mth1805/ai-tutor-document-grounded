import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DocumentListRequestTracker, isDocumentProcessing, shouldApplyDocumentListResponse } from '../src/lib/document-status.ts';

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

test('stale document polling responses cannot overwrite newer responses', () => {
  assert.equal(shouldApplyDocumentListResponse(2, 3), false);
  assert.equal(shouldApplyDocumentListResponse(3, 3), true);
  assert.equal(shouldApplyDocumentListResponse(4, 3), false);
});

test('a mutation invalidates in-flight polling responses', () => {
  const requests = new DocumentListRequestTracker();
  const oldRequest = requests.begin();
  requests.invalidate();
  assert.equal(requests.isCurrent(oldRequest), false);
  const refresh = requests.begin();
  assert.equal(requests.isCurrent(oldRequest), false);
  assert.equal(requests.isCurrent(refresh), true);
  const newerRefresh = requests.begin();
  assert.equal(requests.isCurrent(refresh), false);
  assert.equal(requests.isCurrent(newerRefresh), true);
});
