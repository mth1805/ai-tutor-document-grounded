import type { DocumentItem } from "./api";

export function isDocumentProcessing(document: DocumentItem): boolean {
  if (document.status === "failed") return false;
  return ["uploaded", "queued", "processing"].includes(document.status) ||
    (document.status === "processed" && ["pending", "processing"].includes(document.embedding_status || ""));
}

export class DocumentListRequestTracker {
  private latestStartedRequestId = 0;

  begin(): number {
    this.latestStartedRequestId += 1;
    return this.latestStartedRequestId;
  }

  invalidate(): void {
    this.latestStartedRequestId += 1;
  }

  isCurrent(requestId: number): boolean {
    return shouldApplyDocumentListResponse(requestId, this.latestStartedRequestId);
  }
}

export function shouldApplyDocumentListResponse(requestId: number, latestStartedRequestId: number): boolean {
  return requestId === latestStartedRequestId;
}
