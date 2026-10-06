import type { DocumentItem } from "./api";

export function isDocumentProcessing(document: DocumentItem): boolean {
  if (document.status === "failed") return false;
  return ["uploaded", "queued", "processing"].includes(document.status) ||
    (document.status === "processed" && ["pending", "processing"].includes(document.embedding_status || ""));
}
