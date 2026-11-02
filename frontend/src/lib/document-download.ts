import { API_BASE_URL } from "./config";

/** Fetch the private original; blob downloads preserve the Unicode display name. */
export async function downloadOriginalDocument(id: string, filename: string, token: string) {
  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${id}/download`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error("Unable to download the original document. Please try again.");
  const url = URL.createObjectURL(await response.blob());
  const link = window.document.createElement("a");
  link.href = url;
  link.download = filename;
  window.document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
