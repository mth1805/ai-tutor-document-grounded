"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  FileText,
  UploadCloud,
  Trash2,
  Download,
  AlertCircle,
  Loader2,
  CheckCircle2,
  FileCode,
  FileSpreadsheet,
  Image as ImageIcon,
  FileCheck,
  RefreshCw,
  Eye,
  RotateCw,
  Cpu,
  Search,
} from "lucide-react";
import { apiClient, DocumentItem } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { RetrievalInspectorModal } from "@/components/RetrievalInspectorModal";

interface DocumentManagerProps {
  workspaceId: string;
  onDocumentCountChange?: (count: number) => void;
  compact?: boolean;
  onSelectDocument?: (doc: DocumentItem) => void;
  selectedDocumentId?: string | null;
}

export function DocumentManager({
  workspaceId,
  onDocumentCountChange,
  compact = false,
  onSelectDocument,
  selectedDocumentId,
}: DocumentManagerProps) {
  const { token } = useAuth();

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressText, setUploadProgressText] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [reprocessingId, setReprocessingId] = useState<string | null>(null);
  const [embeddingDocId, setEmbeddingDocId] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper: Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  // Helper: Pick appropriate icon for file
  const getFileIcon = (mimeType: string, filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase();
    if (mimeType.includes("pdf") || ext === "pdf") {
      return <FileText className="w-4 h-4 text-rose-400" />;
    }
    if (
      mimeType.includes("word") ||
      ext === "docx" ||
      ext === "doc"
    ) {
      return <FileSpreadsheet className="w-4 h-4 text-blue-400" />;
    }
    if (mimeType.startsWith("image/") || ["png", "jpg", "jpeg", "webp"].includes(ext || "")) {
      return <ImageIcon className="w-4 h-4 text-purple-400" />;
    }
    if (mimeType.includes("text") || ext === "txt") {
      return <FileCode className="w-4 h-4 text-emerald-400" />;
    }
    return <FileText className="w-4 h-4 text-slate-400" />;
  };

  // Fetch documents for the workspace
  const fetchDocuments = useCallback(async () => {
    if (!token || !workspaceId) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await apiClient.listDocuments(workspaceId, token);
      setDocuments(data);
      if (onDocumentCountChange) {
        onDocumentCountChange(data.length);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load documents";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [token, workspaceId, onDocumentCountChange]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Auto-poll document list if any document is currently processing, queued, or embedding
  useEffect(() => {
    const hasActiveProcessing = documents.some(
      (d) =>
        d.status === "processing" ||
        d.status === "uploaded" ||
        d.embedding_status === "processing"
    );
    if (!hasActiveProcessing || !token || !workspaceId) return;

    const interval = setInterval(() => {
      fetchDocuments();
    }, 3000);

    return () => clearInterval(interval);
  }, [documents, token, workspaceId, fetchDocuments]);

  // Handle file selection and upload
  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0 || !token || isUploading) return;
    const file = files[0];

    // Client-side validation: Max 25MB
    const maxBytes = 25 * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`File "${file.name}" exceeds the maximum 25 MB limit.`);
      return;
    }

    // Client-side extension validation
    const allowed = [
      ".pdf",
      ".doc",
      ".docx",
      ".txt",
      ".png",
      ".jpg",
      ".jpeg",
      ".webp",
    ];
    const ext = "." + (file.name.split(".").pop()?.toLowerCase() || "");
    if (!allowed.includes(ext)) {
      setError(
        `Unsupported file type "${ext}". Supported formats: PDF, DOC, DOCX, TXT, PNG, JPG, WEBP`
      );
      return;
    }

    setIsUploading(true);
    setUploadProgressText(`Uploading ${file.name}...`);
    setError(null);
    setSuccessMessage(null);

    try {
      const uploaded = await apiClient.uploadDocument(workspaceId, file, token);
      setDocuments((prev) => [uploaded, ...prev]);
      if (onDocumentCountChange) {
        onDocumentCountChange(documents.length + 1);
      }
      setSuccessMessage(`"${uploaded.original_filename}" uploaded successfully!`);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to upload document";
      setError(msg);
    } finally {
      setIsUploading(false);
      setUploadProgressText("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // Handle document deletion
  const handleDeleteDocument = async (docId: string, docName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;
    if (!confirm(`Are you sure you want to delete "${docName}"?`)) return;

    setDeletingId(docId);
    setError(null);
    try {
      await apiClient.deleteDocument(docId, token);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      if (onDocumentCountChange) {
        onDocumentCountChange(Math.max(0, documents.length - 1));
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete document";
      setError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  // Handle document download / view
  const handleDownloadDocument = async (docId: string, docName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;

    try {
      const res = await apiClient.getDocumentDownloadUrl(docId, token);
      if (res.download_url) {
        window.open(res.download_url, "_blank", "noopener,noreferrer");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to get download URL";
      setError(msg);
    }
  };

  // Handle re-processing document
  const handleReprocessDocument = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;

    setReprocessingId(docId);
    setError(null);
    try {
      await apiClient.reprocessDocument(docId, token);
      setSuccessMessage("Document ingestion started.");
      setTimeout(() => setSuccessMessage(null), 3000);
      await fetchDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to start document ingestion";
      setError(msg);
    } finally {
      setReprocessingId(null);
    }
  };

  // Handle embedding generation
  const handleEmbedDocument = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;

    setEmbeddingDocId(docId);
    setError(null);
    try {
      await apiClient.embedDocument(docId, token);
      setSuccessMessage("Embedding generation scheduled.");
      setTimeout(() => setSuccessMessage(null), 3000);
      await fetchDocuments();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to start document embedding";
      setError(msg);
    } finally {
      setEmbeddingDocId(null);
    }
  };

  // Render status badge
  const renderStatusBadge = (doc: DocumentItem) => {
    if (doc.status === "processing") {
      return (
        <span
          className="inline-flex items-center text-[10px] font-medium text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/40 px-1.5 py-0.5 rounded"
          title="Parsing, OCR, and chunking in progress..."
        >
          <Loader2 className="w-2.5 h-2.5 mr-1 animate-spin text-amber-600 dark:text-amber-400" />
          Processing
        </span>
      );
    }

    if (doc.status === "failed") {
      return (
        <span
          className="inline-flex items-center text-[10px] font-medium text-rose-700 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/40 px-1.5 py-0.5 rounded cursor-help"
          title={doc.processing_error || "Processing failed"}
        >
          <AlertCircle className="w-2.5 h-2.5 mr-1 text-rose-500 dark:text-rose-400 shrink-0" />
          Failed
        </span>
      );
    }

    if (doc.status === "processed") {
      if (doc.embedding_status === "processing") {
        return (
          <span
            className="inline-flex items-center text-[10px] font-medium text-violet-700 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800/40 px-1.5 py-0.5 rounded"
            title="Generating BGE-M3 vector embeddings..."
          >
            <Loader2 className="w-2.5 h-2.5 mr-1 animate-spin text-violet-600 dark:text-violet-400" />
            Embedding
          </span>
        );
      }
      if (doc.embedding_status === "completed") {
        return (
          <span
            className="inline-flex items-center text-[10px] font-medium text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/40 px-1.5 py-0.5 rounded"
            title="Indexed with BGE-M3 embeddings in pgvector"
          >
            <CheckCircle2 className="w-2.5 h-2.5 mr-1 text-emerald-600 dark:text-emerald-400" />
            Embedded
          </span>
        );
      }
      if (doc.embedding_status === "failed") {
        return (
          <span
            className="inline-flex items-center text-[10px] font-medium text-rose-700 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/40 px-1.5 py-0.5 rounded cursor-help"
            title={doc.embedding_error || "Embedding failed"}
          >
            <AlertCircle className="w-2.5 h-2.5 mr-1 text-rose-500 dark:text-rose-400 shrink-0" />
            Embed Failed
          </span>
        );
      }
      return (
        <span
          className="inline-flex items-center text-[10px] font-medium text-indigo-700 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800/40 px-1.5 py-0.5 rounded"
          title="Document parsed and chunked; awaiting embedding"
        >
          <CheckCircle2 className="w-2.5 h-2.5 mr-1 text-indigo-600 dark:text-indigo-400" />
          Processed
        </span>
      );
    }

    // Default: uploaded
    return (
      <span
        className="inline-flex items-center text-[10px] font-medium text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 px-1.5 py-0.5 rounded"
        title="File stored; awaiting background ingestion"
      >
        <FileText className="w-2.5 h-2.5 mr-1 text-slate-500 dark:text-slate-400" />
        Uploaded
      </span>
    );
  };

  // Drag-and-drop handlers
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = () => {
    setIsDragOver(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-3">
      {/* Hidden File Picker */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => handleFileUpload(e.target.files)}
        accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.webp"
        className="hidden"
      />

      {/* Upload Zone / Action Header */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`rounded-xl border transition-all ${
          isDragOver
            ? "border-brand-500 bg-brand-500/10 shadow-lg shadow-brand-500/10"
            : "border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 hover:border-slate-300 dark:hover:border-slate-700"
        } ${compact ? "p-2.5" : "p-4"}`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/20 text-brand-600 dark:text-brand-400 flex items-center justify-center shrink-0">
              <UploadCloud className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                Workspace Documents
              </h4>
              <p className="text-[10px] text-slate-500 dark:text-slate-400">
                PDF, DOCX, TXT, Images (Max 25MB)
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsInspectorOpen(true)}
              className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 font-medium text-xs border border-slate-200 dark:border-slate-800 shadow-sm transition-all active:scale-95"
              title="Inspect Hybrid Retrieval & Cross-Encoder Reranking"
            >
              <Search className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400" />
              <span>Inspect</span>
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-medium text-xs shadow-md shadow-brand-600/20 transition-all active:scale-95 disabled:opacity-50"
              title="Upload Document"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span>Upload</span>
                </>
              )}
            </button>
          </div>
        </div>

        {isUploading && (
          <div className="mt-3 flex items-center space-x-2 text-xs text-brand-700 dark:text-brand-300 bg-brand-50 dark:bg-brand-950/40 border border-brand-200 dark:border-brand-800/40 px-3 py-1.5 rounded-lg animate-pulse">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-brand-600 dark:text-brand-400" />
            <span>{uploadProgressText}</span>
          </div>
        )}
      </div>

      {/* Success Banner */}
      {successMessage && (
        <div className="p-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/50 flex items-center space-x-2 text-xs text-emerald-700 dark:text-emerald-300">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <span className="truncate">{successMessage}</span>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="p-2.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/50 flex items-start space-x-2 text-xs text-rose-700 dark:text-rose-300">
          <AlertCircle className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0 mt-0.5" />
          <span className="truncate flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-rose-500 hover:text-rose-700 dark:hover:text-rose-200"
          >
            &times;
          </button>
        </div>
      )}

      {/* Document List */}
      <div className="flex-1 overflow-y-auto space-y-1.5 min-h-0 pr-1">
        <div className="flex items-center justify-between px-1 mb-1 text-[11px] text-slate-500 dark:text-slate-400 font-semibold uppercase tracking-wider">
          <span>Files ({documents.length})</span>
          <button
            onClick={fetchDocuments}
            disabled={isLoading}
            className="text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300 p-0.5 rounded transition-colors"
            title="Refresh document list"
          >
            <RefreshCw
              className={`w-3 h-3 ${isLoading ? "animate-spin text-brand-600 dark:text-brand-400" : ""}`}
            />
          </button>
        </div>

        {isLoading && documents.length === 0 ? (
          <div className="py-8 flex flex-col items-center justify-center text-slate-500 text-xs space-y-2">
            <Loader2 className="w-4 h-4 animate-spin text-brand-600 dark:text-brand-400" />
            <span>Loading documents...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="py-8 px-3 text-center space-y-2 border border-dashed border-slate-300 dark:border-slate-800/80 rounded-xl bg-slate-50/50 dark:bg-slate-950/20">
            <FileText className="w-6 h-6 text-slate-400 dark:text-slate-600 mx-auto" />
            <p className="text-xs font-medium text-slate-700 dark:text-slate-300">
              No documents uploaded yet
            </p>
            <p className="text-[11px] text-slate-500 max-w-xs mx-auto">
              Drop PDFs, DOCX, or text files here to ground your AI Tutor in this workspace.
            </p>
          </div>
        ) : (
          documents.map((doc) => {
            const isSelected = selectedDocumentId === doc.id;
            return (
              <div
                key={doc.id}
                onClick={() => onSelectDocument && onSelectDocument(doc)}
                className={`group flex items-center justify-between p-2.5 rounded-lg border transition-all text-xs cursor-pointer ${
                  isSelected
                    ? "bg-brand-50 dark:bg-slate-800 text-brand-700 dark:text-brand-300 border-brand-200 dark:border-brand-500/50 shadow-sm ring-1 ring-brand-500/20"
                    : "bg-white dark:bg-slate-900/60 hover:bg-slate-50 dark:hover:bg-slate-900 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-800/80 hover:border-slate-300 dark:hover:border-slate-700"
                }`}
              >
                <div className="flex items-center space-x-2.5 min-w-0 flex-1 mr-2">
                  <div className="shrink-0 p-1.5 rounded-md bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    {getFileIcon(doc.mime_type, doc.original_filename)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p
                      className={`font-medium truncate transition-colors ${
                        isSelected ? "text-brand-700 dark:text-brand-300 font-semibold" : "text-slate-800 dark:text-slate-200 group-hover:text-brand-600 dark:group-hover:text-brand-300"
                      }`}
                      title={doc.original_filename}
                    >
                      {doc.original_filename}
                    </p>
                    <div className="flex items-center space-x-2 text-[10px] text-slate-500 dark:text-slate-400 mt-0.5 flex-wrap gap-y-1">
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span>•</span>
                      {renderStatusBadge(doc)}
                      <span>•</span>
                      <span>
                        {new Date(doc.created_at).toLocaleDateString([], {
                          month: "short",
                          day: "numeric",
                        })}
                      </span>
                    </div>

                    {/* Show error snippet for failed documents */}
                    {doc.status === "failed" && doc.processing_error && (
                      <div className="mt-1.5 p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/40 text-[10px] text-rose-700 dark:text-rose-300 break-words">
                        <span className="font-semibold text-rose-800 dark:text-rose-200">Processing Error: </span>
                        <span>{doc.processing_error}</span>
                      </div>
                    )}

                    {/* Show embedding error snippet if embedding failed */}
                    {doc.embedding_status === "failed" && doc.embedding_error && (
                      <div className="mt-1.5 p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/40 text-[10px] text-rose-700 dark:text-rose-300 break-words">
                        <span className="font-semibold text-rose-800 dark:text-rose-200">Embedding Error: </span>
                        <span>{doc.embedding_error}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div
                  className="flex items-center space-x-1 shrink-0 ml-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Re-embed document button (available when document is processed) */}
                  {doc.status === "processed" && (
                    <button
                      onClick={(e) => handleEmbedDocument(doc.id, e)}
                      disabled={
                        embeddingDocId === doc.id ||
                        doc.embedding_status === "processing"
                      }
                      className="p-1.5 text-slate-400 hover:text-violet-600 dark:hover:text-violet-300 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors disabled:opacity-40"
                      title={
                        doc.embedding_status === "processing"
                          ? "Embedding generation in progress..."
                          : "Generate or Re-generate BGE-M3 Embeddings"
                      }
                    >
                      {embeddingDocId === doc.id ||
                      doc.embedding_status === "processing" ? (
                        <RotateCw className="w-3.5 h-3.5 animate-spin text-violet-600 dark:text-violet-400" />
                      ) : (
                        <Cpu className="w-3.5 h-3.5" />
                      )}
                    </button>
                  )}

                  {/* Re-process document button */}
                  <button
                    onClick={(e) => handleReprocessDocument(doc.id, e)}
                    disabled={reprocessingId === doc.id || doc.status === "processing"}
                    className="p-1.5 text-slate-400 hover:text-brand-600 dark:hover:text-brand-300 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors disabled:opacity-40"
                    title={
                      doc.status === "processing"
                        ? "Document is currently being processed"
                        : "Re-process document parsing & chunking"
                    }
                  >
                    {reprocessingId === doc.id || doc.status === "processing" ? (
                      <RotateCw className="w-3.5 h-3.5 animate-spin text-brand-600 dark:text-brand-400" />
                    ) : (
                      <RotateCw className="w-3.5 h-3.5" />
                    )}
                  </button>

                  {onSelectDocument && (
                    <button
                      onClick={() => onSelectDocument(doc)}
                      className={`p-1.5 rounded transition-colors ${
                        isSelected
                          ? "text-brand-600 dark:text-brand-400 bg-brand-100/60 dark:bg-slate-900"
                          : "text-slate-400 hover:text-brand-600 dark:hover:text-brand-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                      }`}
                      title="Preview in Left Viewer"
                    >
                      <Eye className="w-3.5 h-3.5" />
                    </button>
                  )}
                  <button
                    onClick={(e) =>
                      handleDownloadDocument(doc.id, doc.original_filename, e)
                    }
                    className="p-1.5 text-slate-400 hover:text-brand-600 dark:hover:text-brand-300 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                    title="Download / View Original File"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={(e) =>
                      handleDeleteDocument(doc.id, doc.original_filename, e)
                    }
                    disabled={deletingId === doc.id}
                    className="p-1.5 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors disabled:opacity-50"
                    title="Delete Document"
                  >
                    {deletingId === doc.id ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-rose-500 dark:text-rose-400" />
                    ) : (
                      <Trash2 className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Phase 7 Retrieval Inspector Modal */}
      <RetrievalInspectorModal
        workspaceId={workspaceId}
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
      />
    </div>
  );
}
