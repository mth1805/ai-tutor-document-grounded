"use client";

import React, { useState, useEffect } from "react";
import {
  FileText,
  Download,
  ExternalLink,
  X,
  FileSpreadsheet,
  FileCode,
  Image as ImageIcon,
  Loader2,
  AlertCircle,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Copy,
  Check,
  ChevronDown,
} from "lucide-react";
import { DocumentItem } from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";
import { useAuth } from "@/lib/auth-context";
import { DocumentManager } from "@/components/DocumentManager";
import { downloadOriginalDocument } from "@/lib/document-download";

interface DocumentViewerProps {
  workspaceId: string;
  document: DocumentItem | null;
  onClose: () => void;
  onSelectDocument: (doc: DocumentItem | null) => void;
}

export function DocumentViewer({
  workspaceId,
  document,
  onClose,
  onSelectDocument,
}: DocumentViewerProps) {
  const { token } = useAuth();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [textContent, setTextContent] = useState<string | null>(null);
  const [previewMime, setPreviewMime] = useState("");

  // Image zoom state
  const [imageZoom, setImageZoom] = useState<number>(100);
  // Text copy state
  const [hasCopiedText, setHasCopiedText] = useState(false);
  // Switcher dropdown state
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);

  // Helper to determine file category
  const getFileCategory = (doc: DocumentItem): "pdf" | "image" | "txt" | "word" | "other" => {
    const ext = doc.original_filename.split(".").pop()?.toLowerCase();
    if (doc.mime_type.includes("pdf") || ext === "pdf") return "pdf";
    if (
      doc.mime_type.startsWith("image/") ||
      ["png", "jpg", "jpeg", "webp"].includes(ext || "")
    ) {
      return "image";
    }
    if (doc.mime_type.includes("text") || ext === "txt") return "txt";
    if (
      doc.mime_type.includes("word") ||
      ext === "docx" ||
      ext === "doc"
    ) {
      return "word";
    }
    return "other";
  };

  // Helper: Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  // Securely load document file using authenticated backend endpoint
  useEffect(() => {
    if (!document || !token) {
      setObjectUrl(null);
      setTextContent(null);
      setPreviewMime("");
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    let localBlobUrl: string | null = null;

    const loadDocument = async () => {
      setIsLoading(true);
      setError(null);
      setImageZoom(100);
      setTextContent(null);
      setObjectUrl(null);
      setPreviewMime("");

      const category = getFileCategory(document);

      try {
        // Fetch document bytes with verified Bearer token
        const res = await fetch(
          `${API_BASE_URL}/api/v1/documents/${document.id}/${category === "word" ? "preview" : "download"}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!res.ok) {
          throw new Error(`Failed to load document (Status ${res.status})`);
        }

        const blob = await res.blob();
        if (!isMounted) return;

        localBlobUrl = URL.createObjectURL(blob);
        setObjectUrl(localBlobUrl);
        setPreviewMime(blob.type);

        if (category === "txt" || blob.type.startsWith("text/plain")) {
          const text = await blob.text();
          if (isMounted) setTextContent(text);
        }
      } catch (err: unknown) {
        if (!isMounted) return;
        const msg =
          err instanceof Error ? err.message : "Failed to load document preview";
        setError(msg);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    loadDocument();

    return () => {
      isMounted = false;
      if (localBlobUrl) {
        URL.revokeObjectURL(localBlobUrl);
      }
    };
  }, [document, token]);

  const handleCopyText = () => {
    if (!textContent) return;
    navigator.clipboard.writeText(textContent);
    setHasCopiedText(true);
    setTimeout(() => setHasCopiedText(false), 2000);
  };

  const handleDownload = async () => {
    if (!document || !token) return;
    try {
      await downloadOriginalDocument(document.id, document.original_filename, token);
    } catch {
      setError("Unable to download the original document. Please try again.");
    }
  };

  // If no document is selected, render the document manager / upload hub
  if (!document) {
    return (
      <div className="flex flex-col h-full bg-white dark:bg-slate-950 p-4 overflow-hidden transition-colors">
        <div className="mb-3 pb-3 border-b border-slate-200 dark:border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
            <FileText className="w-4 h-4 text-brand-600 dark:text-brand-400" />
            <span>Workspace Documents</span>
          </div>
          <span className="text-[11px] text-slate-500">
            Select a document to preview
          </span>
        </div>
        <div className="flex-1 overflow-y-auto">
          <DocumentManager
            workspaceId={workspaceId}
            onSelectDocument={onSelectDocument}
          />
        </div>
      </div>
    );
  }

  const category = getFileCategory(document);

  return (
    <div className="flex flex-col h-full bg-slate-50 dark:bg-slate-950 overflow-hidden relative transition-colors">
      {/* 1. Document Viewer Top Bar */}
      <div className="h-11 border-b border-slate-200 dark:border-slate-800/80 bg-white/90 dark:bg-slate-900/60 px-3 flex items-center justify-between shrink-0 text-xs">
        {/* Left: Document Info & Category Pill */}
        <div className="flex items-center space-x-2 truncate min-w-0 mr-2">
          <div className="p-1 rounded bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 shrink-0">
            {category === "pdf" && <FileText className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400" />}
            {category === "word" && (
              <FileSpreadsheet className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400" />
            )}
            {category === "image" && (
              <ImageIcon className="w-3.5 h-3.5 text-purple-500 dark:text-purple-400" />
            )}
            {category === "txt" && <FileCode className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400" />}
            {category === "other" && <FileText className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />}
          </div>

          <span
            className="font-medium text-slate-800 dark:text-slate-200 truncate"
            title={document.original_filename}
          >
            {document.original_filename}
          </span>

          <span className="text-[10px] text-slate-500 bg-slate-100 dark:bg-slate-950 px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-800 shrink-0 hidden sm:inline">
            {formatFileSize(document.file_size)}
          </span>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center space-x-1 shrink-0">
          {/* Image Zoom controls */}
          {category === "image" && !isLoading && !error && (
            <div className="flex items-center space-x-1 mr-1 border-r border-slate-200 dark:border-slate-800 pr-1">
              <button
                onClick={() => setImageZoom((z) => Math.max(25, z - 25))}
                className="p-1 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 w-9 text-center font-mono">
                {imageZoom}%
              </span>
              <button
                onClick={() => setImageZoom((z) => Math.min(300, z + 25))}
                className="p-1 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Copy Text button for TXT */}
          {category === "txt" && textContent && (
            <button
              onClick={handleCopyText}
              className="flex items-center space-x-1 px-2 py-1 rounded bg-slate-100 dark:bg-slate-900 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800 text-[11px] transition-colors"
              title="Copy readable text"
            >
              {hasCopiedText ? (
                <>
                  <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-emerald-600 dark:text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}

          {/* Download button */}
          <button
            onClick={handleDownload}
            className="p-1.5 text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-300 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Download original file"
          >
            <Download className="w-3.5 h-3.5" />
          </button>

          {/* Open in new window if objectUrl exists */}
          {objectUrl && (
            <button
              onClick={() => window.open(objectUrl, "_blank")}
              className="p-1.5 text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-300 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Open preview in new tab"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          )}

          {/* Close preview button */}
          <button
            onClick={onClose}
            className="p-1.5 text-slate-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Close preview and return to document list"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 2. Document Preview Canvas */}
      <div className="flex-1 overflow-hidden relative flex flex-col items-center justify-center p-2 sm:p-3">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center text-slate-500 dark:text-slate-400 text-xs space-y-3">
            <Loader2 className="w-6 h-6 animate-spin text-brand-600 dark:text-brand-400" />
            <span>Loading private document preview...</span>
          </div>
        ) : error ? (
          <div className="max-w-md p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 text-center space-y-3">
            <AlertCircle className="w-8 h-8 text-rose-500 dark:text-rose-400 mx-auto" />
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-rose-700 dark:text-rose-200">
                Failed to load document
              </h4>
              <p className="text-[11px] text-rose-600 dark:text-rose-300">{error}</p>
            </div>
            <button
              onClick={handleDownload}
              className="px-3 py-1.5 rounded-lg bg-rose-100 dark:bg-rose-900/60 hover:bg-rose-200 dark:hover:bg-rose-800 text-rose-700 dark:text-rose-200 text-xs font-medium border border-rose-300 dark:border-rose-700/60 transition-colors"
            >
              Try Direct Download
            </button>
          </div>
        ) : (category === "pdf" || previewMime === "application/pdf") && objectUrl ? (
          /* PDF Preview: Embedded Sandbox Viewer (Preserving native document legibility) */
          <div className="w-full h-full rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-inner">
            <iframe
              src={objectUrl}
              className="w-full h-full border-0 rounded-lg"
              title={document.original_filename}
            />
          </div>
        ) : category === "image" && objectUrl ? (
          /* Image Preview: Scalable Canvas (Preserving native image rendering) */
          <div className="w-full h-full overflow-auto flex items-center justify-center bg-slate-100/80 dark:bg-slate-950/80 rounded-lg border border-slate-200 dark:border-slate-800/80 p-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={objectUrl}
              alt={document.original_filename}
              style={{ width: `${imageZoom}%`, maxWidth: "none" }}
              className="rounded-lg shadow-md dark:shadow-xl object-contain transition-all duration-150"
            />
          </div>
        ) : textContent !== null ? (
          /* Text Preview: Scrollable Code/Document Reader */
          <div className="w-full h-full overflow-y-auto bg-white dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800/80 p-4 shadow-inner font-mono text-xs text-slate-800 dark:text-slate-200 select-text leading-relaxed">
            {category === "word" && (
              <p className="mb-3 font-sans text-slate-500 dark:text-slate-400">
                Readable text preview. Equations are shown inline; download the original for its page layout.
              </p>
            )}
            <pre className="whitespace-pre-wrap break-words">{textContent}</pre>
          </div>
        ) : category === "word" ? (
          /* DOC/DOCX Safe Fallback */
          <div className="max-w-md w-full p-6 rounded-2xl bg-white dark:bg-gradient-to-b dark:from-slate-900 dark:to-slate-950 border border-slate-200 dark:border-slate-800 shadow-md dark:shadow-xl text-center space-y-4 my-auto">
            <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/30 text-blue-600 dark:text-blue-400 flex items-center justify-center mx-auto shadow-lg shadow-blue-500/10">
              <FileSpreadsheet className="w-7 h-7" />
            </div>

            <div className="space-y-1.5">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                {document.original_filename}
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Microsoft Word Document • {formatFileSize(document.file_size)}
              </p>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed text-left space-y-1">
              <div className="font-semibold text-slate-800 dark:text-slate-300">Preview unavailable</div>
              <p>
                Download the original document to view its content.
              </p>
            </div>

            <button
              onClick={handleDownload}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium text-xs shadow-lg shadow-blue-600/20 transition-all active:scale-98"
            >
              <Download className="w-4 h-4" />
              <span>Download Original Document</span>
            </button>
          </div>
        ) : (
          /* Generic File Fallback */
          <div className="max-w-md w-full p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center space-y-4 my-auto shadow-md">
            <FileText className="w-12 h-12 text-slate-400 dark:text-slate-500 mx-auto" />
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                {document.original_filename}
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {formatFileSize(document.file_size)} • {document.mime_type}
              </p>
            </div>
            <button
              onClick={handleDownload}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-medium shadow-md shadow-brand-500/20 transition-all"
            >
              <Download className="w-4 h-4" />
              <span>Download File</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
