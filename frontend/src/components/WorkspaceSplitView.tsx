"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import {
  FileText,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  GripVertical,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { useDocument } from "@/lib/document-context";
import { DocumentViewer } from "@/components/DocumentViewer";
import { ChatArea } from "@/components/ChatArea";

export function WorkspaceSplitView() {
  const params = useParams();
  const workspaceId = (params?.workspaceId as string) || "";

  const {
    selectedDocument,
    setSelectedDocument,
    isViewerCollapsed,
    setIsViewerCollapsed,
    splitRatio,
    setSplitRatio,
  } = useDocument();

  // Mobile responsive view mode: "viewer" | "chat"
  const [mobileTab, setMobileTab] = useState<"viewer" | "chat">("chat");

  // Dragging state for resizable divider
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // If a document is selected on mobile, auto-switch to viewer
  useEffect(() => {
    if (selectedDocument && typeof window !== "undefined" && window.innerWidth < 1024) {
      setMobileTab("viewer");
    }
  }, [selectedDocument]);

  // Handle divider drag
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const newWidthPct = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitRatio(newWidthPct);
    },
    [isDragging, setSplitRatio]
  );

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
    }
  }, [isDragging]);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    } else {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div
      ref={containerRef}
      className="flex-1 flex flex-col h-full overflow-hidden relative select-none bg-white dark:bg-slate-950 transition-colors"
    >
      {/* Mobile Screen Segmented Tab Switcher (< 1024px) */}
      <div className="lg:hidden flex items-center justify-between p-2 bg-white dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 shrink-0">
        <div className="flex bg-slate-100 dark:bg-slate-900 p-0.5 rounded-lg border border-slate-200 dark:border-slate-800 flex-1 max-w-xs">
          <button
            onClick={() => setMobileTab("viewer")}
            className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              mobileTab === "viewer"
                ? "bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-300 shadow-sm"
                : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span className="truncate">
              {selectedDocument ? selectedDocument.original_filename : "Viewer"}
            </span>
          </button>
          <button
            onClick={() => setMobileTab("chat")}
            className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              mobileTab === "chat"
                ? "bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-300 shadow-sm"
                : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Chat</span>
          </button>
        </div>
      </div>

      {/* Main Split-View Container */}
      <div className="flex-1 flex flex-row h-full overflow-hidden relative">
        {/* ======================================================== */}
        {/* LEFT PANE: Document List & Document Preview Area          */}
        {/* ======================================================== */}
        <div
          style={{
            width: isViewerCollapsed ? "0%" : `${splitRatio}%`,
            display: isViewerCollapsed ? "none" : undefined,
          }}
          className={`h-full flex-col overflow-hidden border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-950 transition-none ${
            mobileTab === "viewer" ? "flex flex-1" : "hidden lg:flex"
          }`}
        >
          <DocumentViewer
            workspaceId={workspaceId}
            document={selectedDocument}
            onClose={() => setSelectedDocument(null)}
            onSelectDocument={(doc) => setSelectedDocument(doc)}
          />
        </div>

        {/* ======================================================== */}
        {/* RESIZABLE DIVIDER (Desktop Only)                         */}
        {/* ======================================================== */}
        {!isViewerCollapsed && (
          <div
            onMouseDown={handleMouseDown}
            onDoubleClick={() => setSplitRatio(50)}
            className="hidden lg:flex w-2 bg-slate-100 hover:bg-brand-500/20 active:bg-brand-500/30 dark:bg-slate-950 dark:hover:bg-brand-500/20 dark:active:bg-brand-500/40 border-r border-slate-200 dark:border-slate-800/80 cursor-col-resize items-center justify-center relative group transition-colors select-none z-10"
            title="Drag to resize panels • Double-click to reset to 50%"
          >
            {/* Divider Grip Indicator */}
            <div className="h-8 w-1 rounded-full bg-slate-300 dark:bg-slate-700 group-hover:bg-brand-500 dark:group-hover:bg-brand-400 transition-colors" />

            {/* Quick Collapse Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsViewerCollapsed(true);
              }}
              className="absolute -left-3.5 p-1 rounded-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-white hover:bg-brand-600 dark:hover:text-white dark:hover:bg-brand-600 opacity-0 group-hover:opacity-100 transition-all shadow-md"
              title="Collapse Left Viewer"
            >
              <ChevronLeft className="w-3 h-3" />
            </button>
          </div>
        )}

        {/* Collapsed Restore Button (When Left Pane is Collapsed) */}
        {isViewerCollapsed && (
          <button
            onClick={() => setIsViewerCollapsed(false)}
            className="hidden lg:flex absolute left-2 top-3 z-20 p-2 rounded-lg bg-white/95 dark:bg-slate-900/90 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white shadow-lg transition-all items-center space-x-1.5 text-xs"
            title="Expand Document Viewer"
          >
            <PanelLeftOpen className="w-4 h-4 text-brand-600 dark:text-brand-400" />
            <span>Show Document Viewer</span>
          </button>
        )}

        {/* ======================================================== */}
        {/* RIGHT PANE: Existing Persistent Conversation / Chat UI   */}
        {/* ======================================================== */}
        <div
          style={{
            width: isViewerCollapsed ? "100%" : `${100 - splitRatio}%`,
          }}
          className={`h-full flex-col overflow-hidden bg-slate-50/50 dark:bg-slate-900/40 ${
            mobileTab === "chat" ? "flex flex-1" : "hidden lg:flex"
          }`}
        >
          <ChatArea />
        </div>
      </div>
    </div>
  );
}
