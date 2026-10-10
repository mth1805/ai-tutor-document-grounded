"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Send,
  MessageSquare,
  Sparkles,
  AlertCircle,
  FolderKanban,
  User,
  GraduationCap,
  Loader2,
  Plus,
  FileText,
  Square,
  BookOpen,
  Compass,
  Lightbulb,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import { apiClient, Message, Citation, WebCitation } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useDocument } from "@/lib/document-context";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import {
  deduplicateCitations,
  preprocessMarkdownWithCitations,
  getCitationKey,
  formatCitationPages,
  ParsedCitationLink,
} from "@/lib/citation-utils";

type ChatMode = "Light Guidance" | "Detailed Guidance" | "Full Solution";

export function ChatArea() {
  const router = useRouter();
  const params = useParams();
  const workspaceId = (params?.workspaceId as string) || null;
  const conversationId = (params?.conversationId as string) || null;

  const { user, token } = useAuth();
  const { setSelectedDocument, setIsViewerCollapsed } = useDocument();

  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState<string | null>(null);
  const [isCreatingConv, setIsCreatingConv] = useState(false);

  // Pedagogical Chat Mode
  const [chatMode, setChatMode] = useState<ChatMode>("Detailed Guidance");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Fetch messages when conversation changes
  const fetchMessages = useCallback(async () => {
    if (!token || !conversationId) {
      setMessages([]);
      return;
    }
    setIsLoadingMessages(true);
    setError(null);
    try {
      const data = await apiClient.listMessages(conversationId, token);
      setMessages(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load messages";
      setError(msg);
    } finally {
      setIsLoadingMessages(false);
    }
  }, [token, conversationId]);

  useEffect(() => {
    if (token && conversationId) {
      fetchMessages();
    } else {
      setMessages([]);
    }
  }, [token, conversationId, fetchMessages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating, generationStatus]);

  // Handle grounded streaming chat response (Phase 8)
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const content = inputValue.trim();
    if (!content || !conversationId || !token || isGenerating) return;

    setInputValue("");
    setError(null);
    setIsGenerating(true);
    setGenerationStatus("Searching uploaded documents...");

    // 1. Optimistic User Message
    const tempUserMsgId = `temp-user-${Date.now()}`;
    const userMsg: Message = {
      id: tempUserMsgId,
      conversation_id: conversationId,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };

    // 2. Optimistic Assistant Message placeholder
    const tempAsstMsgId = `temp-asst-${Date.now()}`;
    const asstMsg: Message = {
      id: tempAsstMsgId,
      conversation_id: conversationId,
      role: "assistant",
      content: "",
      citations: [],
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg, asstMsg]);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      let accumulatedContent = "";

      await apiClient.streamChat(
        conversationId,
        {
          content,
          chat_mode: chatMode,
          workspace_id: workspaceId || undefined,
        },
        token,
        {
          onStatus: (_status, message) => {
            setGenerationStatus(message || "Analyzing documents...");
          },
          onToken: (tokenChunk) => {
            accumulatedContent += tokenChunk;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === tempAsstMsgId
                  ? { ...m, content: accumulatedContent }
                  : m
              )
            );
          },
          onDone: (payload) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === tempAsstMsgId
                  ? {
                      ...m,
                      id: payload.message_id || m.id,
                      content: payload.content || accumulatedContent,
                      citations: payload.citations || [],
                      web_sources: payload.web_sources || [],
                    }
                  : m
              )
            );
            setGenerationStatus(null);
          },
          onError: (errMsg) => {
            setError(errMsg);
            setGenerationStatus(null);
          },
        },
        abortController.signal
      );
    } catch (err: unknown) {
      if (!abortController.signal.aborted) {
        const msg = err instanceof Error ? err.message : "Failed to generate response";
        setError(msg);
      }
    } finally {
      setIsGenerating(false);
      setGenerationStatus(null);
      abortControllerRef.current = null;
    }
  };

  // Stop Generation handler
  const handleStopGenerating = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsGenerating(false);
      setGenerationStatus(null);
      abortControllerRef.current = null;
    }
  };

  // Open cited document in the document viewer
  const handleOpenCitation = async (citation: Citation | ParsedCitationLink) => {
    if (!token) return;
    try {
      setIsViewerCollapsed(false);
      let docId = citation.document_id;
      if (!docId && citation.document_name && workspaceId) {
        // Resolve document ID by filename from workspace documents
        try {
          const docs = await apiClient.listDocuments(workspaceId, token);
          const targetName = citation.document_name.toLowerCase().trim();
          const match = docs.find(
            (d) =>
              d.original_filename.toLowerCase().trim() === targetName ||
              targetName.includes(d.original_filename.toLowerCase().trim()) ||
              d.original_filename.toLowerCase().trim().includes(targetName)
          );
          if (match) docId = match.id;
        } catch (fetchErr) {
          console.warn("Could not resolve document list for citation name lookup", fetchErr);
        }
      }

      if (docId) {
        const doc = await apiClient.getDocument(docId, token);
        setSelectedDocument(doc);
      }
    } catch (err) {
      console.error("Failed to open cited document in viewer", err);
    }
  };

  // Helper to start conversation from empty state
  const handleStartChat = async () => {
    if (!workspaceId || !token || isCreatingConv) return;
    setIsCreatingConv(true);
    try {
      const created = await apiClient.createConversation(
        workspaceId,
        { title: "New Conversation" },
        token
      );
      router.push(`/workspace/${workspaceId}/c/${created.id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to create conversation";
      setError(msg);
    } finally {
      setIsCreatingConv(false);
    }
  };

  return (
    <main className="flex-1 min-w-0 flex flex-col h-full bg-slate-50/50 dark:bg-slate-900/60 relative overflow-hidden transition-colors">
      {/* Workspace / Conversation Sub-header */}
      {workspaceId && (
        <div className="h-10 border-b border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/40 px-4 flex items-center justify-between text-xs text-slate-600 dark:text-slate-300 shrink-0">
          <div className="flex items-center space-x-2 truncate">
            <FolderKanban className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
            <span className="font-medium text-slate-800 dark:text-slate-200 truncate">
              Workspace
            </span>
            <span className="font-mono text-[10px] text-slate-500 bg-slate-100 dark:bg-slate-900 px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-800 shrink-0">
              {workspaceId.slice(0, 8)}...
            </span>

            {conversationId && (
              <>
                <span className="text-slate-400 dark:text-slate-600">/</span>
                <MessageSquare className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
                <span className="text-slate-700 dark:text-slate-300 font-medium">Thread</span>
                <span className="font-mono text-[10px] text-slate-500 bg-slate-100 dark:bg-slate-900 px-1.5 py-0.5 rounded border border-slate-200 dark:border-slate-800 shrink-0">
                  {conversationId.slice(0, 8)}...
                </span>
              </>
            )}
          </div>

          <span className="text-[11px] text-brand-600 dark:text-brand-400 font-medium hidden sm:inline flex items-center space-x-1">
            <Sparkles className="w-3 h-3 inline mr-1" />
            Document-grounded AI Tutor with Web Fallback
          </span>
        </div>
      )}

      {/* Main View Area */}
      {conversationId ? (
        // Active Conversation View
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-4">
          {isLoadingMessages ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs space-y-2">
              <Loader2 className="w-5 h-5 animate-spin text-brand-600 dark:text-brand-400" />
              <span>Loading conversation history...</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-3">
              <div className="w-10 h-10 rounded-xl bg-brand-500/10 border border-brand-500/20 text-brand-600 dark:text-brand-400 flex items-center justify-center">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                AI Tutor Ready
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Ask any question grounded in your uploaded documents. Choose your guidance mode below (Light, Detailed, or Full Solution).
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((msg) => {
                const isUser = msg.role === "user";
                const isStreamingThis = isGenerating && msg.role === "assistant" && !msg.content;

                return (
                  <div
                    key={msg.id}
                    className={`flex items-start space-x-3 ${
                      isUser ? "flex-row-reverse space-x-reverse" : "flex-row"
                    }`}
                  >
                    {/* Avatar */}
                    <div
                      className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 shadow-sm ${
                        isUser
                          ? "bg-brand-600 text-white"
                          : "bg-gradient-to-tr from-indigo-600 to-brand-500 text-white"
                      }`}
                    >
                      {isUser ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <GraduationCap className="w-4 h-4" />
                      )}
                    </div>

                    {/* Message Bubble */}
                    <div
                      className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 text-xs leading-relaxed shadow-sm ${
                        isUser
                          ? "bg-brand-50 dark:bg-brand-600/20 text-brand-950 dark:text-slate-100 border border-brand-200 dark:border-brand-500/30 rounded-tr-none"
                          : "bg-white dark:bg-slate-950/80 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-800 rounded-tl-none"
                      }`}
                    >
                      <div className="flex items-center justify-between space-x-3 mb-1 text-[10px] text-slate-500 dark:text-slate-400">
                        <span className="font-semibold uppercase tracking-wider">
                          {isUser ? "You" : "AI Tutor"}
                        </span>
                        <span>
                          {new Date(msg.created_at).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>

                      {/* Content */}
                      {isStreamingThis ? (
                        <div className="flex items-center space-x-2 py-1 text-slate-500 text-xs">
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-brand-600 dark:text-brand-400" />
                          <span className="italic">{generationStatus || "Formulating grounded answer..."}</span>
                        </div>
                      ) : isUser ? (
                        <div className="whitespace-pre-wrap leading-relaxed">
                          {msg.content}
                        </div>
                      ) : (
                        <MarkdownRenderer
                          content={msg.content}
                          citations={msg.citations}
                          onOpenCitation={handleOpenCitation}
                        />
                      )}

                      {/* Citations & Document Evidence Section */}
                      {!isUser && (() => {
                        const { inlineCitationKeys } = preprocessMarkdownWithCitations(
                          msg.content,
                          msg.citations || []
                        );
                        const deduplicated = deduplicateCitations(msg.citations || []);
                        const bottomCitations = deduplicated.filter(
                          (c) => !inlineCitationKeys.has(getCitationKey(c))
                        );
                        const webSources: WebCitation[] = msg.web_sources || [];

                        if (bottomCitations.length === 0 && webSources.length === 0) return null;

                        return (
                          <div className="mt-3 pt-2.5 border-t border-slate-100 dark:border-slate-800/80 space-y-2">
                            {/* Document Evidence */}
                            {bottomCitations.length > 0 && (
                              <div className="space-y-1.5">
                                <div className="flex items-center space-x-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                                  <BookOpen className="w-3 h-3 text-brand-600 dark:text-brand-400" />
                                  <span>Document Evidence ({bottomCitations.length})</span>
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {bottomCitations.map((c, idx) => (
                                    <span
                                      key={c.chunk_id || `${c.document_id}-${c.page_start}-${idx}`}
                                      className="relative inline-flex group"
                                    >
                                      <button
                                        type="button"
                                        onClick={() => handleOpenCitation(c)}
                                        aria-label={`${c.document_name}, ${c.page_start === c.page_end ? `Page ${c.page_start}` : `Pages ${c.page_start}–${c.page_end}`}`}
                                        className="inline-flex items-center gap-1 px-1.5 py-1 rounded-md bg-brand-50/80 hover:bg-brand-100 dark:bg-brand-950/40 dark:hover:bg-brand-900/60 border border-brand-200/80 dark:border-brand-800/60 text-[11px] font-medium text-brand-800 dark:text-brand-300 transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-500"
                                      >
                                        <span aria-hidden="true">{"\u{1F4C4}"}</span>
                                        <span className="font-mono">
                                          {formatCitationPages(c.page_start, c.page_end)}
                                        </span>
                                      </button>
                                      <span
                                        role="tooltip"
                                        className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:flex group-focus-within:flex flex-col items-center z-30"
                                      >
                                        <span className="bg-slate-900/95 dark:bg-slate-950/95 text-slate-100 border border-slate-700/80 rounded-md px-2.5 py-1.5 shadow-lg text-[10px] whitespace-nowrap leading-tight">
                                          <span className="block font-semibold text-white max-w-[220px] truncate">{c.document_name}</span>
                                          <span className="block text-slate-400 text-[9px] mt-0.5">
                                            {c.page_start === c.page_end ? `Page ${c.page_start}` : `Pages ${c.page_start}–${c.page_end}`}
                                          </span>
                                        </span>
                                      </span>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Web Sources Panel */}
                            {webSources.length > 0 && (
                              <div className="space-y-1.5">
                                <div className="flex items-center space-x-1.5 text-[10px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                                  <ExternalLink className="w-3 h-3" />
                                  <span>Web Sources ({webSources.length})</span>
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {webSources.map((ws, idx) => (
                                    <span
                                      key={`web-${ws.url}-${idx}`}
                                      className="relative inline-flex group"
                                    >
                                      <a
                                        href={ws.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        aria-label={`${ws.title} — ${ws.domain}`}
                                        className="inline-flex items-center gap-1 px-1.5 py-1 rounded-md bg-amber-50/80 hover:bg-amber-100 dark:bg-amber-950/30 dark:hover:bg-amber-900/50 border border-amber-200/80 dark:border-amber-800/50 text-[11px] font-medium text-amber-800 dark:text-amber-300 transition-all focus:outline-none focus-visible:ring-1 focus-visible:ring-amber-500"
                                      >
                                        <span aria-hidden="true">🌐</span>
                                        <span className="font-mono max-w-[120px] truncate">{ws.domain}</span>
                                      </a>
                                      <span
                                        role="tooltip"
                                        className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:flex group-focus-within:flex flex-col items-center z-30"
                                      >
                                        <span className="bg-slate-900/95 dark:bg-slate-950/95 text-slate-100 border border-slate-700/80 rounded-md px-2.5 py-1.5 shadow-lg text-[10px] whitespace-nowrap leading-tight">
                                          <span className="block font-semibold text-white max-w-[260px] truncate">{ws.title || ws.domain}</span>
                                          <span className="block text-amber-400 text-[9px] mt-0.5 font-medium">{ws.domain}</span>
                                        </span>
                                        <span className="w-1.5 h-1.5 -mt-0.5 bg-slate-900/95 border-r border-b border-slate-700/80 rotate-45" />
                                      </span>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                );
              })}

              {/* Streaming in progress indicator bar */}
              {isGenerating && generationStatus && (
                <div className="flex items-center justify-center space-x-2 py-1 text-xs text-brand-600 dark:text-brand-400">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span className="font-medium">{generationStatus}</span>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {error && (
            <div className="max-w-md mx-auto p-3 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800/60 flex items-start space-x-2.5 text-xs text-rose-600 dark:text-rose-300">
              <AlertCircle className="w-4 h-4 text-rose-500 dark:text-rose-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      ) : (
        // No Conversation Selected: Workspace Overview / Welcome View
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 flex flex-col justify-start max-w-4xl mx-auto w-full">
          {workspaceId ? (
            <div className="space-y-6">
              {/* Workspace Header Card */}
              <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-100 via-white to-slate-50 dark:from-slate-900 dark:to-slate-950 border border-slate-200 dark:border-slate-800 shadow-md dark:shadow-xl space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="inline-flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-brand-600 dark:text-brand-400">
                      <FolderKanban className="w-3.5 h-3.5" />
                      <span>Workspace Learning Center</span>
                    </div>
                    <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-slate-100">
                      Document-Grounded AI Learning
                    </h1>
                    <p className="text-xs text-slate-600 dark:text-slate-400 max-w-lg leading-relaxed">
                      Upload your PDF/DOCX course materials and start study threads. All answers are strictly grounded in your documents with verifiable citations.
                    </p>
                  </div>

                  {user && (
                    <button
                      onClick={handleStartChat}
                      disabled={isCreatingConv}
                      className="inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white font-medium text-xs shadow-lg shadow-brand-500/20 transition-all active:scale-95 disabled:opacity-50 shrink-0"
                    >
                      {isCreatingConv ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Creating...</span>
                        </>
                      ) : (
                        <>
                          <Plus className="w-4 h-4" />
                          <span>Start New Chat Thread</span>
                        </>
                      )}
                    </button>
                  )}
                </div>

                <div className="flex items-center space-x-2 pt-2 border-t border-slate-200 dark:border-slate-800/80 text-[11px] text-slate-500 dark:text-slate-400">
                  <Sparkles className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
                  <span>
                    Answers stream with source citations from relevant evidence.
                  </span>
                </div>
              </div>

              {/* Split-View Guidance Card */}
              <div className="p-6 rounded-2xl bg-white dark:bg-slate-950/70 border border-slate-200 dark:border-slate-800/80 shadow-sm dark:shadow-lg space-y-4">
                <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200 flex items-center space-x-2">
                  <Compass className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                  <span>How AI Tutor Works</span>
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-600 dark:text-slate-400">
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <div className="flex items-center space-x-1.5 font-semibold text-slate-800 dark:text-slate-300">
                      <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
                      <span>1. Light Guidance</span>
                    </div>
                    <p className="text-[11px] leading-relaxed">
                      Concise hints and guided next steps. Ideal for active problem solving.
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <div className="flex items-center space-x-1.5 font-semibold text-slate-800 dark:text-slate-300">
                      <Compass className="w-3.5 h-3.5 text-brand-500" />
                      <span>2. Detailed Guidance</span>
                    </div>
                    <p className="text-[11px] leading-relaxed">
                      Step-by-step conceptual walkthroughs with deep educational explanation.
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <div className="flex items-center space-x-1.5 font-semibold text-slate-800 dark:text-slate-300">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                      <span>3. Full Solution</span>
                    </div>
                    <p className="text-[11px] leading-relaxed">
                      Complete answers and solutions grounded in the uploaded course documents.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-2xl w-full text-center space-y-6 mx-auto my-auto">
              <div className="inline-flex p-3 rounded-2xl bg-gradient-to-tr from-brand-500/20 via-indigo-500/10 to-transparent border border-brand-500/30 text-brand-600 dark:text-brand-400 shadow-xl shadow-brand-500/5 mb-2">
                <Sparkles className="w-8 h-8" />
              </div>

              <div className="space-y-2">
                <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
                  Welcome to AI Tutor Assistant
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400 max-w-lg mx-auto leading-relaxed">
                  Your enterprise-grade, document-grounded learning partner. Sign in and select a workspace to organize your learning materials.
                </p>
              </div>

              <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400">
                <Sparkles className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
                <span>
                  Answers stream with verifiable document citations.
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Input Area (Active when inside a conversation) */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm shrink-0">
        <div className="max-w-3xl mx-auto space-y-2.5">
          {/* Pedagogical Guidance Mode Selector */}
          {conversationId && (
            <div className="flex items-center justify-between px-1 text-xs">
              <div className="flex items-center space-x-2">
                <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                  Mode:
                </span>
                <div className="inline-flex p-0.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  {(["Light Guidance", "Detailed Guidance", "Full Solution"] as ChatMode[]).map(
                    (mode) => {
                      const isActive = chatMode === mode;
                      return (
                        <button
                          key={mode}
                          type="button"
                          disabled={isGenerating}
                          onClick={() => setChatMode(mode)}
                          className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                            isActive
                              ? "bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-300 shadow-sm"
                              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                          } disabled:opacity-50`}
                        >
                          {mode === "Light Guidance" && (
                            <Lightbulb className="w-3 h-3 inline mr-1 text-amber-500" />
                          )}
                          {mode === "Detailed Guidance" && (
                            <Compass className="w-3 h-3 inline mr-1 text-brand-500" />
                          )}
                          {mode === "Full Solution" && (
                            <CheckCircle2 className="w-3 h-3 inline mr-1 text-emerald-500" />
                          )}
                          <span>{mode}</span>
                        </button>
                      );
                    }
                  )}
                </div>
              </div>

              {isGenerating && (
                <button
                  type="button"
                  onClick={handleStopGenerating}
                  className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-rose-50 hover:bg-rose-100 dark:bg-rose-950/40 dark:hover:bg-rose-900/60 border border-rose-200 dark:border-rose-800/60 text-[11px] font-medium text-rose-600 dark:text-rose-400 transition-all active:scale-95"
                >
                  <Square className="w-2.5 h-2.5 fill-current" />
                  <span>Stop Generating</span>
                </button>
              )}
            </div>
          )}

          {/* Prompt Form */}
          <form onSubmit={handleSendMessage} className="relative flex items-center">
            <input
              type="text"
              value={inputValue}
              disabled={!conversationId || isGenerating}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                conversationId
                  ? isGenerating
                    ? "AI Tutor is streaming response..."
                    : `Ask a question in ${chatMode} mode...`
                  : "Select or start a conversation to ask questions..."
              }
              className="w-full pl-4 pr-12 py-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 focus:border-brand-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || !conversationId || isGenerating}
              className="absolute right-2 p-2 rounded-lg bg-brand-600 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-brand-500 active:scale-95 transition-all"
            >
              {isGenerating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </form>

          <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 mt-1 px-1">
            <span>AI Tutor Assistant • Grounded RAG Generation</span>
            <span>Document-First • Verifiable Citations</span>
          </div>
        </div>
      </div>
    </main>
  );
}
