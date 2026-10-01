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
  Bot,
  FileText,
} from "lucide-react";
import { apiClient, Message } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export function ChatArea() {
  const router = useRouter();
  const params = useParams();
  const workspaceId = (params?.workspaceId as string) || null;
  const conversationId = (params?.conversationId as string) || null;

  const { user, token } = useAuth();

  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isCreatingConv, setIsCreatingConv] = useState(false);

  // Toggle for testing assistant message persistence
  const [sendAsRole, setSendAsRole] = useState<"user" | "assistant">("user");

  const messagesEndRef = useRef<HTMLDivElement>(null);

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
  }, [messages]);

  // Handle message sending (Phase 3 persistence)
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const content = inputValue.trim();
    if (!content || !conversationId || !token || isSending) return;

    setIsSending(true);
    setError(null);

    try {
      const newMsg = await apiClient.createMessage(
        conversationId,
        { role: sendAsRole, content },
        token
      );
      setMessages((prev) => [...prev, newMsg]);
      setInputValue("");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to send message";
      setError(msg);
    } finally {
      setIsSending(false);
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
    <main className="flex-1 flex flex-col h-full bg-slate-50/50 dark:bg-slate-900/60 relative overflow-hidden transition-colors">
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

          <span className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:inline">
            Phase 4: Document Storage & Multi-Thread Active
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
                <MessageSquare className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                Conversation Thread Ready
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Send your first message below. All messages persist in PostgreSQL
                across page reloads. Real AI generation activates in Phase 8.
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((msg) => {
                const isUser = msg.role === "user";
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
                      className={`max-w-[80%] sm:max-w-[70%] rounded-2xl px-4 py-3 text-xs leading-relaxed shadow-sm ${
                        isUser
                          ? "bg-brand-50 dark:bg-brand-600/20 text-brand-950 dark:text-slate-100 border border-brand-200 dark:border-brand-500/30 rounded-tr-none"
                          : "bg-white dark:bg-slate-950/80 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-800 rounded-tl-none"
                      }`}
                    >
                      <div className="flex items-center justify-between space-x-3 mb-1 text-[10px] text-slate-500 dark:text-slate-400">
                        <span className="font-semibold uppercase tracking-wider">
                          {isUser ? "You" : "AI Tutor (Test Record)"}
                        </span>
                        <span>
                          {new Date(msg.created_at).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                );
              })}
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
                      Document Storage & Study Threads
                    </h1>
                    <p className="text-xs text-slate-600 dark:text-slate-400 max-w-lg leading-relaxed">
                      Upload your PDF/DOCX course materials and study notes below. All documents persist securely in private storage and anchor future AI learning.
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
                  <AlertCircle className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
                  <span>
                    Phase 4 Active: Original file uploads persist in private storage with workspace-level isolation.
                  </span>
                </div>
              </div>

              {/* Split-View Guidance Card */}
              <div className="p-6 rounded-2xl bg-white dark:bg-slate-950/70 border border-slate-200 dark:border-slate-800/80 shadow-sm dark:shadow-lg space-y-4">
                <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200 flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                  <span>Split-View Document & Chat Workspace</span>
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600 dark:text-slate-400">
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <span className="font-semibold text-slate-800 dark:text-slate-300">Left Pane: Document Viewer</span>
                    <p className="text-[11px] leading-relaxed">
                      Select any uploaded document (PDF, Word, TXT, Images) from the left panel to preview it side-by-side. Drag the divider to resize.
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <span className="font-semibold text-slate-800 dark:text-slate-300">Right Pane: Persistent Chat</span>
                    <p className="text-[11px] leading-relaxed">
                      Start or switch conversations to chat with the AI Tutor while keeping your study notes open.
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
                <AlertCircle className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
                <span>
                  Phase 4 Active: Workspace document storage and multi-thread persistence enabled.
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Input Area (Active when inside a conversation) */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm shrink-0">
        <div className="max-w-3xl mx-auto">
          {/* Phase 3 Test Role Switcher Banner */}
          {conversationId && (
            <div className="flex items-center justify-between mb-2 px-1 text-[11px] text-slate-500 dark:text-slate-400">
              <div className="flex items-center space-x-2">
                <span>Post message as:</span>
                <button
                  type="button"
                  onClick={() => setSendAsRole("user")}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                    sendAsRole === "user"
                      ? "bg-brand-600 text-white"
                      : "bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-transparent"
                  }`}
                >
                  User
                </button>
                <button
                  type="button"
                  onClick={() => setSendAsRole("assistant")}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                    sendAsRole === "assistant"
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-transparent"
                  }`}
                  title="Phase 3 test helper: verify assistant message persistence without real AI"
                >
                  <Bot className="w-3 h-3 inline mr-1" />
                  Assistant (Test Note)
                </button>
              </div>
              <span className="text-slate-400 dark:text-slate-500 hidden sm:inline">
                Real AI generation activates in Phase 8
              </span>
            </div>
          )}

          <form onSubmit={handleSendMessage} className="relative flex items-center">
            <input
              type="text"
              value={inputValue}
              disabled={!conversationId}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                conversationId
                  ? `Type a ${sendAsRole} message to test persistence...`
                  : "Select or start a conversation to send messages..."
              }
              className="w-full pl-4 pr-12 py-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 focus:border-brand-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || !conversationId || isSending}
              className="absolute right-2 p-2 rounded-lg bg-brand-600 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-brand-500 active:scale-95 transition-all"
            >
              {isSending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </form>

          <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 mt-2 px-1">
            <span>AI Tutor Assistant • Conversation Persistence</span>
            <span>URL-Driven • PostgreSQL Backed</span>
          </div>
        </div>
      </div>
    </main>
  );
}
