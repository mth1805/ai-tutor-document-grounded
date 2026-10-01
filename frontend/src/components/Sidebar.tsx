"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  FolderKanban,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  MessageSquare,
  Layers,
  Sparkles,
  Loader2,
  AlertCircle,
  LogIn,
  ChevronDown,
  FileText,
} from "lucide-react";
import { apiClient, Workspace, Conversation } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { DocumentManager } from "@/components/DocumentManager";
import { useDocument } from "@/lib/document-context";

export function Sidebar() {
  const router = useRouter();
  const params = useParams();
  const activeWorkspaceId = (params?.workspaceId as string) || null;
  const activeConversationId = (params?.conversationId as string) || null;

  const { user, token, isLoading: authLoading } = useAuth();

  // Safely consume document context for split view synchronization
  let selectedDocument = null;
  let setSelectedDocument: (doc: any) => void = () => {};
  let setIsViewerCollapsed: (val: any) => void = () => {};
  try {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const docCtx = useDocument();
    selectedDocument = docCtx.selectedDocument;
    setSelectedDocument = docCtx.setSelectedDocument;
    setIsViewerCollapsed = docCtx.setIsViewerCollapsed;
  } catch {
    // Graceful fallback if rendered outside DocumentProvider
  }

  // Workspaces State
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [isWsLoading, setIsWsLoading] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);
  const [isCreatingWs, setIsCreatingWs] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [isCreatingWsSubmitting, setIsCreatingWsSubmitting] = useState(false);
  const [editingWsId, setEditingWsId] = useState<string | null>(null);
  const [editingWsName, setEditingWsName] = useState("");

  // Conversations State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isConvLoading, setIsConvLoading] = useState(false);
  const [convError, setConvError] = useState<string | null>(null);
  const [isCreatingConv, setIsCreatingConv] = useState(false);
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editingConvTitle, setEditingConvTitle] = useState("");
  const [isRenameConvSubmitting, setIsRenameConvSubmitting] = useState(false);

  // Tab State: Threads vs Documents (Phase 4)
  const [sidebarTab, setSidebarTab] = useState<"threads" | "documents">("threads");
  const [docCount, setDocCount] = useState(0);

  // Fetch workspaces
  const fetchWorkspaces = useCallback(async () => {
    if (!token) return;
    setIsWsLoading(true);
    setWsError(null);
    try {
      const data = await apiClient.listWorkspaces(token);
      setWorkspaces(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load workspaces";
      setWsError(msg);
    } finally {
      setIsWsLoading(false);
    }
  }, [token]);

  // Fetch conversations for active workspace
  const fetchConversations = useCallback(async () => {
    if (!token || !activeWorkspaceId) {
      setConversations([]);
      return;
    }
    setIsConvLoading(true);
    setConvError(null);
    try {
      const data = await apiClient.listConversations(activeWorkspaceId, token);
      setConversations(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load conversations";
      setConvError(msg);
    } finally {
      setIsConvLoading(false);
    }
  }, [token, activeWorkspaceId]);

  useEffect(() => {
    if (token) {
      fetchWorkspaces();
    } else {
      setWorkspaces([]);
      setConversations([]);
    }
  }, [token, fetchWorkspaces]);

  useEffect(() => {
    if (token && activeWorkspaceId) {
      fetchConversations();
    }
  }, [token, activeWorkspaceId, fetchConversations]);

  // Handle Workspace Creation
  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWorkspaceName.trim() || !token) return;

    setIsCreatingWsSubmitting(true);
    setWsError(null);
    try {
      const created = await apiClient.createWorkspace(
        { name: newWorkspaceName.trim() },
        token
      );
      setWorkspaces((prev) => [created, ...prev]);
      setNewWorkspaceName("");
      setIsCreatingWs(false);
      router.push(`/workspace/${created.id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to create workspace";
      setWsError(msg);
    } finally {
      setIsCreatingWsSubmitting(false);
    }
  };

  // Handle Workspace Rename
  const handleRenameWorkspace = async (workspaceId: string) => {
    if (!editingWsName.trim() || !token) return;

    try {
      const updated = await apiClient.updateWorkspace(
        workspaceId,
        { name: editingWsName.trim() },
        token
      );
      setWorkspaces((prev) =>
        prev.map((ws) => (ws.id === workspaceId ? updated : ws))
      );
      setEditingWsId(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to rename workspace";
      setWsError(msg);
    }
  };

  // Handle Workspace Deletion
  const handleDeleteWorkspace = async (workspaceId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return;
    if (!confirm("Are you sure you want to delete this workspace?")) return;

    try {
      await apiClient.deleteWorkspace(workspaceId, token);
      setWorkspaces((prev) => prev.filter((ws) => ws.id !== workspaceId));
      if (activeWorkspaceId === workspaceId) {
        router.push("/");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete workspace";
      setWsError(msg);
    }
  };

  // Handle Conversation Creation (+ New Chat)
  const handleCreateConversation = async () => {
    if (!activeWorkspaceId || !token || isCreatingConv) return;

    setIsCreatingConv(true);
    setConvError(null);
    try {
      const created = await apiClient.createConversation(
        activeWorkspaceId,
        { title: "New Conversation" },
        token
      );
      setConversations((prev) => [created, ...prev]);
      router.push(`/workspace/${activeWorkspaceId}/c/${created.id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to create conversation";
      setConvError(msg);
    } finally {
      setIsCreatingConv(false);
    }
  };

  // Handle Conversation Rename
  const handleRenameConversation = async (conversationId: string) => {
    if (!editingConvTitle.trim() || !token) return;

    setIsRenameConvSubmitting(true);
    setConvError(null);
    try {
      const updated = await apiClient.updateConversation(
        conversationId,
        { title: editingConvTitle.trim() },
        token
      );
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updated : c))
      );
      setEditingConvId(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to rename conversation";
      setConvError(msg);
    } finally {
      setIsRenameConvSubmitting(false);
    }
  };

  // Handle Conversation Deletion
  const handleDeleteConversation = async (
    conversationId: string,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();
    if (!token) return;
    if (!confirm("Are you sure you want to delete this conversation?")) return;

    setConvError(null);
    try {
      await apiClient.deleteConversation(conversationId, token);
      setConversations((prev) => prev.filter((c) => c.id !== conversationId));
      if (activeConversationId === conversationId) {
        router.push(`/workspace/${activeWorkspaceId}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete conversation";
      setConvError(msg);
    }
  };

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId);

  return (
    <aside className="w-64 bg-white dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800 flex flex-col h-full select-none shrink-0 transition-colors">
      {/* 1. Workspaces Section Header */}
      <div className="p-3 border-b border-slate-200 dark:border-slate-800/80">
        <div className="flex items-center justify-between px-1 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Workspace
          </span>
          {user && (
            <button
              onClick={() => {
                setIsCreatingWs(true);
                setNewWorkspaceName("");
              }}
              className="flex items-center space-x-1 text-xs text-brand-600 hover:text-brand-700 dark:text-brand-400 dark:hover:text-brand-300 transition-colors font-medium"
              title="Create Workspace"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New</span>
            </button>
          )}
        </div>

        {/* Inline Create Workspace Input */}
        {isCreatingWs && (
          <form onSubmit={handleCreateWorkspace} className="mb-2">
            <div className="flex items-center space-x-1 bg-slate-50 dark:bg-slate-900 border border-brand-500/50 rounded-lg p-1">
              <input
                type="text"
                autoFocus
                placeholder="Workspace name..."
                value={newWorkspaceName}
                onChange={(e) => setNewWorkspaceName(e.target.value)}
                className="w-full bg-transparent px-2 py-1 text-xs text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={isCreatingWsSubmitting || !newWorkspaceName.trim()}
                className="p-1 rounded bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40"
              >
                {isCreatingWsSubmitting ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Check className="w-3 h-3" />
                )}
              </button>
              <button
                type="button"
                onClick={() => setIsCreatingWs(false)}
                className="p-1 rounded text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </form>
        )}

        {/* Workspace Dropdown/Selection Pill */}
        {authLoading ? (
          <div className="py-2 text-xs text-slate-500 flex items-center space-x-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-brand-600 dark:text-brand-400" />
            <span>Loading...</span>
          </div>
        ) : !user ? (
          <Link
            href="/login"
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-900 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 text-xs transition-colors"
          >
            <div className="flex items-center space-x-2">
              <LogIn className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400" />
              <span>Sign In to Select</span>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500" />
          </Link>
        ) : workspaces.length === 0 ? (
          <button
            onClick={() => setIsCreatingWs(true)}
            className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded-lg border border-dashed border-slate-300 dark:border-slate-800 text-brand-600 dark:text-brand-400 hover:border-brand-500/50 text-xs transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Create First Workspace</span>
          </button>
        ) : (
          <div className="space-y-1 max-h-36 overflow-y-auto pr-0.5">
            {workspaces.map((ws) => {
              const isSelected = activeWorkspaceId === ws.id;
              const isEditing = editingWsId === ws.id;

              return (
                <div
                  key={ws.id}
                  onClick={() => {
                    if (!isEditing && activeWorkspaceId !== ws.id) {
                      router.push(`/workspace/${ws.id}`);
                    }
                  }}
                  className={`group flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all cursor-pointer ${
                    isSelected
                      ? "bg-brand-50 dark:bg-brand-500/20 text-brand-700 dark:text-brand-200 border border-brand-200 dark:border-brand-500/40 font-medium"
                      : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-900 hover:text-slate-900 dark:hover:text-slate-100 border border-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate flex-1 min-w-0">
                    <FolderKanban
                      className={`w-3.5 h-3.5 shrink-0 ${
                        isSelected ? "text-brand-600 dark:text-brand-400" : "text-slate-400 dark:text-slate-500"
                      }`}
                    />
                    {isEditing ? (
                      <div
                        className="flex items-center space-x-1 flex-1 mr-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="text"
                          autoFocus
                          value={editingWsName}
                          onChange={(e) => setEditingWsName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleRenameWorkspace(ws.id);
                            if (e.key === "Escape") setEditingWsId(null);
                          }}
                          className="w-full bg-white dark:bg-slate-950 px-1.5 py-0.5 rounded border border-brand-500 text-xs text-slate-900 dark:text-slate-100 focus:outline-none"
                        />
                        <button
                          onClick={() => handleRenameWorkspace(ws.id)}
                          className="p-1 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300"
                        >
                          <Check className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => setEditingWsId(null)}
                          className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ) : (
                      <span className="truncate">{ws.name}</span>
                    )}
                  </div>

                  {!isEditing && (
                    <div
                      className="flex items-center space-x-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={() => {
                          setEditingWsId(ws.id);
                          setEditingWsName(ws.name);
                        }}
                        className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded hover:bg-slate-200 dark:hover:bg-slate-800"
                        title="Rename"
                      >
                        <Edit2 className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => handleDeleteWorkspace(ws.id, e)}
                        className="p-1 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 rounded hover:bg-slate-200 dark:hover:bg-slate-800"
                        title="Delete"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {wsError && (
          <div className="p-2 mt-2 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/50 flex items-start space-x-2 text-[11px] text-rose-600 dark:text-rose-300">
            <AlertCircle className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0 mt-0.5" />
            <span className="truncate">{wsError}</span>
          </div>
        )}
      </div>

      {/* 2. Conversations / Documents Section */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Tab Switcher */}
        {activeWorkspaceId && (
          <div className="px-3 pt-2 shrink-0">
            <div className="flex bg-slate-100 dark:bg-slate-900/80 p-0.5 rounded-lg border border-slate-200 dark:border-slate-800">
              <button
                type="button"
                onClick={() => setSidebarTab("threads")}
                className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                  sidebarTab === "threads"
                    ? "bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-300 shadow-sm"
                    : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span>Threads</span>
                {conversations.length > 0 && (
                  <span className="text-[10px] bg-slate-200 dark:bg-slate-950 px-1 rounded text-slate-600 dark:text-slate-400">
                    {conversations.length}
                  </span>
                )}
              </button>
              <button
                type="button"
                onClick={() => setSidebarTab("documents")}
                className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                  sidebarTab === "documents"
                    ? "bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-300 shadow-sm"
                    : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Docs</span>
                {docCount > 0 && (
                  <span className="text-[10px] bg-slate-200 dark:bg-slate-950 px-1 rounded text-slate-600 dark:text-slate-400">
                    {docCount}
                  </span>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeWorkspaceId && sidebarTab === "documents" ? (
          <div className="flex-1 p-3 overflow-hidden min-h-0">
            <DocumentManager
              workspaceId={activeWorkspaceId}
              onDocumentCountChange={setDocCount}
              compact
              onSelectDocument={(doc) => {
                setSelectedDocument(doc);
                setIsViewerCollapsed(false);
              }}
              selectedDocumentId={selectedDocument?.id}
            />
          </div>
        ) : (
          <>
            {/* New Chat Button */}
            {activeWorkspaceId && (
              <div className="p-3 pb-1 shrink-0">
                <button
                  onClick={handleCreateConversation}
                  disabled={isCreatingConv}
                  className="w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white font-medium text-xs shadow-md shadow-brand-500/10 transition-all active:scale-[0.98] disabled:opacity-50"
                  title="Start a new persistent conversation thread"
                >
                  {isCreatingConv ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Creating Chat...</span>
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4" />
                      <span>New Chat</span>
                    </>
                  )}
                </button>
              </div>
            )}

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
          <div className="flex items-center justify-between px-1 mb-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Conversations
            </span>
            {conversations.length > 0 && (
              <span className="text-[10px] text-slate-500 dark:text-slate-400">
                {conversations.length}
              </span>
            )}
          </div>

          {!activeWorkspaceId ? (
            <div className="py-8 px-2 text-center text-slate-400 dark:text-slate-500 text-xs">
              <MessageSquare className="w-6 h-6 mx-auto mb-2 opacity-40" />
              <span>Select a workspace to view conversation threads</span>
            </div>
          ) : isConvLoading ? (
            <div className="py-6 flex flex-col items-center justify-center text-slate-500 text-xs space-y-2">
              <Loader2 className="w-4 h-4 animate-spin text-brand-600 dark:text-brand-400" />
              <span>Loading conversations...</span>
            </div>
          ) : conversations.length === 0 ? (
            <div className="py-8 px-3 text-center space-y-2 border border-dashed border-slate-300 dark:border-slate-800/80 rounded-xl">
              <MessageSquare className="w-6 h-6 text-slate-400 dark:text-slate-600 mx-auto" />
              <p className="text-xs font-medium text-slate-700 dark:text-slate-300">
                No conversations yet
              </p>
              <p className="text-[11px] text-slate-500">
                Click &quot;New Chat&quot; to start your first thread in this workspace
              </p>
            </div>
          ) : (
            conversations.map((conv) => {
              const isSelected = activeConversationId === conv.id;
              const isEditing = editingConvId === conv.id;

              return (
                <div
                  key={conv.id}
                  onClick={() => {
                    if (!isEditing && activeConversationId !== conv.id) {
                      router.push(
                        `/workspace/${activeWorkspaceId}/c/${conv.id}`
                      );
                    }
                  }}
                  className={`group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs transition-all cursor-pointer ${
                    isSelected
                      ? "bg-brand-50 dark:bg-slate-800 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-500/30 font-medium"
                      : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-900 hover:text-slate-900 dark:hover:text-slate-100 border border-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate flex-1 min-w-0">
                    <MessageSquare
                      className={`w-3.5 h-3.5 shrink-0 ${
                        isSelected ? "text-brand-600 dark:text-brand-400" : "text-slate-400 dark:text-slate-500"
                      }`}
                    />
                    {isEditing ? (
                      <div
                        className="flex items-center space-x-1 flex-1 mr-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="text"
                          autoFocus
                          value={editingConvTitle}
                          onChange={(e) => setEditingConvTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleRenameConversation(conv.id);
                            if (e.key === "Escape") setEditingConvId(null);
                          }}
                          className="w-full bg-white dark:bg-slate-950 px-1.5 py-0.5 rounded border border-brand-500 text-xs text-slate-900 dark:text-slate-100 focus:outline-none"
                        />
                        <button
                          onClick={() => handleRenameConversation(conv.id)}
                          disabled={isRenameConvSubmitting}
                          className="p-1 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300"
                        >
                          <Check className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => setEditingConvId(null)}
                          className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ) : (
                      <span className="truncate">{conv.title}</span>
                    )}
                  </div>

                  {!isEditing && (
                    <div
                      className="flex items-center space-x-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={() => {
                          setEditingConvId(conv.id);
                          setEditingConvTitle(conv.title);
                        }}
                        className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded hover:bg-slate-200 dark:hover:bg-slate-800"
                        title="Rename Chat"
                      >
                        <Edit2 className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => handleDeleteConversation(conv.id, e)}
                        className="p-1 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 rounded hover:bg-slate-200 dark:hover:bg-slate-800"
                        title="Delete Chat"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          )}

          {convError && (
            <div className="p-2 mt-2 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/50 flex items-start space-x-2 text-[11px] text-rose-600 dark:text-rose-300">
              <AlertCircle className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0 mt-0.5" />
              <span className="truncate">{convError}</span>
            </div>
          )}
        </div>
          </>
        )}
      </div>

      {/* 3. Sidebar Footer */}
      <div className="p-3 border-t border-slate-200 dark:border-slate-800/80 bg-slate-50 dark:bg-slate-950/60">
        <div className="flex items-center space-x-2 text-[11px] text-slate-500 dark:text-slate-400 px-1">
          <Layers className="w-3.5 h-3.5 text-slate-400" />
          <span>Multi-Thread History</span>
        </div>
        <div className="mt-1 flex items-center space-x-2 text-[10px] text-slate-500 dark:text-slate-400 px-1">
          <Sparkles className="w-3 h-3 text-brand-600 dark:text-brand-400" />
          <span>Persistent across reloads</span>
        </div>
      </div>
    </aside>
  );
}
