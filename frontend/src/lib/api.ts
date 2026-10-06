/**
 * API client abstraction for AI Tutor Assistant.
 * Centralizes all HTTP communication with the FastAPI backend.
 */
import { API_BASE_URL } from "./config";

export interface HealthStatus {
  status: string;
}

export interface Workspace {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceCreate {
  name: string;
}

export interface WorkspaceUpdate {
  name: string;
}

export interface Conversation {
  id: string;
  workspace_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationCreate {
  title?: string;
}

export interface ConversationUpdate {
  title: string;
}

export interface Citation {
  document_id: string;
  document_name: string;
  chunk_id: string;
  page_start: number;
  page_end: number;
  snippet?: string;
}

export interface WebCitation {
  source_type: "web";
  title: string;
  url: string;
  domain: string;
  snippet?: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[];
  web_sources?: WebCitation[];
  created_at: string;
}

export interface ChatStreamCallbacks {
  onStatus?: (status: string, message?: string) => void;
  onToken?: (token: string) => void;
  onDone?: (payload: {
    message_id: string;
    content: string;
    citations: Citation[];
    web_sources?: WebCitation[];
    has_sufficient_evidence: boolean;
    used_web_fallback?: boolean;
  }) => void;
  onError?: (error: string) => void;
}

export interface MessageCreate {
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[];
}

export interface DocumentItem {
  id: string;
  workspace_id: string;
  user_id: string;
  original_filename: string;
  storage_path: string;
  mime_type: string;
  file_size: number;
  status: "uploaded" | "processing" | "processed" | "failed" | string;
  embedding_status?: "pending" | "processing" | "completed" | "failed" | string | null;
  created_at: string;
  updated_at: string;
  processing_started_at?: string | null;
  processed_at?: string | null;
  processing_error?: string | null;
  embedding_started_at?: string | null;
  embedded_at?: string | null;
  embedding_error?: string | null;
  download_url?: string | null;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  workspace_id: string;
  user_id: string;
  chunk_index: number;
  content: string;
  page_number_start: number;
  page_number_end: number;
  token_count: number;
  embedding_model?: string | null;
  embedding_version?: string | null;
  embedded_at?: string | null;
  has_embedding?: boolean;
  created_at: string;
}

export interface DocumentProcessResponse {
  document_id: string;
  status: string;
  message: string;
}

export interface DocumentEmbedResponse {
  document_id: string;
  embedding_status: string;
  message: string;
}

export interface DocumentEmbeddingStatusResponse {
  document_id: string;
  embedding_status: string;
  total_chunks: number;
  embedded_chunks: number;
  embedding_model?: string | null;
  embedding_started_at?: string | null;
  embedded_at?: string | null;
  embedding_error?: string | null;
}

export interface DocumentDownloadResponse {
  download_url: string;
  expires_in: number;
}

export interface RetrievalSearchRequest {
  query: string;
  dense_top_k?: number;
  lexical_top_k?: number;
  rrf_k?: number;
  candidate_pool_size?: number;
  rerank_top_k?: number;
  relevance_threshold?: number;
}

export interface RetrievedChunkItem {
  chunk_id: string;
  document_id: string;
  content: string;
  page_number_start: number;
  page_number_end: number;
  chunk_index: number;
  dense_score?: number | null;
  lexical_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  final_rank: number;
  passed_relevance_gate: boolean;
  retrieval_sources: string[];
}

export interface RetrievalTimingMetrics {
  query_embedding_ms: number;
  dense_retrieval_ms: number;
  lexical_retrieval_ms: number;
  rrf_ms: number;
  rerank_ms: number;
  total_retrieval_ms: number;
}

export interface RetrievalSearchResponse {
  workspace_id: string;
  query: string;
  results: RetrievedChunkItem[];
  total_results: number;
  has_sufficient_evidence: boolean;
  relevance_threshold: number;
  timings: RetrievalTimingMetrics;
  diagnostics?: Record<string, unknown> | null;
}

export class ApiError extends Error {
  status?: number;
  data?: unknown;

  constructor(message: string, status?: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
  token?: string | null
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;

  const headers: Record<string, string> = {
    ...(!isFormData ? { "Content-Type": "application/json" } : {}),
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    if (res.status === 204) {
      return {} as T;
    }

    if (!res.ok) {
      let errorData: unknown;
      try {
        errorData = await res.json();
      } catch {
        errorData = await res.text();
      }
      throw new ApiError(
        `API request failed with status ${res.status}`,
        res.status,
        errorData
      );
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : "Unknown network error";
    throw new ApiError(`Network connection error: ${message}`);
  }
}

export const apiClient = {
  /**
   * Root health check probe.
   */
  async checkHealth(): Promise<HealthStatus> {
    return request<HealthStatus>("/health");
  },

  /**
   * Versioned v1 health check probe.
   */
  async checkV1Health(): Promise<HealthStatus> {
    return request<HealthStatus>("/api/v1/health");
  },

  // ==========================================
  // Workspace Endpoints
  // ==========================================

  /**
   * Retrieves all workspaces for the authenticated user.
   */
  async listWorkspaces(token?: string | null): Promise<Workspace[]> {
    return request<Workspace[]>("/api/v1/workspaces", { method: "GET" }, token);
  },

  /**
   * Creates a new workspace.
   */
  async createWorkspace(
    data: WorkspaceCreate,
    token?: string | null
  ): Promise<Workspace> {
    return request<Workspace>(
      "/api/v1/workspaces",
      {
        method: "POST",
        body: JSON.stringify(data),
      },
      token
    );
  },

  /**
   * Renames an existing workspace.
   */
  async updateWorkspace(
    workspaceId: string,
    data: WorkspaceUpdate,
    token?: string | null
  ): Promise<Workspace> {
    return request<Workspace>(
      `/api/v1/workspaces/${workspaceId}`,
      {
        method: "PATCH",
        body: JSON.stringify(data),
      },
      token
    );
  },

  /**
   * Deletes a workspace.
   */
  async deleteWorkspace(
    workspaceId: string,
    token?: string | null
  ): Promise<void> {
    return request<void>(
      `/api/v1/workspaces/${workspaceId}`,
      { method: "DELETE" },
      token
    );
  },

  // ==========================================
  // Conversation Endpoints
  // ==========================================

  /**
   * Lists all conversations for a specific workspace.
   */
  async listConversations(
    workspaceId: string,
    token?: string | null
  ): Promise<Conversation[]> {
    return request<Conversation[]>(
      `/api/v1/workspaces/${workspaceId}/conversations`,
      { method: "GET" },
      token
    );
  },

  /**
   * Creates a new conversation thread inside a workspace.
   */
  async createConversation(
    workspaceId: string,
    data: ConversationCreate = { title: "New Conversation" },
    token?: string | null
  ): Promise<Conversation> {
    return request<Conversation>(
      `/api/v1/workspaces/${workspaceId}/conversations`,
      {
        method: "POST",
        body: JSON.stringify(data),
      },
      token
    );
  },

  /**
   * Retrieves details of a specific conversation.
   */
  async getConversation(
    conversationId: string,
    token?: string | null
  ): Promise<Conversation> {
    return request<Conversation>(
      `/api/v1/conversations/${conversationId}`,
      { method: "GET" },
      token
    );
  },

  /**
   * Renames an existing conversation.
   */
  async updateConversation(
    conversationId: string,
    data: ConversationUpdate,
    token?: string | null
  ): Promise<Conversation> {
    return request<Conversation>(
      `/api/v1/conversations/${conversationId}`,
      {
        method: "PATCH",
        body: JSON.stringify(data),
      },
      token
    );
  },

  /**
   * Deletes a conversation.
   */
  async deleteConversation(
    conversationId: string,
    token?: string | null
  ): Promise<void> {
    return request<void>(
      `/api/v1/conversations/${conversationId}`,
      { method: "DELETE" },
      token
    );
  },

  // ==========================================
  // Message Endpoints
  // ==========================================

  /**
   * Lists all messages for a specific conversation in chronological order.
   */
  async listMessages(
    conversationId: string,
    token?: string | null
  ): Promise<Message[]> {
    return request<Message[]>(
      `/api/v1/conversations/${conversationId}/messages`,
      { method: "GET" },
      token
    );
  },

  /**
   * Appends and persists a new message in a conversation.
   */
  async createMessage(
    conversationId: string,
    data: MessageCreate,
    token?: string | null
  ): Promise<Message> {
    return request<Message>(
      `/api/v1/conversations/${conversationId}/messages`,
      {
        method: "POST",
        body: JSON.stringify(data),
      },
      token
    );
  },

  // ==========================================
  // Document Endpoints (Phase 4)
  // ==========================================

  /**
   * Uploads an original document to private storage and persists metadata.
   */
  async uploadDocument(
    workspaceId: string,
    file: File,
    token?: string | null
  ): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append("file", file);

    return request<DocumentItem>(
      `/api/v1/workspaces/${workspaceId}/documents`,
      {
        method: "POST",
        body: formData,
      },
      token
    );
  },

  /**
   * Lists all documents belonging to a workspace in reverse chronological order.
   */
  async listDocuments(
    workspaceId: string,
    token?: string | null
  ): Promise<DocumentItem[]> {
    return request<DocumentItem[]>(
      `/api/v1/workspaces/${workspaceId}/documents`,
      { method: "GET" },
      token
    );
  },

  /**
   * Retrieves document metadata by ID.
   */
  async getDocument(
    documentId: string,
    token?: string | null
  ): Promise<DocumentItem> {
    return request<DocumentItem>(
      `/api/v1/documents/${documentId}`,
      { method: "GET" },
      token
    );
  },

  /**
   * Deletes document metadata and its original stored file.
   */
  async deleteDocument(
    documentId: string,
    token?: string | null
  ): Promise<void> {
    return request<void>(
      `/api/v1/documents/${documentId}`,
      { method: "DELETE" },
      token
    );
  },

  /**
   * Retrieves a secure download or signed URL for a document.
   */
  async getDocumentDownloadUrl(
    documentId: string,
    token?: string | null
  ): Promise<DocumentDownloadResponse> {
    return request<DocumentDownloadResponse>(
      `/api/v1/documents/${documentId}/url`,
      { method: "GET" },
      token
    );
  },

  /**
   * Triggers asynchronous parsing, OCR, and chunking for a document.
   */
  async reprocessDocument(
    documentId: string,
    token?: string | null
  ): Promise<DocumentProcessResponse> {
    return request<DocumentProcessResponse>(
      `/api/v1/documents/${documentId}/process`,
      { method: "POST" },
      token
    );
  },

  /**
   * Retrieves parsed chunks for an authorized document.
   */
  async listDocumentChunks(
    documentId: string,
    token?: string | null
  ): Promise<DocumentChunk[]> {
    return request<DocumentChunk[]>(
      `/api/v1/documents/${documentId}/chunks`,
      { method: "GET" },
      token
    );
  },

  // ==========================================
  // Embedding Endpoints (Phase 6)
  // ==========================================

  /**
   * Triggers or restarts asynchronous BGE-M3 embedding generation for a processed document.
   */
  async embedDocument(
    documentId: string,
    token?: string | null
  ): Promise<DocumentEmbedResponse> {
    return request<DocumentEmbedResponse>(
      `/api/v1/documents/${documentId}/embed`,
      { method: "POST" },
      token
    );
  },

  /**
   * Retrieves embedding status, chunk counts, and timestamps for an authorized document.
   */
  async getDocumentEmbeddingStatus(
    documentId: string,
    token?: string | null
  ): Promise<DocumentEmbeddingStatusResponse> {
    return request<DocumentEmbeddingStatusResponse>(
      `/api/v1/documents/${documentId}/embedding-status`,
      { method: "GET" },
      token
    );
  },

  // ==========================================
  // Retrieval Endpoints (Phase 7)
  // ==========================================

  /**
   * Performs hybrid retrieval + Cross-Encoder reranking search on workspace chunks (Phase 7).
   */
  async searchRetrieval(
    workspaceId: string,
    payload: RetrievalSearchRequest,
    token?: string | null
  ): Promise<RetrievalSearchResponse> {
    return request<RetrievalSearchResponse>(
      `/api/v1/workspaces/${workspaceId}/retrieval/search`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      token
    );
  },

  // ==========================================
  // Phase 8 Grounded Chat Streaming Endpoint
  // ==========================================

  /**
   * Streams grounded AI tutor answer tokens over Server-Sent Events (SSE).
   */
  async streamChat(
    conversationId: string,
    payload: {
      content: string;
      chat_mode?: string;
      workspace_id?: string;
    },
    token: string | null | undefined,
    callbacks: ChatStreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const url = `${API_BASE_URL}/api/v1/conversations/${conversationId}/chat`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal,
    });

    if (!response.ok) {
      let errorMsg = `Server error (${response.status})`;
      try {
        const errJson = await response.json();
        errorMsg = errJson.detail || errJson.message || errorMsg;
      } catch {
        // ignore JSON parse error
      }
      callbacks.onError?.(errorMsg);
      throw new ApiError(errorMsg, response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError?.("Streaming not supported by browser environment.");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let terminalEventReceived = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";

        for (const block of blocks) {
          if (!block.trim()) continue;

          let eventName = "message";
          let dataStr = "";

          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) {
              eventName = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.slice(6).trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            if (eventName === "status") {
              callbacks.onStatus?.(data.status, data.message);
            } else if (eventName === "token") {
              callbacks.onToken?.(data.token);
            } else if (eventName === "done") {
              terminalEventReceived = true;
              callbacks.onDone?.(data);
            } else if (eventName === "error") {
              terminalEventReceived = true;
              callbacks.onError?.(data.error);
            }
          } catch {
            // Ignore non-json data
          }
        }
      }
      if (!terminalEventReceived && !signal?.aborted) {
        throw new ApiError("Stream ended before completion. Please retry.");
      }
    } catch (err: unknown) {
      if (signal?.aborted) {
        return;
      }
      const msg = err instanceof Error ? err.message : "Stream connection terminated unexpectedly";
      callbacks.onError?.(msg);
      throw err;
    } finally {
      reader.releaseLock();
    }
  },
};
