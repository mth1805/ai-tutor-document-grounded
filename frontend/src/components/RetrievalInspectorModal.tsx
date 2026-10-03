"use client";

import React, { useState } from "react";
import {
  Search,
  X,
  Loader2,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  FileText,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { apiClient, RetrievalSearchResponse, RetrievedChunkItem } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

interface RetrievalInspectorModalProps {
  workspaceId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function RetrievalInspectorModal({
  workspaceId,
  isOpen,
  onClose,
}: RetrievalInspectorModalProps) {
  const { token } = useAuth();

  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RetrievalSearchResponse | null>(null);

  // Advanced hyperparameters
  const [showConfig, setShowConfig] = useState(false);
  const [denseTopK, setDenseTopK] = useState(25);
  const [lexicalTopK, setLexicalTopK] = useState(25);
  const [rrfK, setRrfK] = useState(60);
  const [candidatePoolSize, setCandidatePoolSize] = useState(30);
  const [rerankTopK, setRerankTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.35);

  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});

  if (!isOpen) return null;

  const toggleExpand = (chunkId: string) => {
    setExpandedChunks((prev) => ({ ...prev, [chunkId]: !prev[chunkId] }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery || !token) return;

    setIsSearching(true);
    setError(null);

    try {
      const data = await apiClient.searchRetrieval(
        workspaceId,
        {
          query: cleanQuery,
          dense_top_k: denseTopK,
          lexical_top_k: lexicalTopK,
          rrf_k: rrfK,
          candidate_pool_size: candidatePoolSize,
          rerank_top_k: rerankTopK,
          relevance_threshold: threshold,
        },
        token
      );
      setResults(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Retrieval search failed";
      setError(msg);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl w-full max-w-4xl shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between shrink-0 bg-slate-50/50 dark:bg-slate-950/40">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/20 text-brand-600 dark:text-brand-400 flex items-center justify-center">
              <Search className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center space-x-2">
                <span>Phase 7 Retrieval Inspector</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-brand-100 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-800">
                  Dense + BM25 + RRF + Cross-Encoder
                </span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Inspect retrieved chunks, rank provenance, and relevance gating in real time.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search Bar Form */}
        <div className="p-6 border-b border-slate-200 dark:border-slate-800 shrink-0 space-y-3">
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter query to inspect document retrieval (e.g., 'What is Newton second law?')..."
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 text-xs focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all"
              />
            </div>
            <button
              type="button"
              onClick={() => setShowConfig(!showConfig)}
              className={`px-3 py-2.5 rounded-xl border text-xs font-medium flex items-center space-x-1.5 transition-colors ${
                showConfig
                  ? "border-brand-500 text-brand-600 bg-brand-50 dark:bg-brand-950/30"
                  : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
              title="Configure Retrieval Parameters"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Config</span>
            </button>
            <button
              type="submit"
              disabled={isSearching || !query.trim()}
              className="px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-medium text-xs shadow-md shadow-brand-600/20 transition-all active:scale-95 disabled:opacity-50 flex items-center space-x-2"
            >
              {isSearching ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search className="w-3.5 h-3.5" />
                  <span>Retrieve Chunks</span>
                </>
              )}
            </button>
          </form>

          {/* Hyperparameter Settings Panel */}
          {showConfig && (
            <div className="grid grid-cols-3 gap-3 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 text-xs">
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  Dense Top-K: {denseTopK}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={denseTopK}
                  onChange={(e) => setDenseTopK(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  Lexical Top-K: {lexicalTopK}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={lexicalTopK}
                  onChange={(e) => setLexicalTopK(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  RRF Constant (k): {rrfK}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={rrfK}
                  onChange={(e) => setRrfK(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  Candidate Pool: {candidatePoolSize}
                </label>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={candidatePoolSize}
                  onChange={(e) => setCandidatePoolSize(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  Rerank Top-K: {rerankTopK}
                </label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  value={rerankTopK}
                  onChange={(e) => setRerankTopK(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                  Relevance Threshold: {threshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full accent-brand-600"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="p-2.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-xs text-rose-700 dark:text-rose-300">
              {error}
            </div>
          )}
        </div>

        {/* Results Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {results && (
            <>
              {/* Timing Instrumentation & Gate Status Summary */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs">
                <div className="flex items-center space-x-2">
                  {results.has_sufficient_evidence ? (
                    <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 font-semibold border border-emerald-300 dark:border-emerald-800">
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                      Sufficient Evidence (Passes Gate)
                    </span>
                  ) : (
                    <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-full bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 font-semibold border border-amber-300 dark:border-amber-800">
                      <AlertTriangle className="w-3.5 h-3.5 mr-1" />
                      Insufficient Evidence (&lt; {results.relevance_threshold.toFixed(2)})
                    </span>
                  )}
                  <span className="text-slate-500 font-mono text-[11px]">
                    {results.total_results} chunks ranked
                  </span>
                </div>

                {/* Timing Breakdown Badges */}
                <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
                  <span className="bg-slate-200 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                    Embed: {results.timings.query_embedding_ms}ms
                  </span>
                  <span className="bg-slate-200 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                    Dense: {results.timings.dense_retrieval_ms}ms
                  </span>
                  <span className="bg-slate-200 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                    BM25: {results.timings.lexical_retrieval_ms}ms
                  </span>
                  <span className="bg-slate-200 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                    RRF: {results.timings.rrf_ms}ms
                  </span>
                  <span className="bg-slate-200 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-700 dark:text-slate-300">
                    Rerank: {results.timings.rerank_ms}ms
                  </span>
                  <span className="bg-brand-100 dark:bg-brand-950 text-brand-700 dark:text-brand-300 font-bold px-2 py-0.5 rounded border border-brand-300 dark:border-brand-800">
                    Total: {results.timings.total_retrieval_ms}ms
                  </span>
                </div>
              </div>

              {/* Retrieved Chunks List */}
              <div className="space-y-3">
                {results.results.map((chunk: RetrievedChunkItem) => {
                  const isExpanded = !!expandedChunks[chunk.chunk_id];
                  return (
                    <div
                      key={chunk.chunk_id}
                      className={`p-4 rounded-xl border transition-all ${
                        chunk.passed_relevance_gate
                          ? "border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/20 dark:bg-emerald-950/10"
                          : "border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 opacity-75"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="w-6 h-6 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-bold flex items-center justify-center">
                            #{chunk.final_rank}
                          </span>
                          <span
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider ${
                              chunk.passed_relevance_gate
                                ? "bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800"
                                : "bg-slate-100 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-700"
                            }`}
                          >
                            {chunk.passed_relevance_gate ? "Gate: Passed" : "Gate: Rejected"}
                          </span>
                          <span className="text-xs font-medium text-slate-600 dark:text-slate-400 flex items-center space-x-1">
                            <FileText className="w-3.5 h-3.5" />
                            <span>
                              Pages {chunk.page_number_start}
                              {chunk.page_number_end !== chunk.page_number_start
                                ? ` - ${chunk.page_number_end}`
                                : ""}
                            </span>
                          </span>
                        </div>

                        {/* Channel & Scores Pills */}
                        <div className="flex items-center space-x-1.5 font-mono text-[11px]">
                          {chunk.retrieval_sources.map((src) => (
                            <span
                              key={src}
                              className="px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold bg-violet-100 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-800"
                            >
                              {src}
                            </span>
                          ))}
                          <span
                            className="px-2 py-0.5 rounded bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-800 font-bold"
                            title="Cross-Encoder Rerank Score"
                          >
                            Rerank: {chunk.rerank_score?.toFixed(4) ?? "N/A"}
                          </span>
                          <span
                            className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300"
                            title="Reciprocal Rank Fusion Score"
                          >
                            RRF: {chunk.rrf_score?.toFixed(4) ?? "N/A"}
                          </span>
                        </div>
                      </div>

                      {/* Chunk Content Snippet */}
                      <div className="text-xs text-slate-700 dark:text-slate-300 bg-slate-50 dark:bg-slate-950 p-3 rounded-lg font-sans leading-relaxed border border-slate-100 dark:border-slate-800/80">
                        {isExpanded
                          ? chunk.content
                          : chunk.content.length > 250
                          ? `${chunk.content.slice(0, 250)}...`
                          : chunk.content}
                      </div>

                      {chunk.content.length > 250 && (
                        <button
                          onClick={() => toggleExpand(chunk.chunk_id)}
                          className="mt-1 text-[11px] font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center space-x-1"
                        >
                          {isExpanded ? (
                            <>
                              <ChevronUp className="w-3 h-3" />
                              <span>Show less</span>
                            </>
                          ) : (
                            <>
                              <ChevronDown className="w-3 h-3" />
                              <span>Show full chunk text</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {!results && !isSearching && (
            <div className="h-64 flex flex-col items-center justify-center text-center text-slate-400">
              <Search className="w-8 h-8 mb-2 opacity-50" />
              <p className="text-xs">Type a query above to execute and inspect the retrieval pipeline.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
