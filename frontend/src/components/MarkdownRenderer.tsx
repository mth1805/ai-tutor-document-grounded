"use client";

import React, { useMemo, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { FileText, ExternalLink, Copy, Check } from "lucide-react";
import { Citation } from "@/lib/api";
import {
  preprocessMarkdownWithCitations,
  parseCitationUrl,
  ParsedCitationLink,
  formatCitationPages,
} from "@/lib/citation-utils";

interface MarkdownRendererProps {
  content: string;
  citations?: Citation[];
  onOpenCitation?: (citation: ParsedCitationLink) => void;
  className?: string;
}

// Copy Code Button component for code blocks
function CodeBlockCopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code snippet", err);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-700/60 transition-colors"
      title="Copy code"
    >
      {copied ? (
        <Check className="w-3.5 h-3.5 text-emerald-400" />
      ) : (
        <Copy className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

export function MarkdownRenderer({
  content,
  citations = [],
  onOpenCitation,
  className = "",
}: MarkdownRendererProps) {
  // Preprocess text to turn [Doc: ...] and [Source X] into citation:// links
  const { processedText } = useMemo(() => {
    return preprocessMarkdownWithCitations(content, citations);
  }, [content, citations]);

  // Safe URL transform: allow citation: protocol, sanitize everything else via defaultUrlTransform
  const customUrlTransform = (url: string) => {
    if (url.startsWith("citation://")) {
      return url;
    }
    return defaultUrlTransform(url);
  };

  return (
    <div
      className={`markdown-body text-xs sm:text-[13px] leading-relaxed break-words ${className}`}
    >
      <ReactMarkdown
        urlTransform={customUrlTransform}
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, errorColor: "#ef4444" }]]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-base sm:text-lg font-bold text-slate-900 dark:text-slate-100 mt-4 mb-2 first:mt-0 pb-1 border-b border-slate-200 dark:border-slate-800">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-sm sm:text-base font-bold text-slate-900 dark:text-slate-100 mt-3.5 mb-1.5 first:mt-0">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs sm:text-sm font-semibold text-slate-900 dark:text-slate-100 mt-3 mb-1 first:mt-0">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200 mt-2.5 mb-1">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="text-xs font-semibold text-slate-800 dark:text-slate-200 mt-2 mb-1">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="text-[11px] font-semibold text-slate-700 dark:text-slate-300 mt-2 mb-1 uppercase tracking-wide">
              {children}
            </h6>
          ),

          // Paragraphs
          p: ({ children }) => <p className="mb-2.5 last:mb-0 leading-relaxed">{children}</p>,

          // Strong & Italic
          strong: ({ children }) => (
            <strong className="font-semibold text-slate-900 dark:text-white">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-slate-800 dark:text-slate-200">{children}</em>
          ),

          // Lists
          ul: ({ children }) => (
            <ul className="list-disc pl-5 my-2 space-y-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 my-2 space-y-1">{children}</ol>
          ),
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,

          // Blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-brand-500/80 dark:border-brand-500 pl-3 py-1 my-2.5 bg-slate-50/80 dark:bg-slate-900/60 rounded-r text-slate-700 dark:text-slate-300 italic text-xs">
              {children}
            </blockquote>
          ),

          // Tables
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
              <table className="w-full border-collapse text-left text-xs">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-100 dark:bg-slate-800/80 font-semibold text-slate-900 dark:text-slate-100 border-b border-slate-200 dark:border-slate-800">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-900/40 transition-colors">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-slate-900 dark:text-slate-100">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-slate-700 dark:text-slate-300">
              {children}
            </td>
          ),

          // Horizontal rule
          hr: () => (
            <hr className="my-3.5 border-t border-slate-200 dark:border-slate-800" />
          ),

          // Code blocks & inline code
          code: ({ className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";
            const isBlock = Boolean(language) || String(children).includes("\n");

            if (isBlock) {
              const codeString = String(children).replace(/\n$/, "");
              return (
                <div className="my-2.5 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 shadow-sm text-slate-100">
                  <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/90 border-b border-slate-800 text-[11px] text-slate-400">
                    <span className="font-mono uppercase">{language || "code"}</span>
                    <CodeBlockCopyButton code={codeString} />
                  </div>
                  <div className="p-3 overflow-x-auto text-[11.5px] font-mono leading-relaxed">
                    <code>{children}</code>
                  </div>
                </div>
              );
            }

            return (
              <code
                className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-brand-700 dark:text-brand-300 font-mono text-[11px] border border-slate-200/60 dark:border-slate-700/60"
                {...props}
              >
                {children}
              </code>
            );
          },

          // Links and Citations
          a: ({ href, children }) => {
            // 1. Intercept citation:// links and render tiny inline citation source marker
            if (href?.startsWith("citation://")) {
              const parsed = parseCitationUrl(href);
              if (parsed) {
                const pageBadge = formatCitationPages(parsed.page_start, parsed.page_end);

                const tooltipPage =
                  parsed.page_start === parsed.page_end
                    ? `Page ${parsed.page_start}`
                    : `Pages ${parsed.page_start}–${parsed.page_end}`;

                return (
                  <span className="relative inline-block align-baseline group select-none mx-0.5">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpenCitation?.(parsed);
                      }}
                      aria-label={`${parsed.document_name}, ${tooltipPage}`}
                      className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[10px] font-mono text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-300 bg-slate-100/90 hover:bg-brand-50/90 dark:bg-slate-800/80 dark:hover:bg-brand-950/70 border border-slate-200/80 hover:border-brand-300 dark:border-slate-700/70 dark:hover:border-brand-700/70 transition-all cursor-pointer align-baseline focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-500 leading-none"
                    >
                      <span className="text-[10px] leading-none" aria-hidden="true">
                        📄
                      </span>
                      <span className="font-medium tracking-tight">{pageBadge}</span>
                    </button>

                    {/* Compact Tooltip on Hover / Focus */}
                    <span
                      role="tooltip"
                      className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:flex group-focus-within:flex flex-col items-center z-30"
                    >
                      <span className="bg-slate-900/95 dark:bg-slate-950/95 text-slate-100 border border-slate-700/80 rounded-md px-2.5 py-1.5 shadow-lg text-[10px] whitespace-nowrap leading-tight backdrop-blur-sm text-center">
                        <span className="block font-semibold text-white max-w-[220px] truncate">
                          {parsed.document_name}
                        </span>
                        <span className="block text-slate-400 text-[9px] mt-0.5 font-medium">
                          {tooltipPage}
                        </span>
                      </span>
                      <span className="w-1.5 h-1.5 -mt-0.5 bg-slate-900/95 border-r border-b border-slate-700/80 rotate-45" />
                    </span>
                  </span>
                );
              }
            }

            // 2. Standard safe external link
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-brand-600 dark:text-brand-400 underline underline-offset-2 hover:text-brand-700 dark:hover:text-brand-300 inline-flex items-center space-x-0.5"
              >
                <span>{children}</span>
                <ExternalLink className="w-2.5 h-2.5 inline opacity-70 ml-0.5" />
              </a>
            );
          },
        }}
      >
        {processedText}
      </ReactMarkdown>
    </div>
  );
}
