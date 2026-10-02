"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { DocumentItem, apiClient } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

interface DocumentContextType {
  selectedDocument: DocumentItem | null;
  setSelectedDocument: (doc: DocumentItem | null) => void;
  isViewerCollapsed: boolean;
  setIsViewerCollapsed: (collapsed: boolean | ((prev: boolean) => boolean)) => void;
  splitRatio: number; // percentage width of left pane (20 to 80)
  setSplitRatio: (ratio: number) => void;
}

const DocumentContext = createContext<DocumentContextType | undefined>(undefined);

export function DocumentProvider({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const [selectedDocument, setSelectedDocumentState] = useState<DocumentItem | null>(null);
  const [isViewerCollapsed, setIsViewerCollapsed] = useState<boolean>(false);
  const [splitRatio, setSplitRatioState] = useState<number>(45);

  // Restore split ratio preference from localStorage if available
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedRatio = localStorage.getItem("ai_tutor_split_ratio");
      if (savedRatio) {
        const parsed = parseFloat(savedRatio);
        if (!isNaN(parsed) && parsed >= 20 && parsed <= 80) {
          setSplitRatioState(parsed);
        }
      }
    }
  }, []);

  const setSplitRatio = (ratio: number) => {
    const clamped = Math.min(80, Math.max(20, ratio));
    setSplitRatioState(clamped);
    if (typeof window !== "undefined") {
      localStorage.setItem("ai_tutor_split_ratio", clamped.toFixed(1));
    }
  };

  // Sync selected document state with URL search param ?doc=
  const setSelectedDocument = useCallback(
    (doc: DocumentItem | null) => {
      setSelectedDocumentState(doc);
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search);
        if (doc) {
          params.set("doc", doc.id);
        } else {
          params.delete("doc");
        }
        const newSearch = params.toString();
        const newUrl = `${pathname}${newSearch ? `?${newSearch}` : ""}`;
        router.replace(newUrl, { scroll: false });
      }
    },
    [pathname, router]
  );

  // Load document from ?doc= on initial mount or URL change
  useEffect(() => {
    const docId = searchParams.get("doc");
    if (!docId || !token) {
      if (!docId) {
        setSelectedDocumentState(null);
      }
      return;
    }

    // If current selected doc already matches, do nothing
    if (selectedDocument?.id === docId) return;

    // Fetch document metadata by ID
    apiClient
      .getDocument(docId, token)
      .then((doc) => {
        setSelectedDocumentState(doc);
      })
      .catch(() => {
        // If not found or unauthorized, clear search param
        setSelectedDocument(null);
      });
  }, [searchParams, token, selectedDocument?.id, setSelectedDocument]);

  return (
    <DocumentContext.Provider
      value={{
        selectedDocument,
        setSelectedDocument,
        isViewerCollapsed,
        setIsViewerCollapsed,
        splitRatio,
        setSplitRatio,
      }}
    >
      {children}
    </DocumentContext.Provider>
  );
}

export function useDocument() {
  const context = useContext(DocumentContext);
  if (!context) {
    return {
      selectedDocument: null,
      setSelectedDocument: () => {},
      isViewerCollapsed: false,
      setIsViewerCollapsed: () => {},
      splitRatio: 45,
      setSplitRatio: () => {},
    };
  }
  return context;
}

