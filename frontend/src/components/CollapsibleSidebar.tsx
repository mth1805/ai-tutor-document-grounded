"use client";

import React, { useEffect, useState } from "react";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";

const STORAGE_KEY = "ai-tutor-sidebar-collapsed";

export function CollapsibleSidebar({ children }: { children: React.ReactNode }) {
  // Match the server render, then restore the non-sensitive preference on mount.
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    try {
      setCollapsed(window.localStorage.getItem(STORAGE_KEY) === "true");
    } catch {
      // Navigation still works when browser storage is unavailable.
    }
  }, []);

  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, String(next));
    } catch {
      // Persistence is optional; the current session can still toggle.
    }
  };

  return (
    <div
      data-sidebar-collapsed={collapsed}
      className={`relative shrink-0 h-full transition-[width] duration-200 ease-out motion-reduce:transition-none ${collapsed ? "w-0" : "w-64 max-w-[80vw]"}`}
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={!collapsed}
        aria-controls="workspace-sidebar"
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        className={`absolute z-30 p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-brand-600 dark:text-brand-400 hover:bg-brand-50 dark:hover:bg-slate-800 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 ${collapsed ? "left-1 top-1/2 -translate-y-1/2" : "right-2 top-2"}`}
      >
        {collapsed ? <PanelLeftOpen className="w-4 h-4" aria-hidden="true" /> : <PanelLeftClose className="w-4 h-4" aria-hidden="true" />}
      </button>
      <div
        id="workspace-sidebar"
        hidden={collapsed}
        className="h-full w-full overflow-hidden"
      >
        {children}
      </div>
    </div>
  );
}
