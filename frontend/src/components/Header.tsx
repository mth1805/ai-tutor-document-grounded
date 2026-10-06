"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  GraduationCap,
  Activity,
  Wifi,
  WifiOff,
  User,
  LogOut,
  LogIn,
  Sun,
  Moon,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";
import { useAuth } from "@/lib/auth-context";
import { useTheme } from "@/lib/theme-context";

export function Header() {
  const { user, signOut } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");

  useEffect(() => {
    let isMounted = true;

    async function verifyBackend() {
      try {
        const res = await apiClient.checkHealth();
        if (isMounted) {
          if (res?.status === "ok") {
            setBackendStatus("connected");
          } else {
            setBackendStatus("disconnected");
          }
        }
      } catch {
        if (isMounted) {
          setBackendStatus("disconnected");
        }
      }
    }

    verifyBackend();
    const interval = setInterval(verifyBackend, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="h-14 border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-950/80 backdrop-blur-md px-4 flex items-center justify-between z-10 shrink-0 transition-colors">
      {/* Branding */}
      <Link href="/" className="flex items-center space-x-3 group">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
          <GraduationCap className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              AI Tutor Assistant
            </span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-brand-500/10 text-brand-600 dark:text-brand-400 border border-brand-500/20 font-medium">
              Phase 5
            </span>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:block">
            Document-Grounded Learning Assistant
          </p>
        </div>
      </Link>

      {/* Right-Hand Controls */}
      <div className="flex items-center space-x-2 sm:space-x-3">
        {/* Backend Status */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
            backendStatus === "connected"
              ? "bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800/40"
              : backendStatus === "checking"
              ? "bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800/40"
              : "bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-400 border-rose-200 dark:border-rose-800/40"
          }`}
          title={
            backendStatus === "connected"
              ? `FastAPI Backend is connected (${API_BASE_URL}/health: ok)`
              : backendStatus === "checking"
              ? `Checking backend connection (${API_BASE_URL})...`
              : `Backend unreachable (${API_BASE_URL}). Ensure FastAPI backend is running and reachable.`
          }
        >
          {backendStatus === "connected" ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
              <Wifi className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Backend API Connected</span>
            </>
          ) : backendStatus === "checking" ? (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-500 dark:bg-amber-400 animate-ping" />
              <Activity className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Connecting API...</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-500 dark:bg-rose-400" />
              <WifiOff className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Backend Offline</span>
            </>
          )}
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          suppressHydrationWarning
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          className="w-8 h-8 rounded-full flex items-center justify-center bg-slate-100 hover:bg-slate-200 dark:bg-slate-900 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-amber-400 transition-transform hover:rotate-45" />
          ) : (
            <Moon className="w-4 h-4 text-slate-700 dark:text-slate-300 transition-transform hover:-rotate-12" />
          )}
        </button>

        {/* User Auth Status */}
        {user ? (
          <div className="flex items-center space-x-2 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-full pl-3 pr-1.5 py-1">
            <div className="flex items-center space-x-1.5 text-xs text-slate-700 dark:text-slate-300">
              <User className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400" />
              <span className="max-w-[130px] truncate font-medium">
                {user.email}
              </span>
            </div>
            <button
              onClick={() => signOut()}
              className="p-1 rounded-full text-slate-500 dark:text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center space-x-2">
            <Link
              href="/login"
              className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-100 hover:bg-slate-200 dark:bg-slate-900 dark:hover:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-800 transition-colors"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In</span>
            </Link>
          </div>
        )}
      </div>
    </header>
  );
}
