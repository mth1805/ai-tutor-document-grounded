"use client";

import React, { useState } from "react";
import Link from "next/link";
import { GraduationCap, ArrowRight, Loader2, AlertCircle } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { SIGNUP_CONFIRMATION_MESSAGE } from "@/lib/auth-messages";

export default function SignUpPage() {
  const { signUp, isLoading: authLoading } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    if (!email || !password) {
      setError("Please provide both email and password.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);
    const res = await signUp(email, password);
    setIsSubmitting(false);

    if (res.error) {
      setError(res.error);
    } else {
      setSuccess(true);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-4 transition-colors">
      <div className="w-full max-w-md bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-xl shadow-xl dark:shadow-2xl dark:shadow-black/50">
        {/* Branding Header */}
        <div className="flex flex-col items-center text-center mb-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-brand-500/20 mb-3">
            <GraduationCap className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100 tracking-tight">
            Create Your Account
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Start learning with document-grounded AI assistance
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-5 p-3 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800/60 flex items-start space-x-2.5 text-xs text-rose-600 dark:text-rose-300">
            <AlertCircle className="w-4 h-4 text-rose-500 dark:text-rose-400 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div role="status" className="mb-5 p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/50 text-sm text-emerald-800 dark:text-emerald-200">
            {SIGNUP_CONFIRMATION_MESSAGE} <Link href="/login" className="underline">Sign in</Link>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Email Address
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="student@example.com"
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 focus:border-brand-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-600"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Password
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 6 characters"
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 focus:border-brand-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-600"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Confirm Password
            </label>
            <input
              type="password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Repeat password"
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500 focus:border-brand-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-600"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting || authLoading}
            className="w-full mt-2 flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white font-medium text-sm shadow-md shadow-brand-500/10 transition-all disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Creating account...</span>
              </>
            ) : (
              <>
                <span>Sign Up</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {/* Footer Link */}
        <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-800/80 text-center text-xs text-slate-500 dark:text-slate-400">
          Already have an account?{" "}
          <Link
            href="/login"
            className="text-brand-600 dark:text-brand-400 hover:text-brand-500 dark:hover:text-brand-300 font-medium underline-offset-4 hover:underline"
          >
            Sign in
          </Link>
        </div>
      </div>
    </div>
  );
}
