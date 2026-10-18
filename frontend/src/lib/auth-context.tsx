"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { User, Session } from "@supabase/supabase-js";
import { createClient, isSupabaseConfigured } from "@/lib/supabase/client";
import { friendlyAuthError } from "@/lib/auth-messages";

interface AuthUser {
  id: string;
  email: string;
}

interface AuthContextType {
  user: AuthUser | null;
  session: Session | null;
  token: string | null;
  isLoading: boolean;
  signIn: (email: string, password: string) => Promise<{ error?: string }>;
  signUp: (email: string, password: string) => Promise<{ error?: string }>;
  signOut: () => Promise<void>;
  isConfigured: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const DEMO_USER_ID = "11111111-1111-1111-1111-111111111111";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const configured = isSupabaseConfigured();

  useEffect(() => {
    // 1. If Supabase is configured, use official Supabase Auth listener
    if (configured) {
      const supabase = createClient();

      supabase.auth.getSession().then(({ data: { session } }) => {
        if (session?.user) {
          setSession(session);
          setToken(session.access_token);
          setUser({
            id: session.user.id,
            email: session.user.email || "user@example.com",
          });
        }
        setIsLoading(false);
      });

      const {
        data: { subscription },
      } = supabase.auth.onAuthStateChange((_event, session) => {
        setSession(session);
        setToken(session?.access_token || null);
        if (session?.user) {
          setUser({
            id: session.user.id,
            email: session.user.email || "user@example.com",
          });
        } else {
          setUser(null);
        }
        setIsLoading(false);
      });

      return () => {
        subscription.unsubscribe();
      };
    } else {
      // 2. Local development fallback (check cookie/localStorage for local demo session)
      if (process.env.NODE_ENV !== "development") {
        setIsLoading(false);
        return;
      }
      try {
        const stored = localStorage.getItem("ai_tutor_local_user");
        if (stored) {
          const parsed = JSON.parse(stored);
          setUser(parsed);
          setToken(`test-token:${parsed.id}`);
        }
      } catch {
        // Ignore JSON/storage errors
      }
      setIsLoading(false);
    }
  }, [configured]);

  const signIn = async (email: string, password: string): Promise<{ error?: string }> => {
    setIsLoading(true);
    try {
      if (configured) {
        const supabase = createClient();
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) {
          setIsLoading(false);
          return { error: friendlyAuthError(error) };
        }
        return {};
      } else {
        if (process.env.NODE_ENV !== "development") {
          setIsLoading(false);
          return { error: "Authentication is not configured." };
        }
        // Fallback for local development demo
        const localUser = {
          id: DEMO_USER_ID,
          email: email.trim().toLowerCase(),
        };
        setUser(localUser);
        const testToken = `test-token:${localUser.id}`;
        setToken(testToken);
        localStorage.setItem("ai_tutor_local_user", JSON.stringify(localUser));
        setIsLoading(false);
        return {};
      }
    } catch (err: unknown) {
      setIsLoading(false);
      const msg = err instanceof Error ? err.message : "Authentication failed";
      return { error: msg };
    }
  };

  const signUp = async (email: string, password: string): Promise<{ error?: string }> => {
    setIsLoading(true);
    try {
      if (configured) {
        const supabase = createClient();
        const { error } = await supabase.auth.signUp({ email, password });
        // Email-confirmation signups may return no session/auth event.
        setIsLoading(false);
        if (error) {
          return { error: friendlyAuthError(error) };
        }
        return {};
      } else {
        if (process.env.NODE_ENV !== "development") {
          setIsLoading(false);
          return { error: "Authentication is not configured." };
        }
        // Fallback for local development demo
        const localUser = {
          id: DEMO_USER_ID,
          email: email.trim().toLowerCase(),
        };
        setUser(localUser);
        const testToken = `test-token:${localUser.id}`;
        setToken(testToken);
        localStorage.setItem("ai_tutor_local_user", JSON.stringify(localUser));
        setIsLoading(false);
        return {};
      }
    } catch (err: unknown) {
      setIsLoading(false);
      const msg = err instanceof Error ? err.message : "Sign up failed";
      return { error: msg };
    }
  };

  const signOut = async () => {
    setIsLoading(true);
    if (configured) {
      const supabase = createClient();
      await supabase.auth.signOut();
    }
    localStorage.removeItem("ai_tutor_local_user");
    setUser(null);
    setSession(null);
    setToken(null);
    setIsLoading(false);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        token,
        isLoading,
        signIn,
        signUp,
        signOut,
        isConfigured: configured,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
