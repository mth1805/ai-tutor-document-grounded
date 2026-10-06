/** Shared browser configuration; never read backend credentials here. */
export function resolveApiBaseUrl(
  primary,
  legacy,
  {
    production = false,
    hosted = false,
    allowLocal = false,
    browserHostname = typeof window !== "undefined" ? window.location?.hostname : undefined,
  } = {}
) {
  const isBrowser = typeof browserHostname === "string" && browserHostname.length > 0;
  const isLocalBrowser = isBrowser && ["localhost", "127.0.0.1", "[::1]"].includes(browserHostname);

  // An environment is considered production/hosted if:
  // - hosted is true, OR
  // - running in a browser on a non-local origin, OR
  // - production is true (unless allowLocal is explicitly enabled for local container builds)
  const isEnforcedProduction = hosted || (isBrowser && !isLocalBrowser) || (production && !allowLocal);

  const configured = primary?.trim() || legacy?.trim();

  // Production must not silently fall back to localhost
  if (!configured) {
    if (isEnforcedProduction) {
      throw new Error("NEXT_PUBLIC_API_BASE_URL is required for production and must be a valid HTTPS origin.");
    }
    return "http://localhost:8000";
  }

  const value = configured.replace(/\/+$/, "");
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("NEXT_PUBLIC_API_BASE_URL must be an absolute HTTP(S) origin.");
  }

  const isLoopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);

  // http://localhost:8000 is allowed only in local development
  if (isEnforcedProduction && (isLoopback || url.protocol !== "https:")) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL must be an HTTPS origin in production (http://localhost:8000 is allowed only in local development)."
    );
  }

  if (hosted && isLoopback) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL must be an HTTPS origin (HTTP loopback is allowed only locally).");
  }

  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.pathname !== "/"
  ) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL must be an HTTPS origin (protocol and host only, no path or query).");
  }

  if (
    primary?.trim() &&
    legacy?.trim() &&
    primary.trim().replace(/\/+$/, "") !== legacy.trim().replace(/\/+$/, "")
  ) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL and legacy NEXT_PUBLIC_API_URL conflict; remove the legacy variable."
    );
  }

  return value;
}

export function validateHostedSupabase(urlValue, key) {
  let url;
  try {
    url = new URL(urlValue || "");
  } catch {
    throw new Error("NEXT_PUBLIC_SUPABASE_URL is required and must be an HTTPS origin.");
  }
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash || /placeholder|your-project/i.test(url.hostname)) {
    throw new Error("NEXT_PUBLIC_SUPABASE_URL must be the configured Supabase HTTPS origin.");
  }
  if (!key?.trim() || /placeholder|your-.*key/i.test(key) || key.startsWith("sb_secret_")) {
    throw new Error("NEXT_PUBLIC_SUPABASE_ANON_KEY requires a browser-safe anon or publishable key.");
  }
  if (key.startsWith("sb_publishable_")) return;
  try {
    const payload = JSON.parse(atob(key.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    if (payload.role === "anon") return;
  } catch {
    // Reject malformed and privileged JWTs without printing their contents.
  }
  throw new Error("NEXT_PUBLIC_SUPABASE_ANON_KEY must be an anon JWT or sb_publishable_ key; privileged keys are forbidden.");
}
