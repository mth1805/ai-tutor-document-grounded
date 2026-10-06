/** @type {import('next').NextConfig} */
import { resolveApiBaseUrl, validateHostedSupabase } from "./src/lib/public-config.mjs";

// Vercel builds must never silently ship localhost or placeholder auth settings.
if (process.env.VERCEL === "1") {
  resolveApiBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL, process.env.NEXT_PUBLIC_API_URL, { hosted: true });
  validateHostedSupabase(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY);
}

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  output: "standalone",
};

export default nextConfig;
