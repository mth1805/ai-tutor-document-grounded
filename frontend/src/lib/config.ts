import { resolveApiBaseUrl } from "./public-config.mjs";

const isBrowser = typeof window !== "undefined";
const isLocalBrowser =
  isBrowser && ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
const isHosted =
  process.env.VERCEL === "1" ||
  Boolean(process.env.NEXT_PUBLIC_VERCEL_ENV) ||
  (isBrowser && !isLocalBrowser);
const isProduction = process.env.NODE_ENV === "production" || isHosted;

export const API_BASE_URL: string = resolveApiBaseUrl(
  process.env.NEXT_PUBLIC_API_BASE_URL,
  process.env.NEXT_PUBLIC_API_URL,
  {
    production: isProduction,
    hosted: isHosted,
  }
);
