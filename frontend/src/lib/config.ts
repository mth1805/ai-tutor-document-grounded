import { resolveApiBaseUrl } from "./public-config.mjs";

export const API_BASE_URL: string = resolveApiBaseUrl(
  process.env.NEXT_PUBLIC_API_BASE_URL,
  process.env.NEXT_PUBLIC_API_URL,
  { production: process.env.NODE_ENV === "production" }
);
