"use client";

import React from "react";
import { DocumentProvider } from "@/lib/document-context";

export default function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <DocumentProvider>{children}</DocumentProvider>;
}
