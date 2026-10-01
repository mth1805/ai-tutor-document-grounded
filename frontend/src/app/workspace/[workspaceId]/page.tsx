import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { WorkspaceSplitView } from "@/components/WorkspaceSplitView";

export default function WorkspacePage() {
  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      <Header />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />
        <WorkspaceSplitView />
      </div>
    </div>
  );
}
