import { Construction } from "lucide-react";
import type { PlannedTabId } from "../types";
import { ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

const PLANNED_META: Record<
  PlannedTabId,
  { title: string; summary: string; endpoints: string[] }
> = {
  evaluation: {
    title: "Evaluation Arena",
    summary:
      "Side-by-side dataset benchmarking, 20-category failure taxonomy breakdown, and bootstrap quality gates.",
    endpoints: [
      "GET /api/v1/evaluation/datasets",
      "POST /api/v1/evaluation/runs",
      "POST /api/v1/evaluation/compare",
    ],
  },
  reflection: {
    title: "Reflection Lab",
    summary:
      "Single-cycle verification ledger, unsupported-claim diagnostics, and corrective retrieval inspection.",
    endpoints: ["GET /api/v1/rag/reflection/status"],
  },
  experiments: {
    title: "Experiment Lab",
    summary:
      "Isolated candidate evaluation sandbox with strict runtime configuration freeze.",
    endpoints: ["GET /api/v1/rsi/status", "POST /api/v1/rsi/cycles"],
  },
  models: {
    title: "Model Manager",
    summary:
      "Local Ollama inventory monitor, 3 GiB RAM admission checker, and explicit missing-model diagnostics.",
    endpoints: ["GET /ready?inference=true", "GET /api/v1/models/status"],
  },
  monitor: {
    title: "System Monitor",
    summary:
      "Real-time RAM/VRAM admission telemetry, SQLite WAL storage stats, and sparse index rebuild status.",
    endpoints: ["GET /health", "GET /ready", "GET /api/v1/index/status"],
  },
  traces: {
    title: "Execution Traces",
    summary:
      "SQLite WAL trace browser for grounded queries, retrieval modes, and abstention outcomes.",
    endpoints: ["GET /api/v1/rag/traces/{trace_id}", "GET /api/v1/rag/status"],
  },
  settings: {
    title: "Console Settings",
    summary:
      "Workspace display preferences, hardware admission visibility, and CLI promotion guardrails.",
    endpoints: ["Local browser state + owner CLI"],
  },
};

export function PlannedPageView({ pageId }: { pageId: PlannedTabId }) {
  const info = PLANNED_META[pageId];
  return (
    <div className="mx-auto max-w-4xl p-6 animate-stagger">
      <SurfaceCard accent="purple" className="p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <Construction className="h-5 w-5 text-purple-400" />
            <h1 className="font-mono text-base font-semibold text-slate-100">{info.title}</h1>
          </div>
          <ProvenanceBadge kind="PLANNED" label="PHASE 11.4+ · NOT IMPLEMENTED YET" />
        </div>
        <p className="text-sm leading-relaxed text-slate-300">{info.summary}</p>
        <div className="rounded-lg border border-slate-800 bg-[#070A10] p-4">
          <div className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
            Target Backend Contracts
          </div>
          <ul className="mt-2 space-y-1 font-mono text-xs text-cyan-300">
            {info.endpoints.map((ep) => (
              <li key={ep}>• {ep}</li>
            ))}
          </ul>
        </div>
        <p className="font-mono text-[11px] text-slate-500">
          Per Phase 11.3 scope, this view awaits owner approval before implementation.
        </p>
      </SurfaceCard>
    </div>
  );
}
