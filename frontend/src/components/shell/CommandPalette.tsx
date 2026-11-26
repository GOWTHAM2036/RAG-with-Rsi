import { useEffect, useMemo, useState } from "react";
import { Command, Search, X } from "lucide-react";
import type { NavTabId } from "../../types";

export interface PaletteCommand {
  id: string;
  label: string;
  group: string;
  hint?: string;
  run: () => void;
}

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  onToggleReducedMotion,
  onToggleLowPerf,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (tab: NavTabId) => void;
  onToggleReducedMotion: () => void;
  onToggleLowPerf: () => void;
}) {
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const commands = useMemo<PaletteCommand[]>(
    () => [
      { id: "nav-command", label: "Go to Intelligence Command Center", group: "Signature Views", hint: "Overview", run: () => onNavigate("command") },
      { id: "nav-explorer", label: "Go to Knowledge Explorer", group: "Signature Views", hint: "Provenance Graph", run: () => onNavigate("explorer") },
      { id: "nav-pipeline", label: "Go to Pipeline Explorer", group: "Signature Views", hint: "Execution DAG", run: () => onNavigate("pipeline") },
      { id: "nav-rsi", label: "Go to RSI Evolution Tree", group: "Signature Views", hint: "Lineage", run: () => onNavigate("rsi") },
      { id: "nav-playground", label: "Go to RAG Playground", group: "AI Workspaces", hint: "Chat & Citations", run: () => onNavigate("playground") },
      { id: "nav-documents", label: "Go to Document Management", group: "AI Workspaces", hint: "Ingest & Chunks", run: () => onNavigate("documents") },
      { id: "nav-retrieval", label: "Go to Retrieval Lab", group: "AI Workspaces", hint: "Rankings & Math", run: () => onNavigate("retrieval") },
      { id: "nav-evaluation", label: "Preview Evaluation Arena (Planned)", group: "Roadmap", run: () => onNavigate("evaluation") },
      { id: "nav-models", label: "Preview Model Manager (Planned)", group: "Roadmap", run: () => onNavigate("models") },
      { id: "act-motion", label: "Toggle Reduced Motion", group: "Appearance", run: onToggleReducedMotion },
      { id: "act-perf", label: "Toggle Low-Performance Mode", group: "Appearance", run: onToggleLowPerf },
    ],
    [onNavigate, onToggleLowPerf, onToggleReducedMotion]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q)
    );
  }, [commands, query]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 px-4 pt-20 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Global Command Palette"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-xl border border-cyan-500/30 bg-[#0B1018] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-slate-800 px-3.5 py-2.5">
          <Search className="h-4 w-4 text-cyan-400" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Jump to view, toggle rendering mode..."
            className="w-full bg-transparent font-mono text-xs text-slate-100 outline-none placeholder:text-slate-500"
          />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close command palette"
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-80 overflow-y-auto p-2">
          {filtered.map((cmd) => (
            <button
              key={cmd.id}
              type="button"
              onClick={() => {
                cmd.run();
                onClose();
              }}
              className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs text-slate-200 transition hover:bg-cyan-500/10 hover:text-cyan-200"
            >
              <div className="flex items-center gap-2">
                <Command className="h-3.5 w-3.5 text-slate-500" />
                <span>{cmd.label}</span>
              </div>
              <span className="font-mono text-[10px] text-slate-500">{cmd.hint ?? cmd.group}</span>
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="px-3 py-6 text-center font-mono text-xs text-slate-500">
              No matching commands.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
