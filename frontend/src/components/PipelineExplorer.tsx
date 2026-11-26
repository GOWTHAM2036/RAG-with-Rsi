import { useRef, useState } from "react";
import { Maximize2, Minimize2, Play, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import type { InspectorPayload, PipelineStageNode, StageStatus } from "../types";
import { ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

const BASE_STAGES: PipelineStageNode[] = [
  {
    id: "query",
    label: "Query",
    subtitle: "HTTP / NDJSON Input",
    status: "completed",
    latencyMs: 1.2,
    provenance: "static",
    x: 40,
    y: 130,
    details: "Validates question length, retrieval mode, and conversation budget.",
    metrics: { endpoint: "/api/v1/rag/query/stream" },
  },
  {
    id: "analysis",
    label: "Analysis",
    subtitle: "CPU Query Classifier",
    status: "completed",
    latencyMs: 2.4,
    provenance: "static",
    x: 195,
    y: 130,
    details: "Deterministic complexity classification & optional bounded rewrite.",
    metrics: { mode: "deterministic" },
  },
  {
    id: "embedding",
    label: "Embedding",
    subtitle: "SQLite Cache / Ollama",
    status: "completed",
    latencyMs: 14.5,
    provenance: "static",
    x: 350,
    y: 130,
    details: "Embeds query vector or falls back to sparse-only when model is absent.",
    metrics: { model: "qwen3-embedding:0.6b" },
  },
  {
    id: "dense_bm25",
    label: "Dense / BM25",
    subtitle: "Float32 + Okapi BM25",
    status: "active",
    latencyMs: 9.8,
    provenance: "static",
    x: 505,
    y: 65,
    details: "Exact float32 cosine similarity paired with custom SQLite inverted BM25.",
    metrics: { vector_dtype: "float32" },
  },
  {
    id: "rrf",
    label: "RRF Fusion",
    subtitle: "Weighted Reciprocal Rank",
    status: "active",
    latencyMs: 1.6,
    provenance: "static",
    x: 505,
    y: 195,
    details: "Fuses dense and sparse ranked lists deterministically (rrf_k=60).",
    metrics: { rrf_k: 60 },
  },
  {
    id: "context",
    label: "Context",
    subtitle: "4096-Token Budget",
    status: "completed",
    latencyMs: 3.1,
    provenance: "static",
    x: 660,
    y: 130,
    details: "Deduplicates chunks and enforces strict 4096-token budget.",
    metrics: { max_tokens: 4096 },
  },
  {
    id: "generation",
    label: "Generation",
    subtitle: "qwen3:4b (3 GiB floor)",
    status: "idle",
    latencyMs: null,
    provenance: "static",
    x: 815,
    y: 130,
    details: "Grounded JSON answer generation guarded by 3 GiB RAM admission.",
    metrics: { model: "qwen3:4b", think: "false" },
  },
  {
    id: "reflection",
    label: "Reflection",
    subtitle: "Bounded 1-Cycle Repair",
    status: "skipped",
    latencyMs: null,
    provenance: "static",
    x: 970,
    y: 65,
    details: "Optional single-cycle claim critique & targeted re-retrieval.",
    metrics: { default: "disabled" },
  },
  {
    id: "verification",
    label: "Verification",
    subtitle: "Citation Admission",
    status: "completed",
    latencyMs: 1.1,
    provenance: "static",
    x: 970,
    y: 195,
    details: "Verifies cited chunk IDs and enforces abstention on weak evidence.",
    metrics: { policy: "strict_json" },
  },
  {
    id: "answer",
    label: "Answer",
    subtitle: "Trace WAL + Citations",
    status: "completed",
    latencyMs: 0.8,
    provenance: "static",
    x: 1125,
    y: 130,
    details: "Persists execution trace to SQLite WAL and streams final payload.",
    metrics: { storage: "sqlite_wal" },
  },
];

const EDGES: Array<[string, string]> = [
  ["query", "analysis"],
  ["analysis", "embedding"],
  ["embedding", "dense_bm25"],
  ["embedding", "rrf"],
  ["dense_bm25", "rrf"],
  ["rrf", "context"],
  ["context", "generation"],
  ["generation", "reflection"],
  ["generation", "verification"],
  ["reflection", "verification"],
  ["verification", "answer"],
];

const STATUS_BADGE: Record<StageStatus, string> = {
  completed: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  active: "border-cyan-500/50 bg-cyan-500/15 text-cyan-200",
  skipped: "border-slate-700 bg-slate-900 text-slate-400",
  error: "border-rose-500/50 bg-rose-500/15 text-rose-300",
  idle: "border-amber-500/40 bg-amber-500/10 text-amber-300",
};

export function PipelineExplorer({
  lastEvents,
  backendOnline,
  onInspect,
}: {
  lastEvents: string[];
  backendOnline: boolean;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedId, setSelectedId] = useState<string>("dense_bm25");
  const [viewMode, setViewMode] = useState<"architecture" | "recorded">(
    lastEvents.length > 0 ? "recorded" : "architecture"
  );
  const [zoom, setZoom] = useState(0.95);
  const [timelineStep, setTimelineStep] = useState<number>(BASE_STAGES.length - 1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const selectedStage = BASE_STAGES.find((s) => s.id === selectedId) ?? BASE_STAGES[0];
  const byId = new Map(BASE_STAGES.map((s) => [s.id, s]));

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      setIsFullscreen(false);
    } else {
      await containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    }
  };

  return (
    <div ref={containerRef} className="mx-auto max-w-[1440px] space-y-4 p-4 lg:p-5 animate-stagger">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-mono text-sm font-semibold tracking-wider text-slate-100">
              PIPELINE EXPLORER · EXECUTION DAG
            </h1>
            <ProvenanceBadge
              kind={viewMode === "recorded" && backendOnline ? "LIVE" : "STATIC"}
              label={
                viewMode === "recorded"
                  ? backendOnline
                    ? "RECORDED EXECUTION"
                    : "NO LIVE RECORDING"
                  : "STATIC ARCHITECTURE"
              }
            />
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Interactive 10-stage grounded RAG execution topology with latency & status inspection.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setViewMode("architecture")}
            className={`rounded-lg border px-2.5 py-1 font-mono text-[11px] ${
              viewMode === "architecture"
                ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
                : "border-slate-800 text-slate-400"
            }`}
          >
            Static Architecture
          </button>
          <button
            type="button"
            onClick={() => setViewMode("recorded")}
            className={`rounded-lg border px-2.5 py-1 font-mono text-[11px] ${
              viewMode === "recorded"
                ? "border-purple-500/40 bg-purple-500/15 text-purple-200"
                : "border-slate-800 text-slate-400"
            }`}
          >
            Recorded Execution
          </button>
          <button
            type="button"
            aria-label="Zoom in pipeline"
            onClick={() => setZoom((z) => Math.min(1.3, Number((z + 0.1).toFixed(2))))}
            className="rounded border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Zoom out pipeline"
            onClick={() => setZoom((z) => Math.max(0.65, Number((z - 0.1).toFixed(2))))}
            className="rounded border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Reset pipeline zoom"
            onClick={() => setZoom(0.95)}
            className="rounded border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Fullscreen pipeline"
            onClick={toggleFullscreen}
            className="rounded border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800"
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Interactive DAG Canvas */}
      <SurfaceCard accent="cyan" className="overflow-x-auto p-4">
        <div style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }} className="min-w-[1260px]">
          <div className="relative h-[280px] w-[1260px]">
            <svg className="pointer-events-none absolute inset-0 h-full w-full">
              {EDGES.map(([fromId, toId]) => {
                const a = byId.get(fromId);
                const b = byId.get(toId);
                if (!a || !b) return null;
                const active = fromId === selectedId || toId === selectedId;
                return (
                  <path
                    key={`${fromId}->${toId}`}
                    d={`M ${a.x + 120} ${a.y + 32} C ${a.x + 150} ${a.y + 32}, ${b.x - 30} ${b.y + 32}, ${b.x} ${b.y + 32}`}
                    fill="none"
                    stroke={active ? "#22D3EE" : "#1E293B"}
                    strokeWidth={active ? 2 : 1.4}
                    className={active ? "animate-flow-dash" : ""}
                  />
                );
              })}
            </svg>

            {BASE_STAGES.map((stage, idx) => {
              const isSelected = stage.id === selectedId;
              const isStepReached = idx <= timelineStep;
              return (
                <button
                  key={stage.id}
                  type="button"
                  onClick={() => {
                    setSelectedId(stage.id);
                    setTimelineStep(idx);
                    onInspect({
                      title: stage.label,
                      kind: "pipeline_stage",
                      provenance: viewMode === "recorded" && backendOnline ? "live" : "static",
                      summary: stage.details,
                      fields: {
                        status: stage.status,
                        latency_ms: stage.latencyMs ?? "unavailable",
                        ...stage.metrics,
                      },
                    });
                  }}
                  style={{ left: stage.x, top: stage.y }}
                  className={`absolute w-[124px] rounded-xl border p-2.5 text-left transition ${
                    isSelected
                      ? "border-cyan-400 bg-[#0D1A26] shadow-[0_0_20px_rgba(34,211,238,0.2)]"
                      : isStepReached
                        ? "border-slate-800 bg-[#090E17] hover:border-slate-600"
                        : "border-slate-900 bg-[#070A10] opacity-55"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono text-[11px] font-semibold text-slate-100">
                      {stage.label}
                    </span>
                    <span
                      className={`rounded border px-1 py-0.2 font-mono text-[8px] uppercase ${STATUS_BADGE[stage.status]}`}
                    >
                      {stage.status}
                    </span>
                  </div>
                  <div className="mt-1 truncate text-[10px] text-slate-400">{stage.subtitle}</div>
                  <div className="mt-1.5 font-mono text-[10px] text-cyan-300">
                    {stage.latencyMs != null ? `${stage.latencyMs.toFixed(1)} ms` : "n/a"}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </SurfaceCard>

      {/* Bottom Timeline & Stage Inspector */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <SurfaceCard className="lg:col-span-7 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-200">
              EXECUTION TIMELINE SCRUBBER
            </span>
            <span className="font-mono text-[11px] text-cyan-300">
              Stage {timelineStep + 1} / {BASE_STAGES.length} ({BASE_STAGES[timelineStep].label})
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={BASE_STAGES.length - 1}
            value={timelineStep}
            aria-label="Pipeline execution timeline step"
            onChange={(e) => {
              const idx = Number(e.target.value);
              setTimelineStep(idx);
              setSelectedId(BASE_STAGES[idx].id);
            }}
            className="w-full accent-cyan-400"
          />
          {lastEvents.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {lastEvents.map((e, i) => (
                <span
                  key={i}
                  className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 font-mono text-[11px] text-cyan-200"
                >
                  {e}
                </span>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
              <Play className="h-3.5 w-3.5 text-slate-500" />
              Run a query in Command Center to attach live execution events to this DAG.
            </div>
          )}
        </SurfaceCard>

        <SurfaceCard accent="purple" className="lg:col-span-5 p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-100">
              STAGE INSPECTOR · {selectedStage.label.toUpperCase()}
            </span>
            <span
              className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase ${STATUS_BADGE[selectedStage.status]}`}
            >
              {selectedStage.status}
            </span>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">{selectedStage.details}</p>
          <div className="rounded-lg border border-slate-800 bg-[#05070B] p-2.5 font-mono text-[11px] space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">reference_latency</span>
              <span className="text-cyan-300">
                {selectedStage.latencyMs != null ? `${selectedStage.latencyMs} ms` : "unavailable"}
              </span>
            </div>
            {Object.entries(selectedStage.metrics).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-slate-500">{k}</span>
                <span className="text-slate-200">{String(v)}</span>
              </div>
            ))}
          </div>
        </SurfaceCard>
      </div>
    </div>
  );
}
