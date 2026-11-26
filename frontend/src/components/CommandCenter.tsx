import { useState } from "react";
import { Activity, AlertTriangle, Cpu, Play, ShieldAlert, Zap } from "lucide-react";
import { api } from "../api";
import type {
  AppearanceSettings,
  InspectorPayload,
  RAGResponse,
  RAGStreamEvent,
} from "../types";
import { LivingRagCore, type CoreStateMode } from "./LivingRagCore";
import { MetricReadout, ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

export function CommandCenter({
  backendOnline,
  ragStatus,
  reflectionStatus,
  rsiStatus,
  readyStatus,
  appearance,
  lastEvents,
  onQueryCompleted,
  onInspect,
}: {
  backendOnline: boolean;
  ragStatus: Record<string, unknown> | null;
  reflectionStatus: Record<string, unknown> | null;
  rsiStatus: Record<string, unknown> | null;
  readyStatus: Record<string, unknown> | null;
  appearance: AppearanceSettings;
  lastEvents: string[];
  onQueryCompleted: (question: string, response: RAGResponse) => void;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const [question, setQuestion] = useState("What database journal mode does EvoRag use?");
  const [retrievalMode, setRetrievalMode] = useState<"hybrid" | "sparse" | "dense">("hybrid");
  const [strategy, setStrategy] = useState<"baseline" | "adaptive">("baseline");
  const [reflect, setReflect] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [events, setEvents] = useState<RAGStreamEvent[]>([]);
  const [result, setResult] = useState<RAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pulseKey, setPulseKey] = useState(0);

  const coreState: CoreStateMode = streaming
    ? reflect
      ? "reflection"
      : "retrieval"
    : rsiStatus?.enabled
      ? "evolution"
      : "idle";

  const runQuery = async () => {
    setStreaming(true);
    setError(null);
    setEvents([]);
    setResult(null);
    setPulseKey((k) => k + 1);
    try {
      await api.streamRagQuery(
        { question, retrieval_mode: retrievalMode, strategy, reflection_enabled: reflect },
        (ev) => {
          setEvents((prev) => [...prev, ev]);
          if (ev.response) {
            setResult(ev.response);
            onQueryCompleted(question, ev.response);
            setPulseKey((k) => k + 1);
          }
          if (ev.error) setError(ev.error);
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query execution failed");
    } finally {
      setStreaming(false);
    }
  };

  const activeConfig = String(rsiStatus?.active_version_id ?? "v0_baseline");
  const cyclesCompleted = String(rsiStatus?.cycles_completed ?? 0);
  const modelsLoaded = Boolean(readyStatus && readyStatus.status === "ready");
  const blockerMessage = !backendOnline
    ? "FastAPI backend unreachable on 127.0.0.1:8000 (showing explicit offline demo state)."
    : modelsLoaded
      ? "No active inference blockers reported."
      : "3 GiB RAM admission floor / missing local Ollama model tag may block live generation.";

  return (
    <div className="mx-auto max-w-[1440px] space-y-4 p-4 lg:p-5 animate-stagger">
      {/* Asymmetrical Bento Top Row */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Hero Core + Query Console (8 cols) */}
        <SurfaceCard accent="cyan" className="xl:col-span-8 p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-mono text-sm font-semibold tracking-wider text-slate-100">
                  INTELLIGENCE COMMAND CENTER
                </h1>
                <ProvenanceBadge kind={backendOnline ? "LIVE" : "DEMO"} />
              </div>
              <p className="mt-0.5 text-xs text-slate-400">
                Retrieval. Reflection. Evolution. — Grounded orchestration & neural core telemetry.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setPulseKey((k) => k + 1)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/35 bg-cyan-500/10 px-2.5 py-1 font-mono text-[11px] text-cyan-200 hover:bg-cyan-500/20"
            >
              <Zap className="h-3.5 w-3.5" />
              Trigger Core Pulse
            </button>
          </div>

          <LivingRagCore
            coreState={coreState}
            pulseKey={pulseKey}
            reducedMotion={appearance.reducedMotion}
            lowPerfMode={appearance.lowPerfMode}
            ambientGlow={appearance.ambientGlow}
          />

          {/* Grounded Query Bar */}
          <div className="rounded-xl border border-slate-800 bg-[#070A10] p-3 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-mono text-[11px] uppercase tracking-wider text-slate-300">
                Buffered Grounded Query Stream (`POST /api/v1/rag/query/stream`)
              </span>
              <ProvenanceBadge kind={backendOnline ? "LIVE" : "UNAVAILABLE"} />
            </div>

            <div className="flex flex-wrap gap-2">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                aria-label="Ask a grounded question"
                placeholder="Ask a grounded question..."
                className="min-w-[220px] flex-1 rounded-lg border border-slate-800 bg-[#05070B] px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500/60"
              />
              <select
                value={retrievalMode}
                aria-label="Retrieval mode"
                onChange={(e) => setRetrievalMode(e.target.value as "hybrid" | "sparse" | "dense")}
                className="rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-200"
              >
                <option value="hybrid">hybrid</option>
                <option value="sparse">sparse</option>
                <option value="dense">dense</option>
              </select>
              <select
                value={strategy}
                aria-label="RAG strategy"
                onChange={(e) => setStrategy(e.target.value as "baseline" | "adaptive")}
                className="rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-200"
              >
                <option value="baseline">baseline</option>
                <option value="adaptive">adaptive</option>
              </select>
              <label className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={reflect}
                  onChange={(e) => setReflect(e.target.checked)}
                  className="accent-purple-400"
                />
                reflect
              </label>
              <button
                type="button"
                onClick={runQuery}
                disabled={!backendOnline || streaming || !question.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-2 font-mono text-xs font-medium text-cyan-200 transition hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-45"
              >
                <Play className="h-3.5 w-3.5" />
                {streaming ? "Streaming..." : "Execute"}
              </button>
            </div>

            {!backendOnline && (
              <div className="font-mono text-[11px] text-amber-300/90">
                Offline Demo Mode: Start FastAPI on `127.0.0.1:8000` to execute live grounded queries.
              </div>
            )}

            {events.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {events.map((ev, i) => (
                  <span
                    key={`${ev.event}-${i}`}
                    className="rounded border border-slate-700 bg-slate-900 px-2 py-0.5 font-mono text-[10px] text-cyan-300"
                  >
                    {ev.event}
                  </span>
                ))}
              </div>
            )}

            {error && (
              <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-2.5 font-mono text-xs text-rose-300">
                {error}
              </div>
            )}

            {result && (
              <div className="rounded-lg border border-slate-800 bg-[#05070B] p-3 space-y-2">
                <div className="flex items-center justify-between font-mono text-[11px]">
                  <span className="text-cyan-300">
                    status: {result.status} · admission: {result.admission_outcome}
                  </span>
                  <span className="text-slate-400">
                    mode: {result.effective_retrieval_mode} · trace: {result.trace_id}
                  </span>
                </div>
                <p className="text-sm leading-relaxed text-slate-100">{result.answer}</p>
                {result.citations.length > 0 && (
                  <div className="space-y-1 border-t border-slate-800 pt-2">
                    <div className="font-mono text-[10px] uppercase text-slate-400">
                      Citations ({result.citations.length})
                    </div>
                    {result.citations.map((c) => (
                      <div
                        key={c.evidence_id}
                        className="rounded border border-slate-800/80 bg-[#0B1018] p-2 font-mono text-[11px] text-slate-300"
                      >
                        [{c.citation_number}] {c.chunk_id} — {c.excerpt}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </SurfaceCard>

        {/* Right Status Stack (4 cols) */}
        <div className="xl:col-span-4 space-y-4">
          <SurfaceCard className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
                <Activity className="h-4 w-4 text-emerald-400" />
                SYSTEM HEALTH & MODELS
              </div>
              <ProvenanceBadge kind={backendOnline ? "LIVE" : "UNAVAILABLE"} />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <MetricReadout
                label="Backend API"
                value={backendOnline ? "HEALTHY" : "OFFLINE"}
                subtext={backendOnline ? "127.0.0.1:8000" : "No live server"}
                provenance={backendOnline ? "LIVE" : "UNAVAILABLE"}
              />
              <MetricReadout
                label="Model Inventory"
                value={backendOnline ? (modelsLoaded ? "READY" : "DEGRADED") : "UNAVAILABLE"}
                subtext="qwen3:4b · embed 0.6b"
                provenance={backendOnline ? (modelsLoaded ? "LIVE" : "BLOCKED") : "UNAVAILABLE"}
              />
            </div>
          </SurfaceCard>

          <SurfaceCard accent="purple" className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
                <Cpu className="h-4 w-4 text-purple-400" />
                RSI & ACTIVE CONFIG
              </div>
              <ProvenanceBadge kind={backendOnline ? "LIVE" : "DEMO"} />
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              <MetricReadout
                label="Active Version"
                value={activeConfig}
                subtext={`Auto-promo: ${String(rsiStatus?.automatic_promotion ?? false)}`}
                provenance={backendOnline ? "LIVE" : "DEMO"}
              />
              <MetricReadout
                label="RSI Cycles"
                value={cyclesCompleted}
                subtext={`Enabled: ${String(rsiStatus?.enabled ?? false)}`}
                provenance={backendOnline ? "LIVE" : "DEMO"}
              />
            </div>
          </SurfaceCard>

          <SurfaceCard className="p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
                <ShieldAlert className="h-4 w-4 text-amber-400" />
                RESOURCE & RUNTIME BLOCKERS
              </div>
              <ProvenanceBadge kind={backendOnline ? "LIVE" : "BLOCKED"} />
            </div>
            <div className="rounded-lg border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200/90 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
              <div>
                <div>{blockerMessage}</div>
                <div className="mt-1 font-mono text-[10px] text-slate-400">
                  Target Floor: 3 GiB free RAM · RTX 3050 (4 GiB VRAM)
                </div>
              </div>
            </div>
          </SurfaceCard>
        </div>
      </div>

      {/* Bottom Bento Row: Analytics, Evaluation Overview, Recent Events */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SurfaceCard className="p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-200">
              RETRIEVAL ANALYTICS
            </span>
            <ProvenanceBadge kind={backendOnline && ragStatus ? "LIVE" : "UNAVAILABLE"} />
          </div>
          <div className="grid grid-cols-2 gap-2 font-mono text-xs">
            <div className="rounded border border-slate-800 bg-[#080C12] p-2.5">
              <div className="text-[10px] text-slate-500">Default Mode</div>
              <div className="mt-1 text-slate-100">
                {String(ragStatus?.default_retrieval_mode ?? "hybrid")}
              </div>
            </div>
            <div className="rounded border border-slate-800 bg-[#080C12] p-2.5">
              <div className="text-[10px] text-slate-500">Recorded Traces</div>
              <div className="mt-1 text-slate-100">
                {ragStatus?.trace_count != null ? String(ragStatus.trace_count) : "Unavailable"}
              </div>
            </div>
          </div>
        </SurfaceCard>

        <SurfaceCard className="p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-200">
              EVALUATION & REFLECTION
            </span>
            <ProvenanceBadge kind={backendOnline && reflectionStatus ? "LIVE" : "UNAVAILABLE"} />
          </div>
          <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">Reflection Default</span>
              <span>{String(reflectionStatus?.default_enabled ?? false)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Max Cycles / Calls</span>
              <span>
                {String(reflectionStatus?.max_cycles ?? 1)} /{" "}
                {String(reflectionStatus?.max_llm_calls ?? 3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Offline Benchmark Gate</span>
              <span className="text-emerald-300">&gt;= 90% coverage</span>
            </div>
          </div>
        </SurfaceCard>

        <SurfaceCard className="p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-200">
              RECENT EXECUTION EVENTS
            </span>
            <button
              type="button"
              onClick={() =>
                onInspect({
                  title: "Command Center Telemetry",
                  kind: "telemetry",
                  provenance: backendOnline ? "live" : "demo",
                  summary: "Latest query & system status snapshot.",
                  fields: {
                    backendOnline,
                    activeConfig,
                    cyclesCompleted,
                    lastEventsCount: lastEvents.length,
                  },
                })
              }
              className="font-mono text-[10px] text-cyan-300 hover:underline"
            >
              Inspect
            </button>
          </div>
          {lastEvents.length > 0 ? (
            <div className="space-y-1 font-mono text-[11px] text-cyan-200">
              {lastEvents.map((e, idx) => (
                <div key={idx} className="rounded border border-slate-800 bg-[#080C12] px-2.5 py-1.5">
                  {e}
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded border border-slate-800 bg-[#080C12] p-3 font-mono text-[11px] text-slate-500">
              No live query executed in this session yet.
            </div>
          )}
        </SurfaceCard>
      </div>
    </div>
  );
}
