import { useMemo, useState } from "react";
import { AlertTriangle, FileText, Network, Play, SlidersHorizontal } from "lucide-react";
import { ApiError, api } from "../api";
import { DEMO_RETRIEVE_RESPONSE } from "../data/demoCorpus";
import type {
  CrossViewNavigationTarget,
  InspectorPayload,
  RetrieveResponse,
  WorkspaceMode,
} from "../types";
import { ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

interface SimSampleDoc {
  id: string;
  label: string;
  denseSim: number;
  dot: number;
  euclideanDist: number;
  bm25: number;
}

const SIM_SAMPLE_DOCS: SimSampleDoc[] = [
  { id: "doc-hybrid:c0", label: "hybrid_bm25_rrf.md#c0", denseSim: 0.89, dot: 0.89, euclideanDist: 0.47, bm25: 6.4 },
  { id: "doc-arch:c0", label: "architecture.md#c0", denseSim: 0.82, dot: 0.82, euclideanDist: 0.60, bm25: 4.1 },
  { id: "doc-ram:c0", label: "resource_admission.md#c0", denseSim: 0.77, dot: 0.77, euclideanDist: 0.68, bm25: 7.2 },
  { id: "doc-rsi:c0", label: "rsi_governance.md#c0", denseSim: 0.65, dot: 0.65, euclideanDist: 0.84, bm25: 2.3 },
];

export function RetrievalLab({
  workspaceMode,
  backendOnline,
  onOpenDocuments,
  onOpenExplorer,
  onInspect,
}: {
  workspaceMode: WorkspaceMode;
  backendOnline: boolean;
  onOpenDocuments: (target: CrossViewNavigationTarget) => void;
  onOpenExplorer: (target: CrossViewNavigationTarget) => void;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const isDemo = workspaceMode === "demo";
  const [query, setQuery] = useState("hybrid BM25 RRF fusion and 3 GiB floor");
  const [mode, setMode] = useState<"hybrid" | "sparse" | "dense">("hybrid");
  const [reranking, setReranking] = useState(true);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [response, setResponse] = useState<RetrieveResponse | null>(
    isDemo ? DEMO_RETRIEVE_RESPONSE : null
  );

  // Educational simulator controls (never sent to backend config)
  const [simMetric, setSimMetric] = useState<"cosine" | "dot" | "euclidean" | "bm25" | "rrf">(
    "rrf"
  );
  const [denseWeight, setDenseWeight] = useState(0.55);
  const sparseWeight = Number((1 - denseWeight).toFixed(2));

  const simRankings = useMemo(() => {
    const denseSorted = SIM_SAMPLE_DOCS.slice().sort((a, b) => b.denseSim - a.denseSim);
    const sparseSorted = SIM_SAMPLE_DOCS.slice().sort((a, b) => b.bm25 - a.bm25);
    const k = 60;

    return SIM_SAMPLE_DOCS.map((d) => {
      const dRank = denseSorted.findIndex((x) => x.id === d.id) + 1;
      const sRank = sparseSorted.findIndex((x) => x.id === d.id) + 1;
      const rrf = denseWeight / (k + dRank) + sparseWeight / (k + sRank);
      const score =
        simMetric === "cosine"
          ? d.denseSim
          : simMetric === "dot"
            ? d.dot
            : simMetric === "euclidean"
              ? 1 / (1 + d.euclideanDist)
              : simMetric === "bm25"
                ? d.bm25
                : rrf;
      return { ...d, dRank, sRank, rrf, score };
    }).sort((a, b) => b.score - a.score);
  }, [denseWeight, simMetric, sparseWeight]);

  const runRetrieve = async () => {
    setError(null);
    if (isDemo) {
      setResponse({ ...DEMO_RETRIEVE_RESPONSE, mode, effective_mode: mode, reranking_applied: reranking });
      setLatencyMs(7.4);
      return;
    }
    setLoading(true);
    const t0 = performance.now();
    try {
      const res = await api.retrieve({
        query,
        mode,
        top_k: topK,
        reranking_enabled: reranking,
        include_explanations: true,
      });
      setResponse(res);
      setLatencyMs(Number((performance.now() - t0).toFixed(1)));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Retrieval request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1440px] space-y-4 p-4 lg:p-5 animate-stagger">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-mono text-sm font-semibold tracking-wider text-slate-100">
              RETRIEVAL LAB · SCIENTIFIC RANKING COMPARISON
            </h1>
            <ProvenanceBadge kind={isDemo ? "DEMO" : backendOnline ? "LIVE" : "UNAVAILABLE"} />
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Compare Dense, BM25, Hybrid RRF, and optional heuristic reranking via `POST /api/v1/retrieve`.
          </p>
        </div>
      </div>

      {/* Query Control Surface */}
      <SurfaceCard accent="cyan" className="p-4 space-y-3">
        <div className="flex flex-wrap gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Retrieval Lab query"
            placeholder="Enter retrieval test query..."
            className="min-w-[240px] flex-1 rounded-lg border border-slate-800 bg-[#05070B] px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500/60"
          />
          <select
            aria-label="Retrieval Lab mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as "hybrid" | "sparse" | "dense")}
            className="rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-200"
          >
            <option value="hybrid">hybrid (Dense + BM25 + RRF)</option>
            <option value="sparse">sparse (BM25)</option>
            <option value="dense">dense (Exact Cosine)</option>
          </select>
          <select
            aria-label="Top K candidates"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-200"
          >
            <option value={3}>top_k = 3</option>
            <option value={5}>top_k = 5</option>
            <option value={10}>top_k = 10</option>
          </select>
          <label className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-2 font-mono text-xs text-slate-300">
            <input
              type="checkbox"
              checked={reranking}
              onChange={(e) => setReranking(e.target.checked)}
              className="accent-purple-400"
            />
            rerank
          </label>
          <button
            type="button"
            onClick={() => void runRetrieve()}
            disabled={loading || !query.trim() || (!isDemo && !backendOnline)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-2 font-mono text-xs text-cyan-200 hover:bg-cyan-500/25 disabled:opacity-45"
          >
            <Play className="h-3.5 w-3.5" />
            {loading ? "Running..." : "Compare Rankings"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 font-mono text-xs text-rose-300">
            {error}
          </div>
        )}

        {response?.degraded && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 font-mono text-xs text-amber-200">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            Degraded Fallback ({response.effective_mode}): {response.degradation_reason}
          </div>
        )}

        {response && (
          <div className="space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-slate-400">
              <span>
                effective_mode: <strong className="text-cyan-300">{response.effective_mode}</strong>{" "}
                · reranking: {String(response.reranking_applied)} · results:{" "}
                {response.results.length}
              </span>
              <span>{latencyMs != null ? `latency: ${latencyMs} ms` : ""}</span>
            </div>

            <div className="space-y-2">
              {response.results.map((r) => {
                const exp = response.explanations.find((e) => e.chunk_id === r.chunk.chunk_id);
                return (
                  <div
                    key={r.chunk.chunk_id}
                    className="rounded-xl border border-slate-800 bg-[#06090F] p-3 space-y-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 font-mono text-xs">
                        <span className="rounded border border-cyan-500/40 bg-cyan-500/15 px-1.5 py-0.5 text-cyan-200">
                          #{r.rank}
                        </span>
                        <span className="font-semibold text-slate-100">{r.chunk.chunk_id}</span>
                        <span className="text-slate-500">doc: {r.chunk.document_id}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() =>
                            onOpenDocuments({
                              documentId: r.chunk.document_id,
                              chunkId: r.chunk.chunk_id,
                            })
                          }
                          className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-0.5 font-mono text-[10px] text-slate-200 hover:border-cyan-500/40"
                        >
                          <FileText className="h-3 w-3 text-cyan-400" />
                          Inspect Chunk
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            onOpenExplorer({
                              documentId: r.chunk.document_id,
                              chunkId: r.chunk.chunk_id,
                              nodeId: `chunk:${r.chunk.chunk_id}`,
                            })
                          }
                          className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-0.5 font-mono text-[10px] text-slate-200 hover:border-purple-500/40"
                        >
                          <Network className="h-3 w-3 text-purple-400" />
                          Graph Node
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            onInspect({
                              title: `Rank #${r.rank} · ${r.chunk.chunk_id}`,
                              kind: "retrieval_candidate",
                              provenance: isDemo ? "demo" : "live",
                              summary: r.chunk.text,
                              fields: {
                                score: r.score,
                                dense_score: exp?.dense_score ?? null,
                                bm25_score: exp?.bm25_score ?? null,
                                rrf_score: exp?.rrf_score ?? null,
                                rerank_score: exp?.rerank_score ?? null,
                              },
                            })
                          }
                          className="font-mono text-[10px] text-cyan-300 hover:underline"
                        >
                          Inspect
                        </button>
                      </div>
                    </div>

                    <p className="text-xs text-slate-300">{r.chunk.text}</p>

                    <div className="grid grid-cols-2 gap-2 font-mono text-[10px] sm:grid-cols-5">
                      <div className="rounded bg-[#0B1018] px-2 py-1">
                        score: <span className="text-cyan-300">{r.score.toFixed(4)}</span>
                      </div>
                      <div className="rounded bg-[#0B1018] px-2 py-1">
                        dense: {exp?.dense_score != null ? exp.dense_score.toFixed(3) : "—"}
                      </div>
                      <div className="rounded bg-[#0B1018] px-2 py-1">
                        bm25: {exp?.bm25_score != null ? exp.bm25_score.toFixed(2) : "—"}
                      </div>
                      <div className="rounded bg-[#0B1018] px-2 py-1">
                        rrf: {exp?.rrf_score != null ? exp.rrf_score.toFixed(4) : "—"}
                      </div>
                      <div className="rounded bg-[#0B1018] px-2 py-1">
                        terms: {exp?.matched_terms.join(", ") || "—"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </SurfaceCard>

      {/* Educational Mathematical Simulator */}
      <SurfaceCard accent="purple" className="p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4 text-purple-400" />
            <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-100">
              Educational Retrieval Math Simulator
            </h2>
          </div>
          <ProvenanceBadge
            kind="DEMO"
            label="EDUCATIONAL SIMULATION · DOES NOT AFFECT LIVE CONFIG"
          />
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-12 items-center">
          <div className="md:col-span-4 space-y-2 font-mono text-xs">
            <label className="block text-[10px] uppercase text-slate-400">Scoring Function</label>
            <select
              aria-label="Simulator metric"
              value={simMetric}
              onChange={(e) =>
                setSimMetric(
                  e.target.value as "cosine" | "dot" | "euclidean" | "bm25" | "rrf"
                )
              }
              className="w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 text-slate-200"
            >
              <option value="rrf">Reciprocal Rank Fusion (RRF)</option>
              <option value="cosine">Cosine Similarity</option>
              <option value="dot">Dot Product</option>
              <option value="euclidean">Euclidean Proximity (1 / (1 + d))</option>
              <option value="bm25">Okapi BM25</option>
            </select>

            <div className="pt-1">
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>Dense Weight: {denseWeight.toFixed(2)}</span>
                <span>Sparse Weight: {sparseWeight.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={denseWeight}
                aria-label="Simulator dense weight"
                onChange={(e) => setDenseWeight(Number(e.target.value))}
                className="w-full accent-purple-400"
              />
            </div>
          </div>

          <div className="md:col-span-8 space-y-1.5">
            {simRankings.map((item, idx) => (
              <div
                key={item.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-[#06090F] px-3 py-2 font-mono text-[11px]"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-purple-300 font-semibold">#{idx + 1}</span>
                  <span className="truncate text-slate-200">{item.label}</span>
                </div>
                <div className="flex items-center gap-3 shrink-0 text-slate-400">
                  <span>d_rank={item.dRank}</span>
                  <span>s_rank={item.sRank}</span>
                  <span className="text-cyan-300 font-semibold">{item.score.toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </SurfaceCard>
    </div>
  );
}
