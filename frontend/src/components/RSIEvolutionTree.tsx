import { useMemo, useState } from "react";
import { GitBranch, Scale, ShieldCheck, Terminal } from "lucide-react";
import { DEMO_RSI_CANDIDATES } from "../data/demoCorpus";
import type {
  ConfigurationVersion,
  InspectorPayload,
  PromotionRecommendation,
  RSICandidateRecord,
} from "../types";
import { ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

const OUTCOME_STYLE: Record<RSICandidateRecord["outcome"], string> = {
  baseline: "border-cyan-500/40 bg-cyan-500/10 text-cyan-200",
  experimental_incumbent: "border-purple-500/40 bg-purple-500/15 text-purple-200",
  inconclusive: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  rejected: "border-rose-500/40 bg-rose-500/10 text-rose-300",
};

export function RSIEvolutionTree({
  versions,
  recommendations,
  backendOnline,
  onInspect,
}: {
  versions: ConfigurationVersion[];
  recommendations: PromotionRecommendation[];
  backendOnline: boolean;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const isDemo = !backendOnline || versions.length === 0;

  const records = useMemo<RSICandidateRecord[]>(() => {
    if (!isDemo) {
      return versions.map((v) => ({
        versionId: v.version_id,
        parentVersionId: v.parent_version_id,
        fingerprint: v.fingerprint,
        outcome: v.active ? "baseline" : "inconclusive",
        approved: v.approved,
        active: v.active,
        isDemo: false,
        changedParameters: {},
        qualityDelta: "see eval store",
        ci95: "via experiment report",
        latencyDeltaMs: "—",
        ramDeltaMiB: "—",
        notes: `Live SQLite RSI configuration (${v.created_at}).`,
      }));
    }
    return DEMO_RSI_CANDIDATES;
  }, [isDemo, versions]);

  const [selectedId, setSelectedId] = useState<string>(records[1]?.versionId ?? records[0]?.versionId ?? "v0_baseline");
  const baseline = records.find((r) => r.outcome === "baseline") ?? records[0];
  const candidate = records.find((r) => r.versionId === selectedId) ?? records[0];

  return (
    <div className="mx-auto max-w-[1440px] space-y-4 p-4 lg:p-5 animate-stagger">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-mono text-sm font-semibold tracking-wider text-slate-100">
              RSI EVOLUTION TREE · CONFIGURATION LINEAGE
            </h1>
            <ProvenanceBadge kind={isDemo ? "DEMO" : "LIVE"} />
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Bounded evolutionary configuration tree with statistical intervals & owner CLI gates.
          </p>
        </div>

        <span className="inline-flex items-center gap-1.5 rounded-lg border border-purple-500/35 bg-purple-500/10 px-2.5 py-1 font-mono text-[11px] text-purple-200">
          <ShieldCheck className="h-3.5 w-3.5" />
          Promotion & Rollback: Owner CLI Only
        </span>
      </div>

      {/* Lineage Tree + Side-by-Side Comparison */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <SurfaceCard accent="purple" className="lg:col-span-6 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
              <GitBranch className="h-4 w-4 text-purple-400" />
              LINEAGE NODES ({records.length})
            </div>
            {isDemo && (
              <span className="font-mono text-[10px] text-amber-300">
                Offline Demo Lineage (no fabricated improvements)
              </span>
            )}
          </div>

          <div className="space-y-2">
            {records.map((rec) => {
              const isSelected = rec.versionId === candidate.versionId;
              return (
                <button
                  key={rec.versionId}
                  type="button"
                  onClick={() => {
                    setSelectedId(rec.versionId);
                    onInspect({
                      title: rec.versionId,
                      kind: "config_version",
                      provenance: rec.isDemo ? "demo" : "live",
                      summary: rec.notes,
                      fields: {
                        outcome: rec.outcome,
                        fingerprint: rec.fingerprint,
                        approved: rec.approved,
                        quality_delta: rec.qualityDelta,
                        ci95: rec.ci95,
                      },
                    });
                  }}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    rec.parentVersionId ? "ml-0 sm:ml-5 sm:w-[calc(100%-1.25rem)]" : ""
                  } ${
                    isSelected
                      ? "border-purple-400 bg-[#140F22]"
                      : "border-slate-800 bg-[#080C12] hover:border-slate-700"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-xs font-semibold text-slate-100">
                      {rec.versionId}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase ${OUTCOME_STYLE[rec.outcome]}`}
                      >
                        {rec.outcome}
                      </span>
                      <span className="font-mono text-[10px] text-slate-500">
                        approved: {String(rec.approved)}
                      </span>
                    </div>
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-slate-400">
                    parent: {rec.parentVersionId ?? "root"} · fp: {rec.fingerprint} · ΔQ:{" "}
                    {rec.qualityDelta}
                  </div>
                  <p className="mt-1.5 text-[11px] text-slate-300">{rec.notes}</p>
                </button>
              );
            })}
          </div>
        </SurfaceCard>

        {/* Side-by-Side Candidate Comparison */}
        <SurfaceCard accent="cyan" className="lg:col-span-6 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
              <Scale className="h-4 w-4 text-cyan-400" />
              SIDE-BY-SIDE COMPARISON
            </div>
            <ProvenanceBadge kind={candidate.isDemo ? "DEMO" : "LIVE"} />
          </div>

          <div className="grid grid-cols-2 gap-2.5 font-mono text-xs">
            <div className="rounded-lg border border-slate-800 bg-[#070A10] p-3">
              <div className="text-[10px] text-slate-500">BASELINE</div>
              <div className="mt-1 font-semibold text-cyan-200">{baseline.versionId}</div>
              <div className="mt-1 text-[10px] text-slate-400">fp: {baseline.fingerprint}</div>
            </div>
            <div className="rounded-lg border border-purple-500/35 bg-[#120D1E] p-3">
              <div className="text-[10px] text-slate-500">SELECTED CANDIDATE</div>
              <div className="mt-1 font-semibold text-purple-200">{candidate.versionId}</div>
              <div className="mt-1 text-[10px] text-slate-400">
                outcome: {candidate.outcome}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 font-mono text-[11px] sm:grid-cols-4">
            <div className="rounded border border-slate-800 bg-[#070A10] p-2">
              <div className="text-[9px] text-slate-500">QUALITY Δ</div>
              <div className="mt-0.5 text-slate-100">{candidate.qualityDelta}</div>
            </div>
            <div className="rounded border border-slate-800 bg-[#070A10] p-2">
              <div className="text-[9px] text-slate-500">95% CI</div>
              <div className="mt-0.5 text-slate-100">{candidate.ci95}</div>
            </div>
            <div className="rounded border border-slate-800 bg-[#070A10] p-2">
              <div className="text-[9px] text-slate-500">LATENCY Δ</div>
              <div className="mt-0.5 text-slate-100">{candidate.latencyDeltaMs}</div>
            </div>
            <div className="rounded border border-slate-800 bg-[#070A10] p-2">
              <div className="text-[9px] text-slate-500">RAM Δ</div>
              <div className="mt-0.5 text-slate-100">{candidate.ramDeltaMiB}</div>
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-[#070A10] p-3 space-y-1.5">
            <div className="font-mono text-[10px] uppercase text-slate-400">
              Changed Parameters
            </div>
            {Object.keys(candidate.changedParameters).length > 0 ? (
              <div className="space-y-1 font-mono text-[11px]">
                {Object.entries(candidate.changedParameters).map(([param, diff]) => (
                  <div key={param} className="flex justify-between gap-2">
                    <span className="text-slate-300">{param}</span>
                    <span className="text-purple-300">
                      {String(diff.before)} → {String(diff.after)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="font-mono text-[11px] text-slate-500">
                No parameter delta (reference baseline or unexpanded live record).
              </div>
            )}
          </div>
        </SurfaceCard>
      </div>

      {/* Recommendations & Owner CLI */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <SurfaceCard className="lg:col-span-6 p-4 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-semibold text-slate-200">
              PROMOTION RECOMMENDATIONS ({recommendations.length})
            </span>
            <ProvenanceBadge kind={backendOnline ? "LIVE" : "UNAVAILABLE"} />
          </div>
          {recommendations.length === 0 ? (
            <p className="text-xs text-slate-400">
              No pending recommendations. Run an isolated RSI cycle via{" "}
              <code className="text-cyan-300">POST /api/v1/rsi/cycles</code> when{" "}
              <code className="text-cyan-300">[rsi].enabled = true</code>.
            </p>
          ) : (
            recommendations.map((rec) => (
              <div
                key={rec.recommendation_id}
                className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-3 font-mono text-[11px]"
              >
                <div className="text-purple-200">{rec.recommendation_id}</div>
                <div className="mt-1 text-slate-300">{rec.expected_benefit}</div>
                <div className="mt-1 text-slate-500">
                  eligible={String(rec.eligible_for_live_promotion)} · rollback=
                  {rec.rollback_target_version}
                </div>
              </div>
            ))
          )}
        </SurfaceCard>

        <SurfaceCard className="lg:col-span-6 p-4 space-y-2">
          <div className="flex items-center gap-2 font-mono text-xs font-semibold text-slate-200">
            <Terminal className="h-4 w-4 text-cyan-400" />
            OWNER-OPERATED CLI WORKFLOW (NO HTTP PROMOTION)
          </div>
          <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-[#05070B] p-3 font-mono text-[11px] text-slate-300">
            {`python -m evorag.rsi_cli status
python -m evorag.rsi_cli configurations
python -m evorag.rsi_cli recommendations`}
          </pre>
        </SurfaceCard>
      </div>
    </div>
  );
}
