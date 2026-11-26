import type { ReactNode } from "react";

export function EvoRagLogo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5 select-none">
      <div className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-cyan-500/35 bg-[#09121C] shadow-[0_0_16px_rgba(34,211,238,0.18)]">
        <svg viewBox="0 0 32 32" className="h-5 w-5" aria-hidden="true">
          <circle
            cx="16"
            cy="16"
            r="10.5"
            fill="none"
            stroke="#22D3EE"
            strokeOpacity="0.35"
            strokeWidth="1.2"
          />
          <line x1="9" y1="11.5" x2="23" y2="11.5" stroke="#22D3EE" strokeWidth="1.3" />
          <line x1="9" y1="11.5" x2="16" y2="23" stroke="#A855F7" strokeWidth="1.3" />
          <line x1="23" y1="11.5" x2="16" y2="23" stroke="#22D3EE" strokeWidth="1.3" />
          <circle cx="9" cy="11.5" r="2.3" fill="#22D3EE" />
          <circle cx="23" cy="11.5" r="2.3" fill="#22D3EE" />
          <circle cx="16" cy="23" r="2.5" fill="#A855F7" />
        </svg>
      </div>
      {!compact && (
        <div className="min-w-0 leading-tight">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-xs font-semibold tracking-[0.16em] text-slate-100">
              EVORAG
            </span>
            <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-1 py-0.2 font-mono text-[9px] text-cyan-300">
              v0.11
            </span>
          </div>
          <div className="truncate text-[10px] text-slate-400">Neural Intelligence Console</div>
        </div>
      )}
    </div>
  );
}

export type ProvenanceKind = "LIVE" | "DEMO" | "UNAVAILABLE" | "BLOCKED" | "STATIC" | "PLANNED";

const PROVENANCE_STYLES: Record<ProvenanceKind, string> = {
  LIVE: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  DEMO: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  UNAVAILABLE: "border-slate-700 bg-slate-900/90 text-slate-400",
  BLOCKED: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  STATIC: "border-cyan-500/35 bg-cyan-500/10 text-cyan-300",
  PLANNED: "border-purple-500/40 bg-purple-500/10 text-purple-300",
};

export function ProvenanceBadge({
  kind,
  label,
}: {
  kind: ProvenanceKind;
  label?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${PROVENANCE_STYLES[kind]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          kind === "LIVE"
            ? "bg-emerald-400"
            : kind === "DEMO"
              ? "bg-amber-400"
              : kind === "BLOCKED"
                ? "bg-rose-400"
                : kind === "PLANNED"
                  ? "bg-purple-400"
                  : kind === "STATIC"
                    ? "bg-cyan-400"
                    : "bg-slate-500"
        }`}
      />
      {label ?? kind}
    </span>
  );
}

export function SurfaceCard({
  children,
  className = "",
  accent = "none",
}: {
  children: ReactNode;
  className?: string;
  accent?: "none" | "cyan" | "purple";
}) {
  const accentRing =
    accent === "cyan"
      ? "hover:border-cyan-500/40 shadow-[0_0_24px_rgba(34,211,238,0.05)]"
      : accent === "purple"
        ? "hover:border-purple-500/40 shadow-[0_0_24px_rgba(168,85,247,0.06)]"
        : "hover:border-slate-700/90";

  return (
    <section
      className={`rounded-xl border border-slate-800/80 bg-[#0B1018]/90 backdrop-blur-sm transition-colors duration-200 ${accentRing} ${className}`}
    >
      {children}
    </section>
  );
}

export function MetricReadout({
  label,
  value,
  subtext,
  provenance,
}: {
  label: string;
  value: string;
  subtext?: string;
  provenance: ProvenanceKind;
}) {
  return (
    <div className="rounded-lg border border-slate-800/80 bg-[#080C12] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
          {label}
        </span>
        <ProvenanceBadge kind={provenance} />
      </div>
      <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-slate-100">
        {value}
      </div>
      {subtext && <div className="mt-0.5 text-[11px] text-slate-400">{subtext}</div>}
    </div>
  );
}

export function SkeletonBlock({ className = "h-12 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-slate-800/50 ${className}`} />;
}
