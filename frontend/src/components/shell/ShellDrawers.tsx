import { Bell, Eye, Sliders, X } from "lucide-react";
import type {
  AppearanceSettings,
  InspectorPayload,
  NotificationItem,
} from "../../types";
import { ProvenanceBadge } from "../ui/Primitives";

export function ShellDrawer({
  activeDrawer,
  onClose,
  notifications,
  inspector,
  appearance,
  onUpdateAppearance,
}: {
  activeDrawer: "notifications" | "inspector" | "appearance" | null;
  onClose: () => void;
  notifications: NotificationItem[];
  inspector: InspectorPayload | null;
  appearance: AppearanceSettings;
  onUpdateAppearance: (patch: Partial<AppearanceSettings>) => void;
}) {
  if (!activeDrawer) return null;

  return (
    <aside
      aria-label={`${activeDrawer} panel`}
      className="fixed inset-y-0 right-0 z-40 flex w-full max-w-sm flex-col border-l border-slate-800 bg-[#090D14]/95 shadow-2xl backdrop-blur-md"
    >
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wider text-slate-200">
          {activeDrawer === "notifications" && <Bell className="h-4 w-4 text-cyan-400" />}
          {activeDrawer === "inspector" && <Eye className="h-4 w-4 text-purple-400" />}
          {activeDrawer === "appearance" && <Sliders className="h-4 w-4 text-cyan-400" />}
          {activeDrawer === "notifications"
            ? "System Notifications"
            : activeDrawer === "inspector"
              ? "Contextual Inspector"
              : "Appearance & Performance"}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close drawer"
          className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4 text-xs">
        {activeDrawer === "notifications" && (
          <>
            {notifications.map((n) => (
              <div
                key={n.id}
                className="rounded-lg border border-slate-800 bg-[#0D131D] p-3 space-y-1.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-slate-100">{n.title}</span>
                  <ProvenanceBadge
                    kind={n.source === "live" ? "LIVE" : n.source === "demo" ? "DEMO" : "STATIC"}
                  />
                </div>
                <p className="text-[11px] leading-relaxed text-slate-400">{n.detail}</p>
                <div className="font-mono text-[10px] text-slate-500">{n.timestamp}</div>
              </div>
            ))}
          </>
        )}

        {activeDrawer === "inspector" && (
          <>
            {inspector ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-semibold text-cyan-200">
                    {inspector.title}
                  </span>
                  <ProvenanceBadge
                    kind={
                      inspector.provenance === "live"
                        ? "LIVE"
                        : inspector.provenance === "demo"
                          ? "DEMO"
                          : "STATIC"
                    }
                  />
                </div>
                <p className="text-slate-300 leading-relaxed">{inspector.summary}</p>
                <div className="rounded-lg border border-slate-800 bg-[#06090E] p-3 font-mono text-[11px] space-y-1.5">
                  {Object.entries(inspector.fields).map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between gap-3">
                      <span className="text-slate-500">{k}</span>
                      <span className="truncate text-slate-200">{String(v ?? "null")}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-slate-800 bg-[#0D131D] p-4 text-slate-400">
                Select any node, pipeline stage, or RSI candidate to inspect its verified metadata
                here.
              </div>
            )}
          </>
        )}

        {activeDrawer === "appearance" && (
          <div className="space-y-3">
            {(
              [
                [
                  "reducedMotion",
                  "Reduced Motion",
                  "Disables continuous neural rotation and dash animations.",
                ],
                [
                  "lowPerfMode",
                  "Low-Performance Mode",
                  "Caps graph DPR to 1x and simplifies canvas glow effects.",
                ],
                [
                  "progressiveLabels",
                  "Progressive Zoom Labels",
                  "Reveals chunk labels smoothly only when zoomed in.",
                ],
                [
                  "ambientGlow",
                  "Ambient Cyan & Purple Glow",
                  "Renders subtle hardware-friendly radial highlights.",
                ],
              ] as const
            ).map(([key, title, desc]) => (
              <label
                key={key}
                className="flex cursor-pointer items-start justify-between gap-3 rounded-lg border border-slate-800 bg-[#0D131D] p-3"
              >
                <div>
                  <div className="font-medium text-slate-100">{title}</div>
                  <div className="mt-0.5 text-[11px] text-slate-400">{desc}</div>
                </div>
                <input
                  type="checkbox"
                  checked={appearance[key]}
                  onChange={(e) => onUpdateAppearance({ [key]: e.target.checked })}
                  className="mt-1 accent-cyan-400"
                />
              </label>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
