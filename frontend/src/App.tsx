import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bell,
  ChevronRight,
  Command,
  Cpu,
  Database,
  Eye,
  FileText,
  FlaskConical,
  GitBranch,
  Layers,
  Menu,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings,
  Sliders,
  Sparkles,
  Terminal,
  Workflow,
  X,
} from "lucide-react";
import { api, DEMO_PROVENANCE_GRAPH } from "./api";
import { CommandCenter } from "./components/CommandCenter";
import { KnowledgeExplorer } from "./components/KnowledgeExplorer";
import { PipelineExplorer } from "./components/PipelineExplorer";
import { PlannedPageView } from "./components/PlannedPageView";
import { RSIEvolutionTree } from "./components/RSIEvolutionTree";
import { CommandPalette } from "./components/shell/CommandPalette";
import { ShellDrawer } from "./components/shell/ShellDrawers";
import { EvoRagLogo, ProvenanceBadge, SkeletonBlock } from "./components/ui/Primitives";
import { DEMO_DOCUMENTS } from "./data/demoCorpus";
import type {
  AppearanceSettings,
  ConfigurationVersion,
  CrossViewNavigationTarget,
  DocumentInfo,
  InspectorPayload,
  ModelAvailability,
  NavTabId,
  NotificationItem,
  PlannedTabId,
  PromotionRecommendation,
  ProvenanceGraphResponse,
  RAGResponse,
  WorkspaceMode,
} from "./types";

const RAGPlayground = lazy(() =>
  import("./components/RAGPlayground").then((m) => ({ default: m.RAGPlayground }))
);
const DocumentManagement = lazy(() =>
  import("./components/DocumentManagement").then((m) => ({ default: m.DocumentManagement }))
);
const RetrievalLab = lazy(() =>
  import("./components/RetrievalLab").then((m) => ({ default: m.RetrievalLab }))
);

interface NavItem {
  id: NavTabId;
  label: string;
  group: "Signature Views" | "AI Workspaces" | "Labs & Infra (Planned)";
  icon: typeof Command;
  planned?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { id: "command", label: "Command Center", group: "Signature Views", icon: Command },
  { id: "explorer", label: "Knowledge Explorer", group: "Signature Views", icon: Network },
  { id: "pipeline", label: "Pipeline Explorer", group: "Signature Views", icon: Workflow },
  { id: "rsi", label: "RSI Evolution Tree", group: "Signature Views", icon: GitBranch },
  { id: "playground", label: "RAG Playground", group: "AI Workspaces", icon: Terminal },
  { id: "documents", label: "Document Management", group: "AI Workspaces", icon: FileText },
  { id: "retrieval", label: "Retrieval Lab", group: "AI Workspaces", icon: Layers },
  { id: "evaluation", label: "Evaluation Arena", group: "Labs & Infra (Planned)", icon: FlaskConical, planned: true },
  { id: "reflection", label: "Reflection Lab", group: "Labs & Infra (Planned)", icon: Sparkles, planned: true },
  { id: "experiments", label: "Experiment Lab", group: "Labs & Infra (Planned)", icon: Database, planned: true },
  { id: "models", label: "Model Manager", group: "Labs & Infra (Planned)", icon: Cpu, planned: true },
  { id: "monitor", label: "System Monitor", group: "Labs & Infra (Planned)", icon: Activity, planned: true },
  { id: "traces", label: "Execution Traces", group: "Labs & Infra (Planned)", icon: Search, planned: true },
  { id: "settings", label: "Settings", group: "Labs & Infra (Planned)", icon: Settings, planned: true },
];

const NAV_GROUPS: Array<NavItem["group"]> = [
  "Signature Views",
  "AI Workspaces",
  "Labs & Infra (Planned)",
];

const EMPTY_LIVE_GRAPH: ProvenanceGraphResponse = {
  nodes: [],
  edges: [],
  total_nodes: 0,
  total_edges: 0,
  truncated: false,
  is_demo: false,
};

export function App() {
  const [tab, setTab] = useState<NavTabId>("command");
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("live");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [activeDrawer, setActiveDrawer] = useState<
    "notifications" | "inspector" | "appearance" | null
  >(null);

  const [backendOnline, setBackendOnline] = useState(false);
  const [liveGraph, setLiveGraph] = useState<ProvenanceGraphResponse>(EMPTY_LIVE_GRAPH);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [models, setModels] = useState<ModelAvailability | null>(null);
  const [readyStatus, setReadyStatus] = useState<Record<string, unknown> | null>(null);
  const [ragStatus, setRagStatus] = useState<Record<string, unknown> | null>(null);
  const [reflectionStatus, setReflectionStatus] = useState<Record<string, unknown> | null>(null);
  const [rsiStatus, setRsiStatus] = useState<Record<string, unknown> | null>(null);
  const [versions, setVersions] = useState<ConfigurationVersion[]>([]);
  const [recommendations, setRecommendations] = useState<PromotionRecommendation[]>([]);
  const [highlightedChunks, setHighlightedChunks] = useState<string[]>([]);
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  const [lastEvents, setLastEvents] = useState<string[]>([]);
  const [inspector, setInspector] = useState<InspectorPayload | null>(null);
  const [docNavTarget, setDocNavTarget] = useState<CrossViewNavigationTarget | null>(null);
  const [explorerFocusNodeId, setExplorerFocusNodeId] = useState<string | null>(null);

  const [appearance, setAppearance] = useState<AppearanceSettings>(() => ({
    reducedMotion:
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    lowPerfMode: false,
    progressiveLabels: true,
    ambientGlow: true,
  }));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
        setActiveDrawer(null);
        setMobileNavOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const refreshLiveState = useCallback(async () => {
    try {
      await api.health();
      const [g, docs, mdl, rd, rs, rfs, rsis, vers, recs] = await Promise.all([
        api.provenanceGraph(250),
        api.listDocuments(50, 0).catch(() => []),
        api.modelsStatus().catch(() => null),
        api.ready().catch(() => null),
        api.ragStatus(),
        api.reflectionStatus(),
        api.rsiStatus(),
        api.rsiConfigurations(),
        api.rsiRecommendations(),
      ]);
      setBackendOnline(true);
      setLiveGraph(g);
      setDocuments(docs);
      setModels(mdl);
      setReadyStatus(rd);
      setRagStatus(rs);
      setReflectionStatus(rfs);
      setRsiStatus(rsis);
      setVersions(vers);
      setRecommendations(recs);
    } catch {
      setBackendOnline(false);
      setLiveGraph(EMPTY_LIVE_GRAPH);
      setDocuments([]);
      setModels(null);
    }
  }, []);

  useEffect(() => {
    void refreshLiveState();
  }, [refreshLiveState]);

  const activeGraph = workspaceMode === "demo" ? DEMO_PROVENANCE_GRAPH : liveGraph;

  const notifications = useMemo<NotificationItem[]>(
    () => [
      {
        id: "n-conn",
        title: backendOnline ? "FastAPI Backend Connected" : "Backend Unreachable (Live Mode)",
        detail: backendOnline
          ? "Live provenance, document ingestion, retrieval, and streaming endpoints are active."
          : "Live endpoints are unreachable on 127.0.0.1:8000. Switch explicitly to Demo Mode to explore synthetic data.",
        level: backendOnline ? "info" : "warning",
        timestamp: "Session check",
        source: workspaceMode,
      },
      {
        id: "n-floor",
        title: "3 GiB RAM Admission Floor Active",
        detail:
          "Inference admission guards local RAM/VRAM so model downloads or OOM calls never occur silently.",
        level: "info",
        timestamp: "Policy",
        source: "system",
      },
    ],
    [backendOnline, workspaceMode]
  );

  const handleQueryCompleted = (question: string, response: RAGResponse) => {
    setActiveQuery(question);
    setHighlightedChunks(response.citations.map((c) => c.chunk_id));
    setLastEvents([
      `mode=${response.effective_retrieval_mode}`,
      `outcome=${response.admission_outcome}`,
      `citations=${response.citations.length}`,
    ]);
  };

  const openDocuments = (target: CrossViewNavigationTarget) => {
    setDocNavTarget(target);
    setTab("documents");
  };

  const openExplorer = (target: CrossViewNavigationTarget) => {
    if (target.chunkId) setHighlightedChunks([target.chunkId]);
    if (target.nodeId) setExplorerFocusNodeId(target.nodeId);
    setTab("explorer");
  };

  const openPipeline = (events: string[]) => {
    if (events.length > 0) setLastEvents(events);
    setTab("pipeline");
  };

  const currentNav = NAV_ITEMS.find((n) => n.id === tab) ?? NAV_ITEMS[0];
  const modelsReady = Boolean(
    models ? models.generation_installed : readyStatus && readyStatus.status === "ready"
  );

  return (
    <div
      className={`flex min-h-screen bg-[#05070B] text-slate-100 ${
        appearance.reducedMotion ? "reduced-motion" : ""
      }`}
    >
      {mobileNavOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      {/* Collapsible Left Sidebar */}
      <aside
        aria-label="Primary Sidebar Navigation"
        className={`fixed inset-y-0 left-0 z-40 flex flex-col border-r border-slate-800/80 bg-[#070B12] transition-all duration-200 md:static ${
          mobileNavOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        } ${sidebarCollapsed ? "md:w-16" : "w-60"}`}
      >
        <div className="flex h-14 items-center justify-between border-b border-slate-800/80 px-3">
          <EvoRagLogo compact={sidebarCollapsed} />
          <button
            type="button"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close mobile sidebar"
            className="rounded p-1 text-slate-400 hover:bg-slate-800 md:hidden"
          >
            <X className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setSidebarCollapsed((c) => !c)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="hidden rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100 md:inline-flex"
          >
            {sidebarCollapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </button>
        </div>

        <nav className="flex-1 space-y-4 overflow-y-auto px-2 py-3">
          {NAV_GROUPS.map((group) => (
            <div key={group} className="space-y-1">
              {!sidebarCollapsed && (
                <div className="px-2 font-mono text-[10px] uppercase tracking-wider text-slate-500">
                  {group}
                </div>
              )}
              {NAV_ITEMS.filter((i) => i.group === group).map((item) => {
                const Icon = item.icon;
                const active = tab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    title={item.label}
                    onClick={() => {
                      setTab(item.id);
                      setMobileNavOpen(false);
                    }}
                    className={`flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-xs transition ${
                      active
                        ? "border border-cyan-500/40 bg-cyan-500/12 text-cyan-200"
                        : "border border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                    }`}
                  >
                    <span className="flex items-center gap-2.5 truncate">
                      <Icon className="h-4 w-4 shrink-0" />
                      {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                    </span>
                    {!sidebarCollapsed && item.planned && (
                      <span className="rounded border border-purple-500/30 bg-purple-500/10 px-1 py-0.2 font-mono text-[9px] text-purple-300">
                        PLAN
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        {!sidebarCollapsed && (
          <div className="border-t border-slate-800/80 p-3 font-mono text-[10px] text-slate-500">
            Retrieval. Reflection. Evolution.
          </div>
        )}
      </aside>

      {/* Main Workspace Column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between gap-2 border-b border-slate-800/80 bg-[#080C14]/95 px-3 lg:px-5 backdrop-blur-md">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation menu"
              className="rounded border border-slate-800 p-1.5 text-slate-300 hover:bg-slate-800 md:hidden"
            >
              <Menu className="h-4 w-4" />
            </button>

            <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 font-mono text-xs truncate">
              <span className="hidden sm:inline text-slate-500">EvoRAG</span>
              <ChevronRight className="hidden sm:inline h-3.5 w-3.5 text-slate-600" />
              <span className="hidden md:inline text-slate-400">{currentNav.group}</span>
              <ChevronRight className="hidden md:inline h-3.5 w-3.5 text-slate-600" />
              <span className="truncate font-semibold text-slate-100">{currentNav.label}</span>
            </nav>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-2">
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="hidden sm:inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1 font-mono text-[11px] text-slate-400 hover:border-cyan-500/40 hover:text-slate-200"
            >
              <Search className="h-3.5 w-3.5" />
              <span>Command</span>
              <kbd className="rounded bg-slate-900 px-1 text-[10px] text-slate-500">Ctrl+K</kbd>
            </button>

            <span className="hidden lg:inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#05070B] px-2 py-1 font-mono text-[10px] text-slate-300">
              <Cpu className="h-3 w-3 text-slate-400" />
              Models: {backendOnline ? (modelsReady ? "Ready" : "Missing/Degraded") : "Unavailable"}
            </span>

            {/* Explicit Consented Live vs Demo Mode Switch */}
            <div
              role="group"
              aria-label="Workspace data mode"
              className="inline-flex rounded-lg border border-slate-800 bg-[#05070B] p-0.5 font-mono text-[10px]"
            >
              <button
                type="button"
                onClick={() => setWorkspaceMode("live")}
                className={`rounded px-2 py-0.5 transition ${
                  workspaceMode === "live"
                    ? "bg-emerald-500/20 text-emerald-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                LIVE
              </button>
              <button
                type="button"
                onClick={() => setWorkspaceMode("demo")}
                className={`rounded px-2 py-0.5 transition ${
                  workspaceMode === "demo"
                    ? "bg-amber-500/20 text-amber-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                DEMO
              </button>
            </div>

            <ProvenanceBadge
              kind={
                workspaceMode === "demo"
                  ? "DEMO"
                  : backendOnline
                    ? "LIVE"
                    : "UNAVAILABLE"
              }
            />

            <button
              type="button"
              aria-label="Open notifications"
              onClick={() =>
                setActiveDrawer((d) => (d === "notifications" ? null : "notifications"))
              }
              className="rounded-lg border border-slate-800 p-1.5 text-slate-300 hover:border-cyan-500/40 hover:text-cyan-200"
            >
              <Bell className="h-4 w-4" />
            </button>
            <button
              type="button"
              aria-label="Open global inspector"
              onClick={() => setActiveDrawer((d) => (d === "inspector" ? null : "inspector"))}
              className="rounded-lg border border-slate-800 p-1.5 text-slate-300 hover:border-purple-500/40 hover:text-purple-200"
            >
              <Eye className="h-4 w-4" />
            </button>
            <button
              type="button"
              aria-label="Open appearance settings"
              onClick={() => setActiveDrawer((d) => (d === "appearance" ? null : "appearance"))}
              className="rounded-lg border border-slate-800 p-1.5 text-slate-300 hover:border-cyan-500/40 hover:text-cyan-200"
            >
              <Sliders className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Explicit Offline Live-Mode Banner */}
        {workspaceMode === "live" && !backendOnline && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-500/30 bg-amber-500/10 px-4 py-1.5 font-mono text-[11px] text-amber-200">
            <span>
              Live Mode active, but FastAPI backend on `127.0.0.1:8000` is unreachable. Live
              data will not be mixed with synthetic demo fixtures.
            </span>
            <button
              type="button"
              onClick={() => setWorkspaceMode("demo")}
              className="rounded border border-amber-400/50 bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-100 hover:bg-amber-500/30"
            >
              Switch to Demo Mode
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {tab === "command" && (
            <CommandCenter
              backendOnline={backendOnline}
              ragStatus={ragStatus}
              reflectionStatus={reflectionStatus}
              rsiStatus={rsiStatus}
              readyStatus={readyStatus}
              appearance={appearance}
              lastEvents={lastEvents}
              onQueryCompleted={handleQueryCompleted}
              onInspect={setInspector}
            />
          )}
          {tab === "explorer" && (
            <KnowledgeExplorer
              graph={activeGraph}
              highlightedChunkIds={highlightedChunks}
              activeQueryLabel={activeQuery}
              focusNodeId={explorerFocusNodeId}
              appearance={appearance}
              onToggleLowPerf={() =>
                setAppearance((a) => ({ ...a, lowPerfMode: !a.lowPerfMode }))
              }
              onInspect={setInspector}
            />
          )}
          {tab === "pipeline" && (
            <PipelineExplorer
              lastEvents={lastEvents}
              backendOnline={backendOnline}
              onInspect={setInspector}
            />
          )}
          {tab === "rsi" && (
            <RSIEvolutionTree
              versions={versions}
              recommendations={recommendations}
              backendOnline={backendOnline}
              onInspect={setInspector}
            />
          )}
          <Suspense
            fallback={
              <div className="p-6 space-y-3">
                <SkeletonBlock className="h-10 w-64" />
                <SkeletonBlock className="h-64 w-full" />
              </div>
            }
          >
            {tab === "playground" && (
              <RAGPlayground
                workspaceMode={workspaceMode}
                backendOnline={backendOnline}
                documents={workspaceMode === "demo" ? DEMO_DOCUMENTS : documents}
                models={models}
                onQueryCompleted={handleQueryCompleted}
                onOpenDocuments={openDocuments}
                onOpenExplorer={openExplorer}
                onOpenPipeline={openPipeline}
                onInspect={setInspector}
              />
            )}
            {tab === "documents" && (
              <DocumentManagement
                workspaceMode={workspaceMode}
                backendOnline={backendOnline}
                documents={documents}
                navigationTarget={docNavTarget}
                onDocumentsChanged={() => void refreshLiveState()}
                onOpenExplorer={openExplorer}
                onInspect={setInspector}
              />
            )}
            {tab === "retrieval" && (
              <RetrievalLab
                workspaceMode={workspaceMode}
                backendOnline={backendOnline}
                onOpenDocuments={openDocuments}
                onOpenExplorer={openExplorer}
                onInspect={setInspector}
              />
            )}
          </Suspense>
          {currentNav.planned && <PlannedPageView pageId={tab as PlannedTabId} />}
        </div>
      </div>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onNavigate={(nextTab) => setTab(nextTab)}
        onToggleReducedMotion={() =>
          setAppearance((a) => ({ ...a, reducedMotion: !a.reducedMotion }))
        }
        onToggleLowPerf={() =>
          setAppearance((a) => ({ ...a, lowPerfMode: !a.lowPerfMode }))
        }
      />

      <ShellDrawer
        activeDrawer={activeDrawer}
        onClose={() => setActiveDrawer(null)}
        notifications={notifications}
        inspector={inspector}
        appearance={appearance}
        onUpdateAppearance={(patch) => setAppearance((a) => ({ ...a, ...patch }))}
      />
    </div>
  );
}
