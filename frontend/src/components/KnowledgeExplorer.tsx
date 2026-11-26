import { useEffect, useMemo, useRef, useState } from "react";
import {
  Compass,
  Filter,
  GitFork,
  Layers,
  Maximize2,
  Minimize2,
  RotateCcw,
  Search,
  Table2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type {
  AppearanceSettings,
  GraphEdge,
  GraphNode,
  InspectorPayload,
  NodeKind,
  ProvenanceGraphResponse,
} from "../types";
import { ProvenanceBadge } from "./ui/Primitives";

interface SimNode extends GraphNode {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  radius: number;
}

const NODE_COLORS: Record<NodeKind, string> = {
  document: "#22D3EE",
  revision: "#38BDF8",
  chunk: "#0EA5E9",
  embedding: "#0284C7",
  config_version: "#A855F7",
  query: "#F59E0B",
};

const NODE_RADIUS: Record<NodeKind, number> = {
  document: 10.5,
  revision: 6.2,
  chunk: 3.6,
  embedding: 2.8,
  config_version: 8.5,
  query: 7.5,
};

function hashUnit(seed: string): number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 1000) / 1000;
}

export function KnowledgeExplorer({
  graph,
  highlightedChunkIds,
  activeQueryLabel,
  focusNodeId,
  appearance,
  onToggleLowPerf,
  onInspect,
}: {
  graph: ProvenanceGraphResponse;
  highlightedChunkIds: string[];
  activeQueryLabel: string | null;
  focusNodeId?: string | null;
  appearance: AppearanceSettings;
  onToggleLowPerf: () => void;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [search, setSearch] = useState("");
  const [selectedCluster, setSelectedCluster] = useState("all");
  const [visibleKinds, setVisibleKinds] = useState<Record<NodeKind, boolean>>({
    document: true,
    revision: true,
    chunk: true,
    embedding: true,
    config_version: true,
    query: true,
  });
  const [density, setDensity] = useState(100);
  const [showEdges, setShowEdges] = useState(true);
  const [perspective3D, setPerspective3D] = useState(false);
  const [tableMode, setTableMode] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [expandHops, setExpandHops] = useState<1 | 2>(1);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [camera, setCamera] = useState({ x: 0, y: 0, k: 0.72 });

  const simNodesRef = useRef<SimNode[]>([]);
  const simEdgesRef = useRef<GraphEdge[]>([]);
  const dragRef = useRef<{ active: boolean; x: number; y: number }>({
    active: false,
    x: 0,
    y: 0,
  });

  const clusters = useMemo(() => {
    const s = new Set(graph.nodes.map((n) => n.cluster));
    return ["all", ...Array.from(s)];
  }, [graph.nodes]);

  const filteredData = useMemo(() => {
    const maxCount = Math.max(1, Math.round((graph.nodes.length * density) / 100));
    const q = search.trim().toLowerCase();
    const nodes = graph.nodes
      .filter((n) => visibleKinds[n.kind])
      .filter((n) => selectedCluster === "all" || n.cluster === selectedCluster)
      .filter((n) => !q || n.label.toLowerCase().includes(q) || n.id.toLowerCase().includes(q))
      .slice(0, maxCount);
    const ids = new Set(nodes.map((n) => n.id));
    const edges = showEdges
      ? graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target))
      : [];
    return { nodes, edges };
  }, [graph, visibleKinds, selectedCluster, search, density, showEdges]);

  const neighborhoodSet = useMemo(() => {
    const focus = hoverId ?? selectedId;
    if (!focus) return new Set<string>();
    const hop1 = new Set<string>([focus]);
    for (const e of filteredData.edges) {
      if (e.source === focus) hop1.add(e.target);
      if (e.target === focus) hop1.add(e.source);
    }
    if (expandHops === 1) return hop1;
    const hop2 = new Set<string>(hop1);
    for (const e of filteredData.edges) {
      if (hop1.has(e.source)) hop2.add(e.target);
      if (hop1.has(e.target)) hop2.add(e.source);
    }
    return hop2;
  }, [expandHops, filteredData.edges, hoverId, selectedId]);

  useEffect(() => {
    const clusterList = Array.from(new Set(filteredData.nodes.map((n) => n.cluster)));
    const clusterAngle = new Map<string, { cx: number; cy: number }>();
    clusterList.forEach((c, idx) => {
      if (c === "rsi_evolution") {
        clusterAngle.set(c, { cx: 0, cy: 0 });
      } else {
        const angle = (idx / Math.max(1, clusterList.length)) * Math.PI * 2;
        const dist = 230 + (idx % 2) * 70;
        clusterAngle.set(c, { cx: Math.cos(angle) * dist, cy: Math.sin(angle) * dist });
      }
    });

    const nodes: SimNode[] = filteredData.nodes.map((n) => {
      const anchor = clusterAngle.get(n.cluster) ?? { cx: 0, cy: 0 };
      const u1 = hashUnit(n.id);
      const u2 = hashUnit(`${n.id}:a`);
      const spread = n.kind === "document" ? 18 : n.kind === "revision" ? 44 : 115;
      const theta = u1 * Math.PI * 2;
      const r = u2 * spread;
      return {
        ...n,
        x: anchor.cx + Math.cos(theta) * r,
        y: anchor.cy + Math.sin(theta) * r,
        z: (hashUnit(`${n.id}:z`) - 0.5) * 90,
        vx: 0,
        vy: 0,
        radius: NODE_RADIUS[n.kind],
      };
    });

    // Run short bounded layout relaxation synchronously once on data change
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const ticks = appearance.lowPerfMode ? 12 : 28;
    for (let t = 0; t < ticks; t++) {
      for (const e of filteredData.edges) {
        const a = byId.get(e.source);
        const b = byId.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        const target = e.kind === "document_has_revision" ? 36 : 58;
        const pull = (d - target) * 0.04;
        a.x += (dx / d) * pull;
        a.y += (dy / d) * pull;
        b.x -= (dx / d) * pull;
        b.y -= (dy / d) * pull;
      }
    }

    simNodesRef.current = nodes;
    simEdgesRef.current = filteredData.edges;
  }, [appearance.lowPerfMode, filteredData]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const obs = new IntersectionObserver(([entry]) => setIsVisible(entry.isIntersecting), {
      threshold: 0.05,
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  useEffect(() => {
    if (tableMode || !isVisible) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const dpr = appearance.lowPerfMode ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (appearance.ambientGlow && !appearance.lowPerfMode) {
      const grad = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, Math.max(w, h) * 0.55);
      grad.addColorStop(0, "rgba(34,211,238,0.07)");
      grad.addColorStop(1, "rgba(5,7,11,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }

    const project = (n: SimNode) => {
      const persp = perspective3D ? 1 / (1 - n.z * 0.0025) : 1;
      return {
        sx: w / 2 + (n.x * persp + camera.x) * camera.k,
        sy: h / 2 + (n.y * persp + camera.y) * camera.k,
        persp,
      };
    };

    const byId = new Map(simNodesRef.current.map((n) => [n.id, n]));
    const hasFocus = Boolean(hoverId ?? selectedId);

    // Draw edges
    for (const e of simEdgesRef.current) {
      const a = byId.get(e.source);
      const b = byId.get(e.target);
      if (!a || !b) continue;
      const pa = project(a);
      const pb = project(b);
      const inFocus =
        !hasFocus || (neighborhoodSet.has(a.id) && neighborhoodSet.has(b.id));
      const isRsiEdge = e.kind === "candidate_derives_from_parent";
      ctx.strokeStyle = isRsiEdge
        ? `rgba(168,85,247,${inFocus ? 0.65 : 0.08})`
        : `rgba(34,211,238,${inFocus ? 0.28 : 0.04})`;
      ctx.lineWidth = inFocus && hasFocus ? 1.3 : 0.75;
      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.lineTo(pb.sx, pb.sy);
      ctx.stroke();
    }

    // Draw nodes
    for (const n of simNodesRef.current) {
      const { sx, sy, persp } = project(n);
      if (sx < -40 || sx > w + 40 || sy < -40 || sy > h + 40) continue;
      const isHighlighted = highlightedChunkIds.includes(
        String(n.metadata.chunk_id ?? "")
      );
      const isSelected = n.id === selectedId || n.id === hoverId;
      const inFocus = !hasFocus || neighborhoodSet.has(n.id);
      const color = isHighlighted ? "#F59E0B" : NODE_COLORS[n.kind];
      const r = n.radius * (perspective3D ? persp : 1) * (isSelected ? 1.35 : 1);

      ctx.globalAlpha = inFocus ? 1 : 0.18;
      if ((n.kind === "document" || n.kind === "config_version" || isSelected) && !appearance.lowPerfMode) {
        ctx.beginPath();
        ctx.arc(sx, sy, r * 2.1, 0, Math.PI * 2);
        ctx.fillStyle = n.kind === "config_version" ? "rgba(168,85,247,0.18)" : "rgba(34,211,238,0.16)";
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = "#F8FAFC";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      const showLabel =
        isSelected ||
        n.kind === "document" ||
        n.kind === "config_version" ||
        (appearance.progressiveLabels ? camera.k >= 1.15 : camera.k >= 0.85);
      if (showLabel) {
        ctx.fillStyle = "#E2E8F0";
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillText(n.label, sx + r + 4, sy + 3);
      }
      ctx.globalAlpha = 1;
    }
  }, [
    appearance.ambientGlow,
    appearance.lowPerfMode,
    appearance.progressiveLabels,
    camera,
    filteredData,
    highlightedChunkIds,
    hoverId,
    isVisible,
    neighborhoodSet,
    perspective3D,
    selectedId,
    tableMode,
  ]);

  const selectedNode = useMemo(
    () => graph.nodes.find((n) => n.id === selectedId) ?? null,
    [graph.nodes, selectedId]
  );

  const focusOnNode = (nodeId: string) => {
    setSelectedId(nodeId);
    const sim = simNodesRef.current.find((n) => n.id === nodeId);
    if (sim) {
      setCamera((c) => ({ x: -sim.x, y: -sim.y, k: Math.max(c.k, 1.15) }));
      onInspect({
        title: sim.label,
        kind: sim.kind,
        provenance: graph.is_demo ? "demo" : "live",
        summary: `Cluster: ${sim.cluster} · Provenance node (${sim.kind})`,
        fields: sim.metadata,
      });
    }
  };

  useEffect(() => {
    if (focusNodeId) {
      focusOnNode(focusNodeId);
    }
  }, [focusNodeId]);

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await containerRef.current.requestFullscreen();
    }
  };

  const pickNodeAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    for (let i = simNodesRef.current.length - 1; i >= 0; i--) {
      const n = simNodesRef.current[i];
      const persp = perspective3D ? 1 / (1 - n.z * 0.0025) : 1;
      const sx = rect.width / 2 + (n.x * persp + camera.x) * camera.k;
      const sy = rect.height / 2 + (n.y * persp + camera.y) * camera.k;
      if (Math.hypot(mx - sx, my - sy) <= Math.max(8, n.radius + 4)) {
        return n;
      }
    }
    return null;
  };

  return (
    <div
      ref={containerRef}
      className="relative flex h-[calc(100vh-3.5rem)] min-h-[560px] flex-col lg:flex-row overflow-hidden bg-[#05070B]"
    >
      {/* Mobile Filter Toggle Bar */}
      <div className="flex items-center justify-between border-b border-slate-800 bg-[#090E17] px-3 py-2 lg:hidden">
        <button
          type="button"
          onClick={() => setMobileFiltersOpen((v) => !v)}
          className="inline-flex items-center gap-1.5 rounded border border-slate-700 px-2.5 py-1 font-mono text-xs text-slate-200"
        >
          <Filter className="h-3.5 w-3.5 text-cyan-400" />
          Filters & Nodes ({filteredData.nodes.length})
        </button>
        <ProvenanceBadge kind={graph.is_demo ? "DEMO" : "LIVE"} />
      </div>

      {/* Left Sidebar Panel */}
      <aside
        className={`${
          mobileFiltersOpen ? "flex" : "hidden"
        } lg:flex w-full lg:w-72 shrink-0 flex-col border-b lg:border-b-0 lg:border-r border-slate-800/80 bg-[#080C13] p-3.5 space-y-3 overflow-y-auto`}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-100">
              Provenance Explorer
            </h2>
            <p className="text-[10px] text-slate-500">
              Document-chunk provenance (not entity GraphRAG)
            </p>
          </div>
          <ProvenanceBadge kind={graph.is_demo ? "DEMO" : "LIVE"} />
        </div>

        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search nodes..."
            aria-label="Search provenance nodes"
            className="w-full rounded-lg border border-slate-800 bg-[#05070B] py-1.5 pl-8 pr-2.5 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500/50"
          />
        </div>

        <div className="space-y-1">
          <label className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
            Cluster Isolation
          </label>
          <select
            value={selectedCluster}
            aria-label="Cluster isolation filter"
            onChange={(e) => setSelectedCluster(e.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 font-mono text-xs text-slate-200"
          >
            {clusters.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
            Node Types
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(Object.keys(visibleKinds) as NodeKind[]).map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => setVisibleKinds((prev) => ({ ...prev, [kind]: !prev[kind] }))}
                className={`rounded border px-2 py-0.5 font-mono text-[10px] transition ${
                  visibleKinds[kind]
                    ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
                    : "border-slate-800 bg-slate-950 text-slate-500"
                }`}
              >
                {kind}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between font-mono text-[10px] text-slate-400">
            <span>GRAPH DENSITY</span>
            <span>{density}%</span>
          </div>
          <input
            type="range"
            min={15}
            max={100}
            value={density}
            aria-label="Graph density"
            onChange={(e) => setDensity(Number(e.target.value))}
            className="w-full accent-cyan-400"
          />
        </div>

        <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
          <label className="flex items-center justify-between">
            <span>Show Edges</span>
            <input
              type="checkbox"
              checked={showEdges}
              onChange={(e) => setShowEdges(e.target.checked)}
              className="accent-cyan-400"
            />
          </label>
          <label className="flex items-center justify-between">
            <span>Low-Perf Mode</span>
            <input
              type="checkbox"
              checked={appearance.lowPerfMode}
              onChange={onToggleLowPerf}
              className="accent-cyan-400"
            />
          </label>
        </div>

        <div className="flex-1 overflow-y-auto border-t border-slate-800/80 pt-2">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-400">
            Nodes ({filteredData.nodes.length} / {graph.nodes.length})
          </div>
          <div className="space-y-1">
            {filteredData.nodes.slice(0, 60).map((n) => (
              <button
                key={n.id}
                type="button"
                onClick={() => focusOnNode(n.id)}
                className={`flex w-full items-center justify-between rounded px-2 py-1 text-left font-mono text-[11px] transition ${
                  selectedId === n.id
                    ? "bg-cyan-500/15 text-cyan-200"
                    : "text-slate-300 hover:bg-slate-800/60"
                }`}
              >
                <span className="truncate">{n.label}</span>
                <span className="ml-2 shrink-0 text-[9px] text-slate-500">{n.kind}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Center Graph Workspace */}
      <main className="relative flex-1 overflow-hidden bg-[#05070B]">
        {/* Top Floating Controls */}
        <div className="pointer-events-none absolute left-3 right-3 top-3 z-20 flex flex-wrap items-center justify-between gap-2">
          <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 rounded-lg border border-slate-800 bg-[#090E17]/90 p-1 backdrop-blur-sm">
            <button
              type="button"
              onClick={() => setTableMode((v) => !v)}
              className="inline-flex items-center gap-1 rounded px-2 py-1 font-mono text-[11px] text-slate-200 hover:bg-slate-800"
            >
              <Table2 className="h-3.5 w-3.5 text-cyan-400" />
              {tableMode ? "Graph View" : "Table View"}
            </button>
            <button
              type="button"
              onClick={() => setPerspective3D((v) => !v)}
              className={`inline-flex items-center gap-1 rounded px-2 py-1 font-mono text-[11px] ${
                perspective3D ? "bg-purple-500/20 text-purple-200" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              {perspective3D ? "3D Depth" : "2D Flat"}
            </button>
            <button
              type="button"
              aria-label="Zoom in graph"
              onClick={() => setCamera((c) => ({ ...c, k: Math.min(2.6, c.k + 0.18) }))}
              className="rounded p-1 text-slate-300 hover:bg-slate-800"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              aria-label="Zoom out graph"
              onClick={() => setCamera((c) => ({ ...c, k: Math.max(0.32, c.k - 0.18) }))}
              className="rounded p-1 text-slate-300 hover:bg-slate-800"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setCamera({ x: 0, y: 0, k: 0.72 })}
              className="inline-flex items-center gap-1 rounded px-2 py-1 font-mono text-[11px] text-slate-300 hover:bg-slate-800"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </button>
            <button
              type="button"
              onClick={toggleFullscreen}
              className="inline-flex items-center gap-1 rounded px-2 py-1 font-mono text-[11px] text-slate-300 hover:bg-slate-800"
            >
              {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
              Fullscreen
            </button>
          </div>

          {activeQueryLabel && (
            <div className="pointer-events-auto rounded-lg border border-amber-500/35 bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] text-amber-200">
              Query Highlight: {activeQueryLabel}
            </div>
          )}
        </div>

        {tableMode ? (
          <div className="h-full overflow-auto p-4 pt-16">
            <table className="w-full border-collapse font-mono text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-left text-slate-400">
                  <th className="py-2 pr-4">Label</th>
                  <th className="py-2 pr-4">Kind</th>
                  <th className="py-2 pr-4">Cluster</th>
                  <th className="py-2">Provenance ID</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.nodes.map((n) => (
                  <tr
                    key={n.id}
                    onClick={() => focusOnNode(n.id)}
                    className="cursor-pointer border-b border-slate-900 text-slate-200 hover:bg-slate-900/60"
                  >
                    <td className="py-1.5 pr-4">{n.label}</td>
                    <td className="py-1.5 pr-4 text-cyan-300">{n.kind}</td>
                    <td className="py-1.5 pr-4">{n.cluster}</td>
                    <td className="py-1.5 text-slate-500">{n.id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            className="h-full w-full cursor-grab active:cursor-grabbing"
            onWheel={(e) => {
              const delta = e.deltaY < 0 ? 0.1 : -0.1;
              setCamera((c) => ({
                ...c,
                k: Math.max(0.32, Math.min(2.6, Number((c.k + delta).toFixed(2)))),
              }));
            }}
            onPointerDown={(e) => {
              const hit = pickNodeAt(e.clientX, e.clientY);
              if (hit) {
                focusOnNode(hit.id);
                return;
              }
              dragRef.current = { active: true, x: e.clientX, y: e.clientY };
            }}
            onPointerMove={(e) => {
              if (dragRef.current.active) {
                const dx = (e.clientX - dragRef.current.x) / camera.k;
                const dy = (e.clientY - dragRef.current.y) / camera.k;
                setCamera((c) => ({ ...c, x: c.x + dx, y: c.y + dy }));
                dragRef.current.x = e.clientX;
                dragRef.current.y = e.clientY;
                return;
              }
              const hit = pickNodeAt(e.clientX, e.clientY);
              setHoverId(hit ? hit.id : null);
            }}
            onPointerUp={() => {
              dragRef.current.active = false;
            }}
          />
        )}

        {/* Bottom Legend & Minimap */}
        {!tableMode && (
          <div className="pointer-events-none absolute bottom-3 left-3 right-3 z-20 flex flex-wrap items-end justify-between gap-2">
            <div className="pointer-events-auto flex flex-wrap items-center gap-3 rounded-lg border border-slate-800 bg-[#090E17]/90 px-3 py-1.5 font-mono text-[10px] text-slate-300 backdrop-blur-sm">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-cyan-400" /> Document Hub
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-sky-500" /> Chunk / Rev
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-purple-500" /> RSI Config
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-amber-400" /> Query / Cited
              </span>
            </div>

            <div className="pointer-events-auto hidden sm:flex items-center gap-2 rounded-lg border border-slate-800 bg-[#090E17]/90 px-2.5 py-1.5 font-mono text-[10px] text-slate-400">
              <Compass className="h-3.5 w-3.5 text-cyan-400" />
              <span>Zoom {(camera.k * 100).toFixed(0)}%</span>
              <span>·</span>
              <span>Hops: {expandHops}</span>
            </div>
          </div>
        )}
      </main>

      {/* Right Contextual Node Inspector */}
      <aside className="w-full lg:w-72 shrink-0 border-t lg:border-t-0 lg:border-l border-slate-800/80 bg-[#080C13] p-3.5 overflow-y-auto">
        <div className="flex items-center justify-between">
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-100">
            Node Inspector
          </h3>
          {selectedNode && (
            <button
              type="button"
              onClick={() => setExpandHops((h) => (h === 1 ? 2 : 1))}
              className="inline-flex items-center gap-1 rounded border border-cyan-500/35 bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] text-cyan-200"
            >
              <GitFork className="h-3 w-3" />
              Expand {expandHops === 1 ? "2-Hop" : "1-Hop"}
            </button>
          )}
        </div>

        {selectedNode ? (
          <div className="mt-3 space-y-3">
            <div className="rounded-lg border border-slate-800 bg-[#05070B] p-2.5">
              <div className="font-mono text-xs font-semibold text-cyan-200">
                {selectedNode.label}
              </div>
              <div className="mt-0.5 font-mono text-[10px] text-slate-400">
                {selectedNode.kind} · cluster: {selectedNode.cluster}
              </div>
            </div>
            <div className="space-y-1 rounded-lg border border-slate-800 bg-[#05070B] p-2.5 font-mono text-[11px]">
              {Object.entries(selectedNode.metadata).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <span className="text-slate-500">{k}</span>
                  <span className="truncate text-slate-200">{String(v ?? "null")}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            Select a node in the graph or hierarchy to inspect its provenance, token spans,
            neighbor expansion, and connected edges.
          </p>
        )}
      </aside>
    </div>
  );
}
