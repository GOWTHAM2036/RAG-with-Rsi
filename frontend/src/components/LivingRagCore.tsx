import { useEffect, useRef, useState } from "react";
import { Maximize2, Minimize2, RotateCcw, Sparkles, ZoomIn, ZoomOut } from "lucide-react";

export type CoreStateMode = "idle" | "retrieval" | "reflection" | "evolution";

interface SphereNode {
  x: number;
  y: number;
  z: number;
  kind: "cyan" | "purple";
  weight: number;
}

function buildFibonacciSphere(count: number): SphereNode[] {
  const nodes: SphereNode[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    nodes.push({
      x: Math.cos(theta) * radius,
      y,
      z: Math.sin(theta) * radius,
      kind: i % 5 === 0 ? "purple" : "cyan",
      weight: i % 9 === 0 ? 1.7 : 1,
    });
  }
  return nodes;
}

const SPHERE_NODES = buildFibonacciSphere(76);

export function LivingRagCore({
  coreState,
  pulseKey,
  reducedMotion,
  lowPerfMode,
  ambientGlow,
}: {
  coreState: CoreStateMode;
  pulseKey: number;
  reducedMotion: boolean;
  lowPerfMode: boolean;
  ambientGlow: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode2D, setMode2D] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [isVisible, setIsVisible] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const rotRef = useRef({ rx: 0.28, ry: 0.4 });
  const dragRef = useRef<{ active: boolean; x: number; y: number }>({
    active: false,
    x: 0,
    y: 0,
  });
  const pulseProgressRef = useRef(0);

  useEffect(() => {
    if (pulseKey > 0) {
      pulseProgressRef.current = 1;
    }
  }, [pulseKey]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const obs = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.05 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;

    const renderFrame = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        if (!reducedMotion && isVisible) {
          rafId = requestAnimationFrame(renderFrame);
        }
        return;
      }

      const dpr = lowPerfMode ? 1 : Math.min(window.devicePixelRatio || 1, 2);
      const targetW = Math.floor(rect.width * dpr);
      const targetH = Math.floor(rect.height * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) * 0.34 * zoom;

      ctx.clearRect(0, 0, w, h);

      if (ambientGlow && !lowPerfMode) {
        const isPurple = coreState === "reflection" || coreState === "evolution";
        const glow = ctx.createRadialGradient(cx, cy, radius * 0.1, cx, cy, radius * 1.4);
        glow.addColorStop(
          0,
          isPurple ? "rgba(168,85,247,0.18)" : "rgba(34,211,238,0.16)"
        );
        glow.addColorStop(1, "rgba(5,7,11,0)");
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, w, h);
      }

      if (!reducedMotion && !dragRef.current.active && isVisible) {
        rotRef.current.ry += 0.0045;
      }
      if (pulseProgressRef.current > 0) {
        pulseProgressRef.current = Math.max(0, pulseProgressRef.current - 0.025);
      }

      const { rx, ry } = rotRef.current;
      const cosY = Math.cos(ry);
      const sinY = Math.sin(ry);
      const cosX = Math.cos(rx);
      const sinX = Math.sin(rx);

      const projected = SPHERE_NODES.map((n, idx) => {
        if (mode2D) {
          const angle = (idx / SPHERE_NODES.length) * Math.PI * 2;
          const ring = 0.45 + 0.5 * ((idx % 4) / 3);
          return {
            sx: cx + Math.cos(angle + ry * 0.4) * radius * ring,
            sy: cy + Math.sin(angle + ry * 0.4) * radius * ring,
            z: 0,
            kind: n.kind,
            weight: n.weight,
          };
        }
        const x1 = n.x * cosY - n.z * sinY;
        const z1 = n.x * sinY + n.z * cosY;
        const y2 = n.y * cosX - z1 * sinX;
        const z2 = n.y * sinX + z1 * cosX;
        const scale = 2.6 / (2.6 - z2);
        return {
          sx: cx + x1 * radius * scale,
          sy: cy + y2 * radius * scale,
          z: z2,
          kind: n.kind,
          weight: n.weight,
        };
      });

      // Sparse lattice edges
      const pulse = pulseProgressRef.current;
      ctx.lineWidth = 0.8 + pulse * 0.9;
      for (let i = 0; i < projected.length; i++) {
        const a = projected[i];
        if (!mode2D && a.z < -0.55) continue;
        for (let step = 1; step <= 2; step++) {
          const b = projected[(i + step * 5) % projected.length];
          const usePurple =
            coreState === "reflection" ||
            coreState === "evolution" ||
            a.kind === "purple";
          const alpha = (mode2D ? 0.25 : 0.12 + (a.z + 1) * 0.14) + pulse * 0.28;
          ctx.strokeStyle = usePurple
            ? `rgba(168,85,247,${Math.min(0.8, alpha)})`
            : `rgba(34,211,238,${Math.min(0.8, alpha)})`;
          ctx.beginPath();
          ctx.moveTo(a.sx, a.sy);
          ctx.lineTo(b.sx, b.sy);
          ctx.stroke();
        }
      }

      // Depth-sorted nodes
      projected
        .slice()
        .sort((a, b) => a.z - b.z)
        .forEach((p) => {
          const depthAlpha = mode2D ? 0.9 : 0.28 + ((p.z + 1) / 2) * 0.72;
          const isPurple =
            p.kind === "purple" ||
            coreState === "reflection" ||
            coreState === "evolution";
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, (1.8 + pulse * 1.4) * p.weight, 0, Math.PI * 2);
          ctx.fillStyle = isPurple
            ? `rgba(192,132,252,${depthAlpha})`
            : `rgba(34,211,238,${depthAlpha})`;
          ctx.fill();
        });

      if (!reducedMotion && isVisible) {
        rafId = requestAnimationFrame(renderFrame);
      }
    };

    renderFrame();
    return () => cancelAnimationFrame(rafId);
  }, [ambientGlow, coreState, isVisible, lowPerfMode, mode2D, pulseKey, reducedMotion, zoom]);

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await containerRef.current.requestFullscreen();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative flex h-64 w-full flex-col overflow-hidden rounded-xl border border-slate-800/90 bg-[#05080F]"
    >
      <div className="z-10 flex items-center justify-between border-b border-slate-800/70 bg-[#090E17]/80 px-3 py-1.5 backdrop-blur-sm">
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-slate-300">
          <Sparkles
            className={`h-3.5 w-3.5 ${
              coreState === "reflection" || coreState === "evolution"
                ? "text-purple-400"
                : "text-cyan-400"
            }`}
          />
          <span>Living RAG Core</span>
          <span className="rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 text-[9px] text-cyan-300">
            {mode2D ? "2D FALLBACK" : "3D SPHERE"} · {coreState.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setMode2D((v) => !v)}
            className="rounded border border-slate-700/80 px-1.5 py-0.5 font-mono text-[10px] text-slate-300 hover:border-cyan-500/40"
          >
            {mode2D ? "3D" : "2D"}
          </button>
          <button
            type="button"
            aria-label="Zoom in neural core"
            onClick={() => setZoom((z) => Math.min(1.6, Number((z + 0.15).toFixed(2))))}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Zoom out neural core"
            onClick={() => setZoom((z) => Math.max(0.65, Number((z - 0.15).toFixed(2))))}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Reset neural core camera"
            onClick={() => {
              setZoom(1);
              rotRef.current = { rx: 0.28, ry: 0.4 };
            }}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Toggle core fullscreen"
            onClick={toggleFullscreen}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="h-full w-full flex-1 cursor-grab active:cursor-grabbing"
        onPointerDown={(e) => {
          dragRef.current = { active: true, x: e.clientX, y: e.clientY };
        }}
        onPointerMove={(e) => {
          if (!dragRef.current.active) return;
          const dx = e.clientX - dragRef.current.x;
          const dy = e.clientY - dragRef.current.y;
          rotRef.current.ry += dx * 0.008;
          rotRef.current.rx += dy * 0.008;
          dragRef.current.x = e.clientX;
          dragRef.current.y = e.clientY;
        }}
        onPointerUp={() => {
          dragRef.current.active = false;
        }}
        onPointerLeave={() => {
          dragRef.current.active = false;
        }}
      />
    </div>
  );
}
