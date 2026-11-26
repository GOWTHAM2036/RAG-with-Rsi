import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  Copy,
  Edit3,
  FileText,
  Network,
  Plus,
  RefreshCw,
  Send,
  ShieldAlert,
  Square,
  Trash2,
  Workflow,
} from "lucide-react";
import { ApiError, api } from "../api";
import { DEMO_CONVERSATIONS, DEMO_DOCUMENTS } from "../data/demoCorpus";
import type {
  CitationReference,
  ConversationMessage,
  ConversationThread,
  CrossViewNavigationTarget,
  DocumentInfo,
  InspectorPayload,
  ModelAvailability,
  RAGResponse,
  RAGStreamEvent,
  WorkspaceMode,
} from "../types";
import { parseMarkdownBlocks, renderInlineMarkdown } from "../utils/markdown";
import { ProvenanceBadge } from "./ui/Primitives";

const MAX_LOCAL_CONVERSATIONS = 20;

export function RAGPlayground({
  workspaceMode,
  backendOnline,
  documents,
  models,
  onQueryCompleted,
  onOpenDocuments,
  onOpenExplorer,
  onOpenPipeline,
  onInspect,
}: {
  workspaceMode: WorkspaceMode;
  backendOnline: boolean;
  documents: DocumentInfo[];
  models: ModelAvailability | null;
  onQueryCompleted: (question: string, response: RAGResponse) => void;
  onOpenDocuments: (target: CrossViewNavigationTarget) => void;
  onOpenExplorer: (target: CrossViewNavigationTarget) => void;
  onOpenPipeline: (events: string[], traceId?: string) => void;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const isDemo = workspaceMode === "demo";
  const [threads, setThreads] = useState<ConversationThread[]>(() =>
    isDemo
      ? DEMO_CONVERSATIONS
      : [
          {
            id: "live-conv-1",
            title: "Session 1",
            createdAt: new Date().toISOString(),
            isDemo: false,
            messages: [],
          },
        ]
  );
  const [activeId, setActiveId] = useState<string>(threads[0]?.id ?? "live-conv-1");
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;

  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<"hybrid" | "sparse" | "dense">("hybrid");
  const [strategy, setStrategy] = useState<"baseline" | "adaptive">("baseline");
  const [reflectionMode, setReflectionMode] = useState<"off" | "deterministic" | "semantic">(
    "off"
  );
  const [docScope, setDocScope] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [streamEvents, setStreamEvents] = useState<RAGStreamEvent[]>([]);
  const [streamWarning, setStreamWarning] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<CitationReference | null>(null);
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (isDemo) {
      setThreads(DEMO_CONVERSATIONS);
      setActiveId(DEMO_CONVERSATIONS[0].id);
    } else {
      setThreads((prev) => {
        const liveOnly = prev.filter((t) => !t.isDemo);
        if (liveOnly.length > 0) return liveOnly;
        const fresh: ConversationThread = {
          id: `live-conv-${Date.now()}`,
          title: "Live Research Session",
          createdAt: new Date().toISOString(),
          isDemo: false,
          messages: [],
        };
        return [fresh];
      });
    }
    setSelectedCitation(null);
  }, [isDemo]);

  const activeThread = useMemo(
    () => threads.find((t) => t.id === activeId) ?? threads[0],
    [activeId, threads]
  );

  const lastAssistantMsg = useMemo(
    () =>
      activeThread?.messages
        .slice()
        .reverse()
        .find((m) => m.role === "assistant" && m.response) ?? null,
    [activeThread]
  );

  const availableDocs = isDemo ? DEMO_DOCUMENTS : documents;

  const createConversation = () => {
    const newThread: ConversationThread = {
      id: `conv-${Date.now()}`,
      title: `Session ${threads.length + 1}`,
      createdAt: new Date().toISOString(),
      isDemo,
      messages: [],
    };
    setThreads((prev) => [newThread, ...prev].slice(0, MAX_LOCAL_CONVERSATIONS));
    setActiveId(newThread.id);
    setSelectedCitation(null);
  };

  const removeConversation = (id: string) => {
    setThreads((prev) => {
      const next = prev.filter((t) => t.id !== id);
      if (next.length === 0) {
        const fallback: ConversationThread = {
          id: `conv-${Date.now()}`,
          title: "Session 1",
          createdAt: new Date().toISOString(),
          isDemo,
          messages: [],
        };
        setActiveId(fallback.id);
        return [fallback];
      }
      if (activeId === id) setActiveId(next[0].id);
      return next;
    });
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  };

  const submitQuestion = async (overrideQuestion?: string) => {
    const q = (overrideQuestion ?? question).trim();
    if (!q || loading) return;

    const targetThreadId = activeThread.id;
    const userMsg: ConversationMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: q,
      timestamp: new Date().toISOString(),
      isDemo,
    };

    setThreads((prev) =>
      prev.map((t) => (t.id === targetThreadId ? { ...t, messages: [...t.messages, userMsg] } : t))
    );
    if (!overrideQuestion) setQuestion("");
    setStreamEvents([]);
    setStreamWarning(null);
    setLoading(true);

    if (isDemo) {
      const demoResponse: RAGResponse = {
        answer: `**[DEMO RESPONSE]** Simulated ${retrievalMode}/${strategy} answer for "${q}".\n\nEvoRag enforces bounded 4096-token context allocation [1] and strict citation provenance [2].`,
        status: "answered",
        admission_outcome: "admitted_demo",
        citations: [
          {
            citation_number: 1,
            evidence_id: "ev-demo-ctx",
            document_id: "doc-context",
            version: 1,
            chunk_id: "doc-context:c0",
            source_id: "context_budget.md",
            excerpt: "[DEMO] Deterministic 4096-token allocation & deduplication — segment 1.",
          },
          {
            citation_number: 2,
            evidence_id: "ev-demo-cite",
            document_id: "doc-cite",
            version: 1,
            chunk_id: "doc-cite:c0",
            source_id: "grounded_citations.md",
            excerpt: "[DEMO] Strict JSON output parsing & abstention admission — segment 1.",
          },
        ],
        limitations: ["Synthetic demo response; no backend or LLM executed."],
        retrieval_mode: retrievalMode,
        effective_retrieval_mode: retrievalMode,
        degraded: false,
        degradation_reason: null,
        trace_id: `demo-trace-${Date.now()}`,
        strategy_mode: strategy,
        metrics: { retrieval_ms: 8.2, total_ms: 14.5 },
        reflection_summary:
          reflectionMode === "off"
            ? null
            : { reflection_mode: reflectionMode, supported_claims: 2, contradicted_claims: 0 },
      };
      const assistantMsg: ConversationMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: demoResponse.answer,
        timestamp: new Date().toISOString(),
        isDemo: true,
        response: demoResponse,
      };
      setThreads((prev) =>
        prev.map((t) =>
          t.id === targetThreadId ? { ...t, messages: [...t.messages, assistantMsg] } : t
        )
      );
      setSelectedCitation(demoResponse.citations[0]);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    const localEvents: RAGStreamEvent[] = [];
    let finalResponse: RAGResponse | null = null;
    let streamError: string | null = null;
    let resourceRejected = false;

    try {
      await api.streamRagQuery(
        {
          question: q,
          retrieval_mode: retrievalMode,
          strategy,
          conversation_id: targetThreadId,
          reflection_enabled: reflectionMode !== "off",
          reflection_mode: reflectionMode,
        },
        (ev) => {
          localEvents.push(ev);
          if (activeIdRef.current === targetThreadId) {
            setStreamEvents([...localEvents]);
          }
          if (ev.response) finalResponse = ev.response;
          if (ev.error) streamError = ev.error;
        },
        controller.signal,
        (raw) => {
          if (activeIdRef.current === targetThreadId) {
            setStreamWarning(`Ignored malformed NDJSON frame: ${raw.slice(0, 80)}`);
          }
        }
      );

      if (activeIdRef.current !== targetThreadId) return;

      if (finalResponse) {
        const r: RAGResponse = finalResponse;
        const assistantMsg: ConversationMessage = {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: r.answer,
          timestamp: new Date().toISOString(),
          isDemo: false,
          response: r,
          events: localEvents,
        };
        setThreads((prev) =>
          prev.map((t) =>
            t.id === targetThreadId ? { ...t, messages: [...t.messages, assistantMsg] } : t
          )
        );
        setSelectedCitation(r.citations[0] ?? null);
        onQueryCompleted(q, r);
      } else if (streamError) {
        const assistantMsg: ConversationMessage = {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: "",
          timestamp: new Date().toISOString(),
          error: streamError,
          events: localEvents,
        };
        setThreads((prev) =>
          prev.map((t) =>
            t.id === targetThreadId ? { ...t, messages: [...t.messages, assistantMsg] } : t
          )
        );
      }
    } catch (err) {
      if (activeIdRef.current !== targetThreadId) return;
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (err instanceof ApiError && err.isResourceRejection) {
        resourceRejected = true;
      }
      const msg = isAbort
        ? "Generation cancelled by user or request deadline."
        : err instanceof Error
          ? err.message
          : "Query stream failed";
      const assistantMsg: ConversationMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        error: msg,
        resourceRejected,
        events: localEvents,
      };
      setThreads((prev) =>
        prev.map((t) =>
          t.id === targetThreadId ? { ...t, messages: [...t.messages, assistantMsg] } : t
        )
      );
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  };

  const copyText = async (msgId: string, text: string) => {
    try {
      await navigator.clipboard?.writeText(text);
      setCopiedMsgId(msgId);
      window.setTimeout(() => setCopiedMsgId(null), 1500);
    } catch {
      // Ignore clipboard permission errors in headless tests
    }
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)] min-h-[580px] flex-col lg:flex-row overflow-hidden bg-[#05070B]">
      {/* Left Panel: Conversations & Config */}
      <aside className="w-full lg:w-72 shrink-0 border-b lg:border-b-0 lg:border-r border-slate-800/80 bg-[#080C13] p-3.5 flex flex-col gap-3 overflow-y-auto">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-100">
              RAG Playground
            </h1>
            <p className="text-[10px] text-slate-500">Grounded conversational workbench</p>
          </div>
          <ProvenanceBadge kind={isDemo ? "DEMO" : backendOnline ? "LIVE" : "UNAVAILABLE"} />
        </div>

        <button
          type="button"
          onClick={createConversation}
          className="flex items-center justify-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/12 px-3 py-1.5 font-mono text-xs text-cyan-200 hover:bg-cyan-500/20"
        >
          <Plus className="h-3.5 w-3.5" />
          New Conversation
        </button>

        <div className="space-y-1 max-h-44 overflow-y-auto">
          {threads.map((t) => (
            <div
              key={t.id}
              className={`flex items-center justify-between gap-1 rounded-lg border px-2 py-1.5 text-xs ${
                t.id === activeThread?.id
                  ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-100"
                  : "border-slate-800/80 bg-[#05070B] text-slate-300"
              }`}
            >
              {renamingId === t.id ? (
                <input
                  autoFocus
                  value={renameDraft}
                  onChange={(e) => setRenameDraft(e.target.value)}
                  onBlur={() => {
                    if (renameDraft.trim()) {
                      setThreads((prev) =>
                        prev.map((item) =>
                          item.id === t.id ? { ...item, title: renameDraft.trim() } : item
                        )
                      );
                    }
                    setRenamingId(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") e.currentTarget.blur();
                  }}
                  className="w-full bg-transparent font-mono text-xs outline-none"
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setActiveId(t.id)}
                  className="flex-1 truncate text-left font-mono text-[11px]"
                >
                  {t.title}
                </button>
              )}
              <button
                type="button"
                aria-label={`Rename ${t.title}`}
                onClick={() => {
                  setRenamingId(t.id);
                  setRenameDraft(t.title);
                }}
                className="p-0.5 text-slate-500 hover:text-slate-200"
              >
                <Edit3 className="h-3 w-3" />
              </button>
              <button
                type="button"
                aria-label={`Remove ${t.title}`}
                onClick={() => removeConversation(t.id)}
                className="p-0.5 text-slate-500 hover:text-rose-300"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>

        {/* Retrieval & Strategy Controls */}
        <div className="space-y-2.5 border-t border-slate-800/80 pt-3 font-mono text-[11px]">
          <div>
            <label className="block text-[10px] uppercase text-slate-400">Retrieval Mode</label>
            <select
              aria-label="Playground retrieval mode"
              value={retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value as "hybrid" | "sparse" | "dense")}
              className="mt-1 w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 text-slate-200"
            >
              <option value="hybrid">hybrid (Dense + BM25 + RRF)</option>
              <option value="sparse">sparse (BM25)</option>
              <option value="dense">dense (Float32 Cosine)</option>
            </select>
          </div>

          <div>
            <label className="block text-[10px] uppercase text-slate-400">RAG Strategy</label>
            <select
              aria-label="Playground RAG strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as "baseline" | "adaptive")}
              className="mt-1 w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 text-slate-200"
            >
              <option value="baseline">baseline</option>
              <option value="adaptive">adaptive</option>
            </select>
          </div>

          <div>
            <label className="block text-[10px] uppercase text-slate-400">Reflection Mode</label>
            <select
              aria-label="Playground reflection mode"
              value={reflectionMode}
              onChange={(e) =>
                setReflectionMode(e.target.value as "off" | "deterministic" | "semantic")
              }
              className="mt-1 w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 text-slate-200"
            >
              <option value="off">off</option>
              <option value="deterministic">deterministic</option>
              <option value="semantic">semantic (when available)</option>
            </select>
          </div>

          <div>
            <label className="block text-[10px] uppercase text-slate-400">Document Scope</label>
            <select
              aria-label="Document scope"
              value={docScope}
              onChange={(e) => setDocScope(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-800 bg-[#05070B] px-2.5 py-1.5 text-slate-200"
            >
              <option value="all">All indexed documents ({availableDocs.length})</option>
              {availableDocs.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.source}
                </option>
              ))}
            </select>
          </div>

          <div className="rounded-lg border border-slate-800 bg-[#05070B] p-2.5 text-[10px] text-slate-400 space-y-1">
            <div>Context Budget: 4096 tokens</div>
            <div>
              Generation Model:{" "}
              {models?.generation_installed ? "Installed" : "Unavailable / Offline"}
            </div>
            <div>RSI remains isolated from single-query runs.</div>
          </div>
        </div>
      </aside>

      {/* Center Panel: Conversation Stream */}
      <main className="flex flex-1 flex-col overflow-hidden bg-[#05070B]">
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {!isDemo && !backendOnline && (
            <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 p-3 font-mono text-xs text-amber-200">
              Live Mode selected while backend on `127.0.0.1:8000` is offline. Switch to Demo Mode
              in the header to test synthetic responses, or start FastAPI for live execution.
            </div>
          )}

          {streamWarning && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 font-mono text-[11px] text-amber-200">
              {streamWarning}
            </div>
          )}

          {activeThread?.messages.length === 0 && (
            <div className="mx-auto mt-12 max-w-lg rounded-xl border border-slate-800 bg-[#090E17] p-6 text-center space-y-2">
              <h2 className="font-mono text-sm font-semibold text-slate-100">
                Grounded Research Workspace
              </h2>
              <p className="text-xs text-slate-400">
                Ask questions across ingested revisions. Answers display verified citations,
                partial-answer warnings, or abstention states.
              </p>
            </div>
          )}

          {activeThread?.messages.map((m) => (
            <div
              key={m.id}
              className={`rounded-xl border p-4 space-y-2.5 ${
                m.role === "user"
                  ? "ml-auto max-w-2xl border-cyan-500/30 bg-[#0A131E]"
                  : m.resourceRejected
                    ? "border-rose-500/45 bg-rose-500/10"
                    : "border-slate-800 bg-[#0B1018]"
              }`}
            >
              <div className="flex items-center justify-between gap-2 font-mono text-[10px] text-slate-400">
                <span>{m.role === "user" ? "RESEARCHER" : "EVORAG GROUNDED ENGINE"}</span>
                <div className="flex items-center gap-2">
                  {m.isDemo && <ProvenanceBadge kind="DEMO" />}
                  {m.role === "assistant" && m.content && (
                    <button
                      type="button"
                      onClick={() => copyText(m.id, m.content)}
                      className="inline-flex items-center gap-1 rounded border border-slate-700 px-1.5 py-0.5 text-slate-300 hover:border-cyan-500/40"
                    >
                      {copiedMsgId === m.id ? (
                        <Check className="h-3 w-3 text-emerald-400" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                      Copy
                    </button>
                  )}
                </div>
              </div>

              {m.error ? (
                <div className="space-y-2">
                  <div className="flex items-start gap-2 font-mono text-xs text-rose-300">
                    <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold">
                        {m.resourceRejected ? "Resource Admission Rejected (503)" : "Request Failed"}
                      </div>
                      <div className="mt-0.5 text-rose-200/90">{m.error}</div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      const prevUser = activeThread.messages
                        .slice(0, activeThread.messages.indexOf(m))
                        .reverse()
                        .find((x) => x.role === "user");
                      if (prevUser) void submitQuestion(prevUser.content);
                    }}
                    className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 font-mono text-[11px] text-slate-200 hover:border-cyan-500/40"
                  >
                    <RefreshCw className="h-3 w-3" />
                    Retry
                  </button>
                </div>
              ) : (
                <div className="space-y-2.5 text-sm leading-relaxed text-slate-200">
                  {parseMarkdownBlocks(m.content).map((blk, bIdx) => {
                    if (blk.kind === "code") {
                      return (
                        <pre
                          key={bIdx}
                          className="overflow-x-auto rounded-lg border border-slate-800 bg-[#05070B] p-3 font-mono text-xs text-cyan-200"
                        >
                          <code>{blk.content}</code>
                        </pre>
                      );
                    }
                    if (blk.kind === "heading") {
                      return (
                        <h3 key={bIdx} className="font-mono text-xs font-semibold text-cyan-200">
                          {blk.content}
                        </h3>
                      );
                    }
                    if (blk.kind === "list" && blk.items) {
                      return (
                        <ul key={bIdx} className="list-disc pl-5 space-y-1 text-xs">
                          {blk.items.map((it, iIdx) => (
                            <li key={iIdx}>
                              {renderInlineMarkdown(it, (cNum) => {
                                const found = m.response?.citations.find(
                                  (c) => c.citation_number === cNum
                                );
                                if (found) setSelectedCitation(found);
                              })}
                            </li>
                          ))}
                        </ul>
                      );
                    }
                    return (
                      <p key={bIdx} className="text-sm text-slate-200">
                        {renderInlineMarkdown(blk.content, (cNum) => {
                          const found = m.response?.citations.find(
                            (c) => c.citation_number === cNum
                          );
                          if (found) setSelectedCitation(found);
                        })}
                      </p>
                    );
                  })}
                </div>
              )}

              {m.response?.status === "partially_answered" && (
                <div className="flex items-center gap-2 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-1.5 font-mono text-[11px] text-amber-200">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                  Partial Answer: Some aspects of the question lack sufficient evidence.
                </div>
              )}

              {m.response?.status === "insufficient_evidence" && (
                <div className="flex items-center gap-2 rounded-lg border border-purple-500/35 bg-purple-500/10 px-3 py-1.5 font-mono text-[11px] text-purple-200">
                  <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
                  Abstained (Insufficient Evidence): Grounded threshold not met.
                </div>
              )}

              {m.response && m.response.citations.length > 0 && (
                <div className="flex flex-wrap gap-1.5 border-t border-slate-800/80 pt-2">
                  {m.response.citations.map((c) => (
                    <button
                      key={c.evidence_id}
                      type="button"
                      onClick={() => {
                        setSelectedCitation(c);
                        onInspect({
                          title: `Citation [${c.citation_number}] · ${c.chunk_id}`,
                          kind: "citation",
                          provenance: m.isDemo ? "demo" : "live",
                          summary: c.excerpt,
                          fields: {
                            document_id: c.document_id,
                            version: c.version,
                            chunk_id: c.chunk_id,
                            source_id: c.source_id,
                          },
                        });
                      }}
                      className={`rounded border px-2 py-0.5 font-mono text-[10px] transition ${
                        selectedCitation?.evidence_id === c.evidence_id
                          ? "border-cyan-400 bg-cyan-500/20 text-cyan-100"
                          : "border-slate-700 bg-[#05070B] text-slate-300 hover:border-cyan-500/40"
                      }`}
                    >
                      [{c.citation_number}] {c.source_id ?? c.document_id}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Stream Progress + Input Bar */}
        <div className="border-t border-slate-800/80 bg-[#080C13] p-3 space-y-2">
          {loading && streamEvents.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {streamEvents.map((ev, i) => (
                <span
                  key={i}
                  className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] text-cyan-200"
                >
                  {ev.event}
                </span>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submitQuestion();
                }
              }}
              aria-label="Playground question input"
              placeholder="Ask a grounded question across indexed documents..."
              className="flex-1 rounded-lg border border-slate-800 bg-[#05070B] px-3 py-2 text-xs text-slate-100 outline-none focus:border-cyan-500/60"
            />
            {loading ? (
              <button
                type="button"
                onClick={stopGeneration}
                className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/45 bg-rose-500/15 px-3.5 py-2 font-mono text-xs text-rose-200 hover:bg-rose-500/25"
              >
                <Square className="h-3.5 w-3.5" />
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void submitQuestion()}
                disabled={!question.trim() || (!isDemo && !backendOnline)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-2 font-mono text-xs text-cyan-200 hover:bg-cyan-500/25 disabled:opacity-45"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            )}
          </div>
        </div>
      </main>

      {/* Right Panel: Execution & Citation Inspector */}
      <aside className="w-full lg:w-80 shrink-0 border-t lg:border-t-0 lg:border-l border-slate-800/80 bg-[#080C13] p-3.5 space-y-4 overflow-y-auto">
        <div>
          <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-slate-100">
            Execution & Citation Inspector
          </h2>
          <p className="text-[10px] text-slate-500">
            Citation structural validity is distinct from semantic claim support.
          </p>
        </div>

        {selectedCitation ? (
          <div className="rounded-xl border border-cyan-500/35 bg-[#05080F] p-3 space-y-2.5">
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="font-semibold text-cyan-200">
                Citation [{selectedCitation.citation_number}]
              </span>
              <span className="rounded border border-emerald-500/35 bg-emerald-500/10 px-1.5 py-0.5 text-[9px] text-emerald-300">
                STRUCTURALLY VALID
              </span>
            </div>
            <div className="space-y-1 font-mono text-[11px] text-slate-300">
              <div>doc: {selectedCitation.source_id ?? selectedCitation.document_id}</div>
              <div>revision: v{selectedCitation.version ?? "—"}</div>
              <div>chunk: {selectedCitation.chunk_id}</div>
            </div>
            <p className="rounded border border-slate-800 bg-[#090E17] p-2 text-[11px] text-slate-300">
              {selectedCitation.excerpt}
            </p>
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() =>
                  onOpenDocuments({
                    documentId: selectedCitation.document_id,
                    version: selectedCitation.version,
                    chunkId: selectedCitation.chunk_id,
                  })
                }
                className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 font-mono text-[10px] text-slate-200 hover:border-cyan-500/40"
              >
                <FileText className="h-3 w-3 text-cyan-400" />
                Open Document
              </button>
              <button
                type="button"
                onClick={() =>
                  onOpenExplorer({
                    documentId: selectedCitation.document_id,
                    chunkId: selectedCitation.chunk_id,
                    nodeId: `chunk:${selectedCitation.chunk_id}`,
                  })
                }
                className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 font-mono text-[10px] text-slate-200 hover:border-purple-500/40"
              >
                <Network className="h-3 w-3 text-purple-400" />
                Open Graph Node
              </button>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-slate-800 bg-[#05070B] p-3 text-xs text-slate-400">
            Select any citation chip `[1]` in an answer to inspect chunk provenance and jump to its
            document or graph node.
          </div>
        )}

        {lastAssistantMsg?.response && (
          <div className="rounded-xl border border-slate-800 bg-[#05080F] p-3 space-y-2 font-mono text-[11px]">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">TRACE METADATA</span>
              <button
                type="button"
                onClick={() =>
                  onOpenPipeline(
                    [
                      `mode=${lastAssistantMsg.response?.effective_retrieval_mode}`,
                      `status=${lastAssistantMsg.response?.status}`,
                      `trace=${lastAssistantMsg.response?.trace_id}`,
                    ],
                    lastAssistantMsg.response?.trace_id
                  )
                }
                className="inline-flex items-center gap-1 text-[10px] text-purple-300 hover:underline"
              >
                <Workflow className="h-3 w-3" />
                Open Pipeline
              </button>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">trace_id</span>
              <span className="truncate text-cyan-300">{lastAssistantMsg.response.trace_id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">effective_mode</span>
              <span className="text-slate-200">
                {lastAssistantMsg.response.effective_retrieval_mode}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">admission</span>
              <span className="text-slate-200">{lastAssistantMsg.response.admission_outcome}</span>
            </div>
            {lastAssistantMsg.response.reflection_summary && (
              <div className="border-t border-slate-800 pt-1.5 text-[10px] text-purple-300">
                Reflection: {JSON.stringify(lastAssistantMsg.response.reflection_summary)}
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
