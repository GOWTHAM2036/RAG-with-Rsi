import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Cpu,
  FileText,
  Network,
  Search,
  Sparkles,
  Square,
  Upload,
} from "lucide-react";
import { ApiError, api } from "../api";
import {
  DEMO_CHUNKS_BY_DOC,
  DEMO_DOCUMENTS,
  DEMO_VERSIONS_BY_DOC,
} from "../data/demoCorpus";
import type {
  CrossViewNavigationTarget,
  DocumentInfo,
  EmbeddingJob,
  IngestResult,
  InspectorPayload,
  StoredChunk,
  VersionInfo,
  WorkspaceMode,
} from "../types";
import { ProvenanceBadge, SurfaceCard } from "./ui/Primitives";

const ALLOWED_EXTENSIONS = [".txt", ".md", ".markdown", ".csv", ".json"];

export function isSupportedDocumentFile(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function DocumentManagement({
  workspaceMode,
  backendOnline,
  documents,
  navigationTarget,
  onDocumentsChanged,
  onOpenExplorer,
  onInspect,
}: {
  workspaceMode: WorkspaceMode;
  backendOnline: boolean;
  documents: DocumentInfo[];
  navigationTarget: CrossViewNavigationTarget | null;
  onDocumentsChanged: () => void;
  onOpenExplorer: (target: CrossViewNavigationTarget) => void;
  onInspect: (payload: InspectorPayload) => void;
}) {
  const isDemo = workspaceMode === "demo";
  const list = isDemo ? DEMO_DOCUMENTS : documents;

  const [search, setSearch] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(
    navigationTarget?.documentId ?? list[0]?.document_id ?? null
  );
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(
    navigationTarget?.version ?? null
  );
  const [chunks, setChunks] = useState<StoredChunk[]>([]);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(
    navigationTarget?.chunkId ?? null
  );

  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<IngestResult | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [embedJob, setEmbedJob] = useState<EmbeddingJob | null>(null);
  const [embedError, setEmbedError] = useState<string | null>(null);

  useEffect(() => {
    if (navigationTarget?.documentId) {
      setSelectedDocId(navigationTarget.documentId);
      setSelectedVersion(navigationTarget.version ?? null);
      setSelectedChunkId(navigationTarget.chunkId ?? null);
    } else if (!selectedDocId && list[0]) {
      setSelectedDocId(list[0].document_id);
    }
  }, [list, navigationTarget, selectedDocId]);

  const filteredDocs = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return list;
    return list.filter(
      (d) => d.source.toLowerCase().includes(q) || d.document_id.toLowerCase().includes(q)
    );
  }, [list, search]);

  const selectedDoc = useMemo(
    () => list.find((d) => d.document_id === selectedDocId) ?? null,
    [list, selectedDocId]
  );

  useEffect(() => {
    if (!selectedDoc) {
      setVersions([]);
      setChunks([]);
      return;
    }
    if (isDemo) {
      const demoVers = DEMO_VERSIONS_BY_DOC[selectedDoc.document_id] ?? [];
      setVersions(demoVers);
      const vNum = selectedVersion ?? demoVers[0]?.version ?? 1;
      setSelectedVersion(vNum);
      const demoCh = DEMO_CHUNKS_BY_DOC[selectedDoc.document_id] ?? [];
      setChunks(demoCh);
      if (!selectedChunkId && demoCh[0]) setSelectedChunkId(demoCh[0].chunk_id);
      return;
    }

    let active = true;
    const ctrl = new AbortController();
    const loadDetails = async () => {
      try {
        const vers = await api.getDocumentVersions(selectedDoc.document_id, 20, 0, ctrl.signal);
        if (!active) return;
        setVersions(vers);
        const vNum = selectedVersion ?? selectedDoc.latest_version ?? vers[0]?.version ?? 1;
        setSelectedVersion(vNum);
        const ch = await api.getDocumentChunks(selectedDoc.document_id, vNum, 50, 0, ctrl.signal);
        if (!active) return;
        setChunks(ch);
        if (ch[0] && (!selectedChunkId || !ch.some((x) => x.chunk_id === selectedChunkId))) {
          setSelectedChunkId(ch[0].chunk_id);
        }
      } catch {
        if (!active) return;
        setVersions([]);
        setChunks([]);
      }
    };
    void loadDetails();
    return () => {
      active = false;
      ctrl.abort();
    };
  }, [isDemo, selectedDoc, selectedVersion]);

  const activeVersionInfo = useMemo(
    () => versions.find((v) => v.version === selectedVersion) ?? versions[0] ?? null,
    [selectedVersion, versions]
  );

  const selectedChunk = useMemo(
    () => chunks.find((c) => c.chunk_id === selectedChunkId) ?? chunks[0] ?? null,
    [chunks, selectedChunkId]
  );

  const handleFileUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploadError(null);
    setUploadResult(null);

    if (!isSupportedDocumentFile(file.name)) {
      setUploadError(
        `Unsupported file format "${file.name}". Verified formats: TXT, Markdown (.md), CSV, JSON.`
      );
      return;
    }
    if (file.size === 0) {
      setUploadError("File is empty (0 bytes).");
      return;
    }
    if (isDemo) {
      setUploadError("Upload disabled in Offline Demo Mode. Switch to Live Mode with FastAPI running.");
      return;
    }

    setUploading(true);
    try {
      const res = await api.uploadDocument(file);
      setUploadResult(res);
      setSelectedDocId(res.document_id);
      setSelectedVersion(res.version);
      onDocumentsChanged();
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.detail : "Document upload failed");
    } finally {
      setUploading(false);
    }
  };

  const triggerEmbed = async () => {
    if (!selectedDoc || isDemo) return;
    setEmbedError(null);
    try {
      const job = await api.embedDocument(selectedDoc.document_id, selectedVersion);
      setEmbedJob(job);
    } catch (err) {
      setEmbedError(err instanceof ApiError ? err.detail : "Failed to start embedding job");
    }
  };

  const cancelEmbed = async () => {
    if (!embedJob || isDemo) return;
    try {
      const updated = await api.cancelEmbeddingJob(embedJob.job_id);
      setEmbedJob(updated);
    } catch (err) {
      setEmbedError(err instanceof ApiError ? err.detail : "Failed to cancel embedding job");
    }
  };

  return (
    <div className="mx-auto max-w-[1440px] space-y-4 p-4 lg:p-5 animate-stagger">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-mono text-sm font-semibold tracking-wider text-slate-100">
              DOCUMENT MANAGEMENT & PROVENANCE INSPECTOR
            </h1>
            <ProvenanceBadge kind={isDemo ? "DEMO" : backendOnline ? "LIVE" : "UNAVAILABLE"} />
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            UTF-8 ingestion (TXT, Markdown, CSV, JSON), atomic revisions, chunk spans & embedding jobs.
          </p>
        </div>
      </div>

      {/* Upload Dropzone */}
      <SurfaceCard className="p-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            void handleFileUpload(e.dataTransfer.files[0]);
          }}
          className={`flex flex-wrap items-center justify-between gap-3 rounded-xl border border-dashed p-4 transition ${
            dragOver
              ? "border-cyan-400 bg-cyan-500/10"
              : "border-slate-700 bg-[#070A10]"
          }`}
        >
          <div className="flex items-center gap-3">
            <Upload className="h-5 w-5 text-cyan-400" />
            <div>
              <div className="font-mono text-xs font-semibold text-slate-100">
                Drop UTF-8 document here or select file
              </div>
              <div className="text-[11px] text-slate-400">
                Supported formats only: <code className="text-cyan-300">.txt .md .csv .json</code>{" "}
                (PDF/DOCX not supported)
              </div>
            </div>
          </div>

          <label className="cursor-pointer rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-1.5 font-mono text-xs text-cyan-200 hover:bg-cyan-500/25">
            {uploading ? "Uploading..." : "Choose File"}
            <input
              type="file"
              accept=".txt,.md,.markdown,.csv,.json"
              className="hidden"
              onChange={(e) => void handleFileUpload(e.target.files?.[0])}
            />
          </label>
        </div>

        {uploadError && (
          <div className="mt-2.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 font-mono text-xs text-rose-300">
            {uploadError}
          </div>
        )}

        {uploadResult && (
          <div className="mt-2.5 flex flex-wrap items-center gap-2 rounded-lg border border-emerald-500/35 bg-emerald-500/10 px-3 py-2 font-mono text-xs text-emerald-200">
            <CheckCircle2 className="h-4 w-4" />
            <span>
              {uploadResult.status === "duplicate"
                ? `Duplicate revision detected (v${uploadResult.version})`
                : `Ingested new revision v${uploadResult.version}`}
            </span>
            <span className="text-slate-400">
              · doc={uploadResult.document_id} · chunks={uploadResult.chunk_count}
            </span>
          </div>
        )}
      </SurfaceCard>

      {/* Main Split: Document Table + Detail & Chunk Inspector */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <SurfaceCard className="lg:col-span-5 p-4 space-y-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search ingested documents"
              placeholder="Filter documents by source or ID..."
              className="w-full rounded-lg border border-slate-800 bg-[#05070B] py-1.5 pl-8 pr-3 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500/50"
            />
          </div>

          {filteredDocs.length === 0 ? (
            <div className="rounded-lg border border-slate-800 bg-[#05070B] p-6 text-center font-mono text-xs text-slate-500">
              No documents available in current mode.
            </div>
          ) : (
            <div className="max-h-[420px] space-y-1.5 overflow-y-auto">
              {filteredDocs.map((d) => (
                <button
                  key={d.document_id}
                  type="button"
                  onClick={() => {
                    setSelectedDocId(d.document_id);
                    setSelectedVersion(d.latest_version);
                    setSelectedChunkId(null);
                  }}
                  className={`flex w-full items-center justify-between rounded-lg border p-2.5 text-left transition ${
                    selectedDoc?.document_id === d.document_id
                      ? "border-cyan-400/60 bg-cyan-500/10"
                      : "border-slate-800 bg-[#070A10] hover:border-slate-700"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5 font-mono text-xs font-semibold text-slate-100 truncate">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
                      {d.source}
                    </div>
                    <div className="mt-0.5 font-mono text-[10px] text-slate-500 truncate">
                      {d.document_id}
                    </div>
                  </div>
                  <span className="ml-2 shrink-0 rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
                    v{d.latest_version}
                  </span>
                </button>
              ))}
            </div>
          )}
        </SurfaceCard>

        {/* Right Detail Inspector */}
        <SurfaceCard accent="cyan" className="lg:col-span-7 p-4 space-y-4">
          {selectedDoc ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div>
                  <h2 className="font-mono text-sm font-semibold text-cyan-200">
                    {selectedDoc.source}
                  </h2>
                  <div className="font-mono text-[10px] text-slate-400">
                    document_id: {selectedDoc.document_id}
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={() =>
                      onOpenExplorer({
                        documentId: selectedDoc.document_id,
                        version: selectedVersion,
                        nodeId: `doc:${selectedDoc.document_id}`,
                      })
                    }
                    className="inline-flex items-center gap-1 rounded-lg border border-purple-500/40 bg-purple-500/10 px-2.5 py-1 font-mono text-[11px] text-purple-200 hover:bg-purple-500/20"
                  >
                    <Network className="h-3.5 w-3.5" />
                    Open in Knowledge Explorer
                  </button>
                  <button
                    type="button"
                    disabled={isDemo || !backendOnline}
                    onClick={() => void triggerEmbed()}
                    className="inline-flex items-center gap-1 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1 font-mono text-[11px] text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-45"
                  >
                    <Cpu className="h-3.5 w-3.5" />
                    Embed Revision
                  </button>
                </div>
              </div>

              {embedJob && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-cyan-500/35 bg-cyan-500/10 px-3 py-2 font-mono text-[11px]">
                  <span className="text-cyan-200">
                    Job {embedJob.job_id}: {embedJob.status} ({embedJob.completed}/{embedJob.total})
                  </span>
                  {embedJob.status === "running" && (
                    <button
                      type="button"
                      onClick={() => void cancelEmbed()}
                      className="inline-flex items-center gap-1 rounded border border-rose-500/40 px-2 py-0.5 text-rose-300"
                    >
                      <Square className="h-3 w-3" />
                      Cancel Job
                    </button>
                  )}
                </div>
              )}

              {embedError && (
                <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 font-mono text-xs text-rose-300">
                  {embedError}
                </div>
              )}

              {/* Revisions & Metadata */}
              {activeVersionInfo && (
                <div className="grid grid-cols-2 gap-2 font-mono text-[11px] sm:grid-cols-4">
                  <div className="rounded border border-slate-800 bg-[#06090F] p-2">
                    <div className="text-[9px] text-slate-500">REVISION</div>
                    <select
                      aria-label="Document revision selector"
                      value={selectedVersion ?? activeVersionInfo.version}
                      onChange={(e) => setSelectedVersion(Number(e.target.value))}
                      className="mt-0.5 w-full bg-transparent text-slate-100 outline-none"
                    >
                      {versions.map((v) => (
                        <option key={v.version} value={v.version} className="bg-[#0B1018]">
                          v{v.version} ({v.chunk_count} chunks)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="rounded border border-slate-800 bg-[#06090F] p-2">
                    <div className="text-[9px] text-slate-500">PARSER</div>
                    <div className="mt-0.5 truncate text-slate-100">{activeVersionInfo.parser}</div>
                  </div>
                  <div className="rounded border border-slate-800 bg-[#06090F] p-2">
                    <div className="text-[9px] text-slate-500">CONTENT HASH</div>
                    <div className="mt-0.5 truncate text-cyan-300">
                      {activeVersionInfo.content_hash}
                    </div>
                  </div>
                  <div className="rounded border border-slate-800 bg-[#06090F] p-2">
                    <div className="text-[9px] text-slate-500">CHUNKS</div>
                    <div className="mt-0.5 text-slate-100">{activeVersionInfo.chunk_count}</div>
                  </div>
                </div>
              )}

              {/* Chunk Selector + Provenance Detail */}
              <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
                <div className="md:col-span-5 max-h-64 space-y-1 overflow-y-auto rounded-lg border border-slate-800 bg-[#06090F] p-2">
                  <div className="px-1 font-mono text-[10px] uppercase text-slate-400">
                    Chunks ({chunks.length})
                  </div>
                  {chunks.map((c) => (
                    <button
                      key={c.chunk_id}
                      type="button"
                      onClick={() => {
                        setSelectedChunkId(c.chunk_id);
                        onInspect({
                          title: c.chunk_id,
                          kind: "chunk",
                          provenance: isDemo ? "demo" : "live",
                          summary: c.text.slice(0, 180),
                          fields: {
                            document_id: c.document_id,
                            start: c.start,
                            end: c.end,
                            token_count: c.token_count,
                          },
                        });
                      }}
                      className={`flex w-full items-center justify-between rounded px-2 py-1 text-left font-mono text-[11px] ${
                        selectedChunk?.chunk_id === c.chunk_id
                          ? "bg-cyan-500/15 text-cyan-200"
                          : "text-slate-300 hover:bg-slate-800/60"
                      }`}
                    >
                      <span className="truncate">{c.chunk_id}</span>
                      <span className="ml-1 shrink-0 text-[9px] text-slate-500">
                        {c.token_count}t
                      </span>
                    </button>
                  ))}
                </div>

                <div className="md:col-span-7 rounded-lg border border-slate-800 bg-[#06090F] p-3 space-y-2">
                  {selectedChunk ? (
                    <>
                      <div className="flex items-center justify-between font-mono text-[10px] text-slate-400">
                        <span className="flex items-center gap-1 text-cyan-300">
                          <Sparkles className="h-3 w-3" />
                          {selectedChunk.chunk_id}
                        </span>
                        <span>
                          span [{selectedChunk.start}, {selectedChunk.end}] ·{" "}
                          {selectedChunk.token_count} tokens
                        </span>
                      </div>
                      <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded border border-slate-800 bg-[#05070B] p-2.5 text-xs leading-relaxed text-slate-200">
                        {selectedChunk.text}
                      </pre>
                    </>
                  ) : (
                    <div className="text-xs text-slate-500">Select a chunk to inspect text & spans.</div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="p-8 text-center text-xs text-slate-400">
              Select a document to inspect revisions, chunks, and embedding status.
            </div>
          )}
        </SurfaceCard>
      </div>
    </div>
  );
}
