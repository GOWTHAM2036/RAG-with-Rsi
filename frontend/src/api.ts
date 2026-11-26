import { DEMO_PROVENANCE_GRAPH } from "./data/demoCorpus";
import type {
  ConfigurationVersion,
  DocumentInfo,
  EmbeddingJob,
  IngestResult,
  ModelAvailability,
  PromotionRecommendation,
  ProvenanceGraphResponse,
  RAGExecutionTrace,
  RAGRequest,
  RAGResponse,
  RAGStreamEvent,
  RetrieveRequest,
  RetrieveResponse,
  SparseIndexJob,
  SparseIndexStatus,
  StoredChunk,
  VersionInfo,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly isResourceRejection: boolean;

  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.isResourceRejection = status === 503;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text();
    if (!text) return `HTTP ${res.status}`;
    const parsed = JSON.parse(text) as Record<string, unknown>;
    if (typeof parsed.detail === "string") return parsed.detail;
    if (typeof parsed.error === "string") return parsed.error;
    return text.slice(0, 240);
  } catch {
    return `HTTP ${res.status}`;
  }
}

async function requestJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs = 5000
): Promise<T> {
  const controller = new AbortController();
  const externalSignal = init?.signal;
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    else externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) {
      throw new ApiError(res.status, await parseErrorDetail(res));
    }
    return (await res.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function consumeNdjsonStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (ev: RAGStreamEvent) => void,
  onMalformedLine?: (rawLine: string) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let buffer = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      const parsed = JSON.parse(trimmed) as RAGStreamEvent;
      if (parsed && typeof parsed.event === "string") {
        onEvent(parsed);
      } else {
        onMalformedLine?.(trimmed);
      }
    } catch {
      onMalformedLine?.(trimmed);
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        processLine(line);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      processLine(buffer);
    }
  } finally {
    reader.releaseLock();
  }
}

export const api = {
  health: (signal?: AbortSignal) =>
    requestJson<{ status: string }>("/health", { signal }, 1500),
  ready: (signal?: AbortSignal) =>
    requestJson<Record<string, unknown>>("/ready?inference=true", { signal }, 2500),
  modelsStatus: (signal?: AbortSignal) =>
    requestJson<ModelAvailability>("/api/v1/models/status", { signal }, 2500),
  ragStatus: (signal?: AbortSignal) =>
    requestJson<Record<string, unknown>>("/api/v1/rag/status", { signal }),
  reflectionStatus: (signal?: AbortSignal) =>
    requestJson<Record<string, unknown>>("/api/v1/rag/reflection/status", { signal }),
  rsiStatus: (signal?: AbortSignal) =>
    requestJson<Record<string, unknown>>("/api/v1/rsi/status", { signal }),
  rsiConfigurations: (signal?: AbortSignal) =>
    requestJson<ConfigurationVersion[]>("/api/v1/rsi/configurations", { signal }),
  rsiRecommendations: (signal?: AbortSignal) =>
    requestJson<PromotionRecommendation[]>("/api/v1/rsi/recommendations", { signal }),
  provenanceGraph: (maxNodes = 250, signal?: AbortSignal) =>
    requestJson<ProvenanceGraphResponse>(
      `/api/v1/graph/provenance?max_nodes=${maxNodes}`,
      { signal }
    ),

  // Documents & Ingestion
  listDocuments: (limit = 50, offset = 0, signal?: AbortSignal) =>
    requestJson<DocumentInfo[]>(
      `/api/v1/documents?limit=${limit}&offset=${offset}`,
      { signal }
    ),
  getDocument: (documentId: string, signal?: AbortSignal) =>
    requestJson<DocumentInfo>(
      `/api/v1/documents/${encodeURIComponent(documentId)}`,
      { signal }
    ),
  getDocumentVersions: (documentId: string, limit = 20, offset = 0, signal?: AbortSignal) =>
    requestJson<VersionInfo[]>(
      `/api/v1/documents/${encodeURIComponent(documentId)}/versions?limit=${limit}&offset=${offset}`,
      { signal }
    ),
  getDocumentChunks: (
    documentId: string,
    version?: number | null,
    limit = 50,
    offset = 0,
    signal?: AbortSignal
  ) => {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    });
    if (version != null) params.set("version", String(version));
    return requestJson<StoredChunk[]>(
      `/api/v1/documents/${encodeURIComponent(documentId)}/chunks?${params.toString()}`,
      { signal }
    );
  },
  uploadDocument: async (file: File, signal?: AbortSignal): Promise<IngestResult> => {
    const buffer = await file.arrayBuffer();
    return requestJson<IngestResult>(
      "/api/v1/documents",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Filename": file.name,
        },
        body: buffer,
        signal,
      },
      20000
    );
  },

  // Embeddings & Index
  embedDocument: (documentId: string, version?: number | null, signal?: AbortSignal) => {
    const q = version != null ? `?version=${version}` : "";
    return requestJson<EmbeddingJob>(
      `/api/v1/documents/${encodeURIComponent(documentId)}/embed${q}`,
      { method: "POST", signal },
      10000
    );
  },
  getEmbeddingJob: (jobId: string, signal?: AbortSignal) =>
    requestJson<EmbeddingJob>(
      `/api/v1/embedding-jobs/${encodeURIComponent(jobId)}`,
      { signal }
    ),
  cancelEmbeddingJob: (jobId: string, signal?: AbortSignal) =>
    requestJson<EmbeddingJob>(
      `/api/v1/embedding-jobs/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST", signal }
    ),
  indexStatus: (signal?: AbortSignal) =>
    requestJson<SparseIndexStatus>("/api/v1/index/status", { signal }),
  rebuildIndex: (signal?: AbortSignal) =>
    requestJson<SparseIndexJob>("/api/v1/index/rebuild", { method: "POST", signal }),

  // Retrieval & RAG
  retrieve: (payload: RetrieveRequest, signal?: AbortSignal) =>
    requestJson<RetrieveResponse>(
      "/api/v1/retrieve",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      },
      15000
    ),
  ragPlan: (
    payload: { question: string; strategy: "baseline" | "adaptive" },
    signal?: AbortSignal
  ) =>
    requestJson<Record<string, unknown>>("/api/v1/rag/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }),
  ragQuery: (payload: RAGRequest, signal?: AbortSignal) =>
    requestJson<RAGResponse>(
      "/api/v1/rag/query",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      },
      45000
    ),
  getTrace: (traceId: string, signal?: AbortSignal) =>
    requestJson<RAGExecutionTrace>(
      `/api/v1/rag/traces/${encodeURIComponent(traceId)}`,
      { signal }
    ),

  async streamRagQuery(
    body: RAGRequest,
    onEvent: (ev: RAGStreamEvent) => void,
    signal?: AbortSignal,
    onMalformedLine?: (rawLine: string) => void
  ): Promise<void> {
    const controller = new AbortController();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
    const deadlineTimer = window.setTimeout(() => controller.abort(), 45000);
    try {
      const res = await fetch("/api/v1/rag/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...body,
          include_citations: body.include_citations ?? true,
          include_metrics: body.include_metrics ?? true,
          include_query_plan: body.include_query_plan ?? body.strategy === "adaptive",
          include_retrieval_diagnostics: body.include_retrieval_diagnostics ?? true,
          include_reflection_summary: body.include_reflection_summary ?? body.reflection_enabled,
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new ApiError(res.status, await parseErrorDetail(res));
      }
      await consumeNdjsonStream(res.body, onEvent, onMalformedLine);
    } finally {
      window.clearTimeout(deadlineTimer);
    }
  },
};

export { DEMO_PROVENANCE_GRAPH };
