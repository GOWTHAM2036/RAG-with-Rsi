export type NodeKind =
  | "document"
  | "revision"
  | "chunk"
  | "embedding"
  | "config_version"
  | "query";

export type EdgeKind =
  | "document_contains_chunk"
  | "document_has_revision"
  | "chunk_has_embedding"
  | "candidate_derives_from_parent"
  | "query_retrieved_chunk"
  | "answer_cites_chunk";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  cluster: string;
  metadata: Record<string, string | number | boolean | null>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
}

export interface ProvenanceGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
  is_demo: boolean;
}

export interface SourceSpan {
  label: string;
  start: number;
  end: number;
  parsed_start: number | null;
  parsed_end: number | null;
}

export interface Chunk {
  chunk_id: string;
  document_id: string;
  start: number;
  end: number;
  text: string;
}

export interface StoredChunk extends Chunk {
  token_count: number;
  estimated: boolean;
  sources: SourceSpan[];
}

export interface DocumentInfo {
  document_id: string;
  source: string;
  latest_version: number;
}

export interface VersionInfo {
  document_id: string;
  version: number;
  content_hash: string;
  fingerprint: string;
  parser: string;
  normalization: string;
  chunk_count: number;
  created_at: string;
}

export interface IngestResult {
  job_id: string;
  document_id: string;
  version: number;
  chunk_count: number;
  status: "created" | "duplicate";
}

export interface EmbeddingSpace {
  endpoint: string;
  provider: string;
  model: string;
  digest: string;
  dimensions: number;
  normalized: boolean;
  formatting_version: string;
}

export interface EmbeddingJob {
  job_id: string;
  document_id: string;
  version: number;
  total: number;
  completed: number;
  status: string;
  space: EmbeddingSpace | null;
  error: string | null;
}

export interface SparseIndexStatus {
  doc_count: number;
  chunk_count: number;
  term_count: number;
  total_length: number;
  avg_chunk_length: number;
  index_generation: number;
  analyzer_version: string;
}

export interface SparseIndexJob {
  job_id: string;
  status: string;
  total_chunks: number;
  indexed_chunks: number;
  error: string | null;
}

export interface ModelInfo {
  name: string;
  digest: string;
  size_bytes: number;
  vram_bytes: number | null;
  quantization: string | null;
  parameters: string | null;
  context_length: number | null;
}

export interface ModelAvailability {
  inventory: ModelInfo[];
  generation_installed: boolean;
  embedding_installed: boolean;
}

export interface RetrieveRequest {
  query: string;
  mode: "dense" | "sparse" | "hybrid";
  top_k: number;
  metric?: "cosine" | "dot" | "euclidean";
  document_id?: string | null;
  version?: number | null;
  latest_only?: boolean;
  reranking_enabled?: boolean;
  include_explanations?: boolean;
  dense_weight_override?: number | null;
  sparse_weight_override?: number | null;
}

export interface RetrievalResult {
  chunk: Chunk;
  rank: number;
  score: number;
}

export interface RetrievalExplanation {
  chunk_id: string;
  document_id: string;
  version: number | null;
  sources: SourceSpan[];
  dense_score: number | null;
  dense_rank: number | null;
  bm25_score: number | null;
  sparse_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  final_rank: number;
  matched_terms: string[];
  retrieval_mode: string;
  effective_mode: string;
  degraded: boolean;
  degradation_reason: string | null;
}

export interface RetrieveResponse {
  query_id: string;
  mode: string;
  effective_mode: string;
  degraded: boolean;
  degradation_reason: string | null;
  reranking_applied: boolean;
  model: string | null;
  digest: string | null;
  metric: string | null;
  results: RetrievalResult[];
  explanations: RetrievalExplanation[];
}

export interface CitationReference {
  citation_number: number;
  evidence_id: string;
  document_id: string;
  version: number | null;
  chunk_id: string;
  source_id: string | null;
  excerpt: string;
}

export interface RAGRequest {
  question: string;
  retrieval_mode: "hybrid" | "sparse" | "dense";
  top_k?: number;
  reranking_enabled?: boolean;
  conversation_id?: string | null;
  include_citations?: boolean;
  include_metrics?: boolean;
  strategy: "baseline" | "adaptive";
  query_rewriting_enabled?: boolean;
  query_decomposition_enabled?: boolean;
  include_query_plan?: boolean;
  include_retrieval_diagnostics?: boolean;
  reflection_enabled: boolean;
  reflection_mode?: "off" | "deterministic" | "semantic";
  include_reflection_summary?: boolean;
}

export interface RAGResponse {
  answer: string;
  status: "answered" | "partially_answered" | "insufficient_evidence";
  admission_outcome: string;
  citations: CitationReference[];
  limitations: string[];
  retrieval_mode: string;
  effective_retrieval_mode: string;
  degraded: boolean;
  degradation_reason: string | null;
  trace_id: string;
  metrics?: Record<string, unknown> | null;
  strategy_mode?: string | null;
  query_plan?: Record<string, unknown> | null;
  retrieval_diagnostics?: Record<string, unknown> | null;
  reflection_summary?: Record<string, unknown> | null;
}

export interface RAGStreamEvent {
  event: string;
  trace_id?: string | null;
  detail?: string | null;
  response?: RAGResponse | null;
  error?: string | null;
}

export interface RAGExecutionTrace {
  trace_id: string;
  request_id: string;
  conversation_id: string | null;
  pipeline_version: string;
  prompt_template_version: string;
  retrieval_mode: string;
  effective_retrieval_mode: string;
  embedding_model: string | null;
  generation_model: string | null;
  model_digest: string | null;
  selected_evidence_ids: string[];
  stage_metrics: Record<string, number>;
  estimated_input_tokens: number;
  provider_usage: Record<string, unknown> | null;
  citation_validation: Record<string, unknown> | null;
  admission_outcome: string;
  degraded: boolean;
  degradation_reason: string | null;
  resource_rejection_reason: string | null;
  query_execution: Record<string, unknown> | null;
  reflection_execution: Record<string, unknown> | null;
  created_at: string;
}

export interface ConfigurationVersion {
  version_id: string;
  parent_version_id: string | null;
  fingerprint: string;
  parameters: Record<string, unknown>;
  active: boolean;
  approved: boolean;
  created_at: string;
}

export interface PromotionRecommendation {
  recommendation_id: string;
  baseline_fingerprint: string;
  candidate_fingerprint: string;
  changes: Record<string, unknown>;
  expected_benefit: string;
  execution_mode: string;
  eligible_for_live_promotion: boolean;
  rollback_target_version: string;
  known_limitations: string;
}

export type WorkspaceMode = "live" | "demo";

export type SignatureTabId = "command" | "explorer" | "pipeline" | "rsi";

export type FunctionalWorkspaceTabId = "playground" | "documents" | "retrieval";

export type PlannedTabId =
  | "evaluation"
  | "reflection"
  | "experiments"
  | "models"
  | "monitor"
  | "traces"
  | "settings";

export type NavTabId = SignatureTabId | FunctionalWorkspaceTabId | PlannedTabId;

export interface AppearanceSettings {
  reducedMotion: boolean;
  lowPerfMode: boolean;
  progressiveLabels: boolean;
  ambientGlow: boolean;
}

export interface NotificationItem {
  id: string;
  title: string;
  detail: string;
  level: "info" | "warning" | "error";
  timestamp: string;
  source: "live" | "demo" | "system";
}

export interface InspectorPayload {
  title: string;
  kind: string;
  provenance: "live" | "demo" | "static";
  summary: string;
  fields: Record<string, string | number | boolean | null>;
}

export type StageStatus = "completed" | "active" | "skipped" | "error" | "idle";

export interface PipelineStageNode {
  id: string;
  label: string;
  subtitle: string;
  status: StageStatus;
  latencyMs: number | null;
  provenance: "static" | "recorded" | "demo";
  x: number;
  y: number;
  details: string;
  metrics: Record<string, string | number>;
}

export interface RSICandidateRecord {
  versionId: string;
  parentVersionId: string | null;
  fingerprint: string;
  outcome: "baseline" | "rejected" | "inconclusive" | "experimental_incumbent";
  approved: boolean;
  active: boolean;
  isDemo: boolean;
  changedParameters: Record<
    string,
    { before: string | number | boolean; after: string | number | boolean }
  >;
  qualityDelta: string;
  ci95: string;
  latencyDeltaMs: string;
  ramDeltaMiB: string;
  notes: string;
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  isDemo?: boolean;
  error?: string | null;
  resourceRejected?: boolean;
  response?: RAGResponse | null;
  events?: RAGStreamEvent[];
}

export interface ConversationThread {
  id: string;
  title: string;
  createdAt: string;
  isDemo?: boolean;
  messages: ConversationMessage[];
}

export interface CrossViewNavigationTarget {
  documentId?: string;
  version?: number | null;
  chunkId?: string;
  traceId?: string;
  nodeId?: string;
}
