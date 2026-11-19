// Mirrors backend/schemas.py. Keep both in sync when the API contract changes.

export interface SearchResult {
  rank: number;
  image_id: string;
  score: number;
  image_url: string;
  file_name: string | null;
  captions: string[];
  split: string | null;
  image_available: boolean;
}

export interface SearchResponse {
  query: string;
  model: string;
  top_k: number;
  num_results: number;
  unknown_tokens: string[];
  score_type: string;
  score_note: string;
  latency_ms: number;
  results: SearchResult[];
}

export interface MetricSet {
  [metric: string]: number;
}

export interface ModelMetadata {
  name?: string;
  architecture?: Record<string, unknown>;
  training_config?: Record<string, unknown>;
  word_vectors?: Record<string, unknown>;
  dataset?: { name?: string; synthetic?: boolean; is_subset?: boolean; split_source?: string; num_images?: Record<string, number> };
  checkpoint?: { file?: string; version?: string; sha256?: string };
  training?: Record<string, unknown>;
  evaluation?: Record<string, { metrics: MetricSet; num_images?: number; report_file?: string }>;
  status?: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  embedding_dim: number | null;
  available: boolean;
  unavailable_reason: string | null;
  num_images: number;
  index_backend: string | null;
  metadata: ModelMetadata;
}

export interface ModelsResponse {
  default_model: string | null;
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  models_total: number;
  models_available: number;
  default_model: string | null;
  warnings: string[];
}

export interface SplitStats {
  num_images: number;
  num_captions: number;
  caption_len_mean?: number;
  caption_len_median?: number;
  caption_len_p95?: number;
  caption_len_max?: number;
  oov_rate?: number;
}

export interface StatsResponse {
  available: boolean;
  message: string | null;
  dataset: { name?: string; split_source?: string; is_subset?: boolean; synthetic?: boolean; vocab_size?: number; captions_per_image?: number };
  stats: { splits?: Record<string, SplitStats>; total_images?: number; total_captions?: number; vocab_size?: number };
  indexed_images: Record<string, number>;
  synthetic: boolean;
}

export interface EvalResult {
  name: string;
  model_type: string;
  split: string;
  protocol: string;
  metrics: MetricSet;
  num_images: number;
  num_captions: number;
  embedding_dim: number | null;
  latency_ms_median: number | null;
  latency_setup: string | null;
  training_seconds: number | null;
  dataset: string | null;
  is_subset: boolean;
  synthetic: boolean;
  file: string;
}

export interface HistoryEntry {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
}

export interface EvaluationResponse {
  available: boolean;
  message: string | null;
  results: EvalResult[];
  histories: Record<string, HistoryEntry[]>;
  reference: { source?: string; model?: string; "t2i_R@1"?: number; "t2i_R@10"?: number; note?: string };
  synthetic_warning: boolean;
}
