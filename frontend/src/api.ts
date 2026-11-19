import type { EvaluationResponse, HealthResponse, ModelsResponse, SearchResponse, StatsResponse } from "./types";

/** Empty by default: in development Vite proxies /api to the backend. Set VITE_API_BASE to override. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError(0, "network_error", "Cannot reach the API. Is the backend running (uvicorn backend.main:app --port 8000)?");
  }
  if (!res.ok) {
    let code = "http_error";
    let message = `Request failed (HTTP ${res.status}).`;
    try {
      const body = await res.json();
      if (body && body.error) {
        code = String(body.error.code ?? code);
        message = String(body.error.message ?? message);
      }
    } catch {
      /* body was not JSON; keep the generic message */
    }
    throw new ApiError(res.status, code, message);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<ModelsResponse>("/api/models"),
  stats: () => request<StatsResponse>("/api/stats"),
  evaluation: () => request<EvaluationResponse>("/api/evaluation"),
  search: (query: string, topK: number, model?: string, signal?: AbortSignal) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK, model: model || null }),
      signal,
    }),
};

export function imageSrc(imageUrl: string): string {
  return `${API_BASE}${imageUrl}`;
}
