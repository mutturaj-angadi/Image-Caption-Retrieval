import { Search } from "lucide-react";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import { Banner, EmptyState, Spinner } from "../components/Feedback";
import { Lightbox } from "../components/Lightbox";
import { ResultCard } from "../components/ResultCard";
import { useAsync } from "../hooks";
import type { SearchResponse } from "../types";

const COCO_EXAMPLES = [
  "A dog running through snow",
  "A person riding a bicycle beside a train",
  "Three teddy bears lying together on a bed",
  "A man standing beside a red car",
  "A group of people playing football in a park",
];
const SYNTHETIC_EXAMPLES = ["a red circle on a white background", "a small blue square", "a large yellow triangle at the top"];
const TOP_K_OPTIONS = [5, 10, 20, 40, 80, 100];

export function SearchPage() {
  const models = useAsync(() => api.models());
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const requestId = useRef(0);
  const controller = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      requestId.current += 1;
      controller.current?.abort();
    },
    [],
  );

  const available = models.data?.models.filter((m) => m.available) ?? [];
  const synthetic = available.some((m) => m.metadata.dataset?.synthetic);
  const examples = synthetic ? SYNTHETIC_EXAMPLES : COCO_EXAMPLES;
  const activeModel = model || models.data?.default_model || "";

  const run = useCallback(async (q: string, k: number, m: string) => {
    const id = ++requestId.current;
    controller.current?.abort();
    const activeController = new AbortController();
    controller.current = activeController;
    setLoading(true);
    setError(null);
    try {
      const res = await api.search(q, k, m || undefined, activeController.signal);
      if (id === requestId.current) setResponse(res);
    } catch (e) {
      if (id !== requestId.current) return;
      if (e instanceof Error && e.name === "AbortError") return;
      setResponse(null);
      setError(e instanceof ApiError ? e.message : "Something went wrong while searching.");
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, []);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const q = query.trim();
    if (!q) {
      setValidation("Please describe the image you are looking for.");
      return;
    }
    setValidation(null);
    void run(q, topK, model);
  };

  const useExample = (text: string) => {
    setQuery(text);
    setValidation(null);
    void run(text, topK, model);
  };

  const canLoadMore = !!response && response.num_results >= response.top_k && response.top_k < 100;
  const loadMore = () => {
    if (!response) return;
    const next = Math.min(100, response.top_k * 2);
    setTopK(next);
    void run(response.query, next, model);
  };

  const results = response?.results ?? [];
  const open = openIndex !== null ? results[openIndex] : undefined;

  return (
    <div className="space-y-6">
      <section className="search-hero overflow-hidden rounded-[28px] border border-white/20 px-6 py-10 text-white sm:px-10 sm:py-12">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Find images by describing them</h1>
        <p className="mt-3 max-w-2xl text-base leading-7 text-teal-50/90">
          Your sentence and every indexed image are mapped into one shared embedding space; the closest images win.
        </p>
        <form onSubmit={submit} className="mt-6 space-y-3" role="search">
          <div className="flex flex-col gap-3 sm:flex-row">
            <label htmlFor="query" className="sr-only">
              Describe the image
            </label>
            <input
              id="query"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. a dog running through snow"
              maxLength={300}
              autoComplete="off"
              aria-invalid={validation ? true : undefined}
              aria-describedby={validation ? "query-error" : undefined}
              className="w-full rounded-xl border border-white/30 bg-white px-4 py-3.5 text-base text-slate-900 shadow-lg shadow-slate-950/10 placeholder:text-slate-400 focus-visible:outline-white"
            />
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-white px-6 py-3.5 font-semibold text-brand-900 shadow-md transition hover:-translate-y-0.5 hover:bg-brand-50 disabled:opacity-60"
            >
              <Search className="h-5 w-5" aria-hidden="true" />
              Search
            </button>
          </div>
          {validation && (
            <p id="query-error" role="alert" className="text-sm font-medium text-amber-200">
              {validation}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
            <label className="flex items-center gap-2">
              Results
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="rounded-lg border border-white/30 bg-white px-2.5 py-1.5 text-slate-900 shadow-sm"
                aria-label="Number of results"
              >
                {TOP_K_OPTIONS.map((k) => (
                  <option key={k} value={k}>
                    Top {k}
                  </option>
                ))}
              </select>
            </label>
            {available.length > 1 && (
              <label className="flex items-center gap-2">
                Model
                <select
                  value={activeModel}
                  onChange={(e) => setModel(e.target.value)}
                  className="rounded-lg border border-white/30 bg-white px-2.5 py-1.5 text-slate-900 shadow-sm"
                  aria-label="Retrieval model"
                >
                  {available.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name} ({m.type})
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <div className="flex flex-wrap gap-2" aria-label="Example queries">
            {examples.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => useExample(ex)}
                className="rounded-full border border-white/35 bg-white/10 px-3.5 py-1.5 text-sm text-white/95 backdrop-blur transition hover:bg-white/20"
              >
                {ex}
              </button>
            ))}
          </div>
        </form>
      </section>

      {models.error && (
        <Banner tone="error" title="Cannot load models">
          {models.error}
        </Banner>
      )}
      {models.data && available.length === 0 && (
        <Banner tone="warning" title="No model is available yet">
          Train a model and build its index first (see the README), or run <code>python scripts/smoke_test.py</code> for a synthetic demo.
          {models.data.models.map((m) => (
            <p key={m.name} className="mt-1">
              <strong>{m.name}:</strong> {m.unavailable_reason ?? "unknown reason"}
            </p>
          ))}
        </Banner>
      )}
      {synthetic && (
        <Banner tone="warning" title="Demo workspace: synthetic data">
          These results come from procedurally generated shapes used for smoke tests, not from MS-COCO. They demonstrate the pipeline only.
        </Banner>
      )}

      <div aria-live="polite" className="min-h-[2rem]">
        {loading && <Spinner label="Searching…" />}
        {error && !loading && (
          <Banner tone="error" title="Search failed">
            {error}
          </Banner>
        )}
      </div>

      {response && !error && (
        <section aria-label="Search results" className="space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-xl font-semibold tracking-tight text-slate-900">
              {response.num_results} results for “{response.query}”
            </h2>
            <p className="text-sm text-slate-500">
              Model <strong>{response.model}</strong> · ranked by cosine similarity · {response.latency_ms.toFixed(1)} ms
            </p>
          </div>
          {response.unknown_tokens.length > 0 && (
            <Banner tone="info">
              Words the model does not know were ignored: <strong>{response.unknown_tokens.join(", ")}</strong>
            </Banner>
          )}
          {results.length === 0 ? (
            <EmptyState title="No images matched">Try a different description.</EmptyState>
          ) : (
            <ol className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {results.map((r, i) => (
                <ResultCard key={r.image_id} result={r} onOpen={() => setOpenIndex(i)} />
              ))}
            </ol>
          )}
          <p className="text-xs text-slate-500">{response.score_note}</p>
          {canLoadMore && (
            <button
              type="button"
              onClick={loadMore}
              disabled={loading}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium hover:bg-slate-100 disabled:opacity-60 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Load more results
            </button>
          )}
        </section>
      )}

      {!response && !loading && !error && available.length > 0 && (
        <EmptyState title="Type a description to search">Or pick one of the example queries above.</EmptyState>
      )}

      {open && (
        <Lightbox
          result={open}
          onClose={() => setOpenIndex(null)}
          onPrev={openIndex! > 0 ? () => setOpenIndex(openIndex! - 1) : undefined}
          onNext={openIndex! < results.length - 1 ? () => setOpenIndex(openIndex! + 1) : undefined}
        />
      )}
    </div>
  );
}
