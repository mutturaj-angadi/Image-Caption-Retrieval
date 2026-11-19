import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useRef } from "react";
import type { SearchResult } from "../types";
import { ImageBox, ScoreBar } from "./ResultCard";

interface Props {
  result: SearchResult;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}

export function Lightbox({ result, onClose, onPrev, onNext }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && onPrev) onPrev();
      if (e.key === "ArrowRight" && onNext) onNext();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previouslyFocused?.focus?.();
    };
  }, [onClose, onPrev, onNext]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Result ${result.rank} details`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-full w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-white shadow-xl dark:bg-slate-900 md:flex-row"
        onClick={(e) => e.stopPropagation()}
      >
        <ImageBox result={result} className="max-h-[50vh] w-full md:max-h-[80vh] md:w-3/5" />
        <div className="flex-1 space-y-4 overflow-y-auto p-5">
          <div className="flex items-start justify-between gap-3">
            <h2 className="text-lg font-semibold">Result #{result.rank}</h2>
            <button
              ref={closeRef}
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="rounded-md p-1 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <X className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
          <ScoreBar score={result.score} />
          <div>
            <h3 className="text-sm font-semibold">Human-written captions</h3>
            {result.captions.length ? (
              <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
                {result.captions.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-sm text-slate-500">No captions available.</p>
            )}
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-slate-600 dark:text-slate-400">
            <dt>Image id</dt>
            <dd className="font-mono">{result.image_id}</dd>
            <dt>File</dt>
            <dd className="break-all font-mono">{result.file_name ?? "unknown"}</dd>
            <dt>Dataset split</dt>
            <dd>{result.split ?? "unknown"}</dd>
          </dl>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={!onPrev}
              onClick={onPrev}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm disabled:opacity-40 dark:border-slate-700"
            >
              <ChevronLeft className="h-4 w-4" aria-hidden="true" /> Previous
            </button>
            <button
              type="button"
              disabled={!onNext}
              onClick={onNext}
              className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm disabled:opacity-40 dark:border-slate-700"
            >
              Next <ChevronRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
