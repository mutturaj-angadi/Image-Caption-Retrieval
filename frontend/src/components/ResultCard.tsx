import { ImageOff } from "lucide-react";
import { useState } from "react";
import { imageSrc } from "../api";
import type { SearchResult } from "../types";

export function ScoreBar({ score }: { score: number }) {
  // The bar is only a visual aid for the cosine score (negative values are drawn as empty).
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <div className="flex items-center gap-2" title="Cosine similarity between the query and image embeddings. Not a probability.">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700" aria-hidden="true">
        <div className="h-full rounded-full bg-brand-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs tabular-nums">{score.toFixed(3)}</span>
    </div>
  );
}

export function ImageBox({ result, className = "" }: { result: SearchResult; className?: string }) {
  const [broken, setBroken] = useState(false);
  if (broken || !result.image_available) {
    return (
      <div className={`flex flex-col items-center justify-center gap-1 bg-slate-100 text-xs text-slate-500 dark:bg-slate-800 ${className}`}>
        <ImageOff className="h-6 w-6" aria-hidden="true" />
        Image file unavailable
      </div>
    );
  }
  return (
    <img
      src={imageSrc(result.image_url)}
      alt={result.captions[0] ?? `Result ${result.rank}`}
      loading="lazy"
      onError={() => setBroken(true)}
      className={`bg-slate-100 object-contain dark:bg-slate-800 ${className}`}
    />
  );
}

export function ResultCard({ result, onOpen }: { result: SearchResult; onOpen: () => void }) {
  return (
    <li className="result-card overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-[0_6px_20px_rgb(15_23_42/5%)]">
      <button
        type="button"
        onClick={onOpen}
        aria-label={`Open result ${result.rank}: ${result.captions[0] ?? result.image_id}`}
        className="relative block w-full"
      >
        <ImageBox result={result} className="aspect-[4/3] w-full" />
        <span className="absolute left-2 top-2 rounded-md bg-slate-900/80 px-2 py-0.5 text-xs font-semibold text-white">
          #{result.rank}
        </span>
      </button>
      <div className="space-y-2 p-3">
        <ScoreBar score={result.score} />
        {result.captions.length > 0 ? (
          <p className="line-clamp-2 text-sm text-slate-700 dark:text-slate-300">{result.captions[0]}</p>
        ) : (
          <p className="text-sm text-slate-500">No caption available</p>
        )}
        <p className="font-mono text-[11px] text-slate-500">id {result.image_id}</p>
      </div>
    </li>
  );
}
