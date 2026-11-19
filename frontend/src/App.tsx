import { useEffect, useState } from "react";
import { AboutPage } from "./pages/AboutPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { ModelsPage } from "./pages/ModelsPage";
import { SearchPage } from "./pages/SearchPage";

const PAGES = [
  { id: "search", label: "Search" },
  { id: "models", label: "Models" },
  { id: "evaluation", label: "Evaluation" },
  { id: "about", label: "About" },
] as const;

type PageId = (typeof PAGES)[number]["id"];

function currentPage(): PageId {
  const h = window.location.hash.replace("#/", "").replace("#", "");
  return (PAGES.find((p) => p.id === h)?.id ?? "search") as PageId;
}

export default function App() {
  const [page, setPage] = useState<PageId>(currentPage());

  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="min-h-screen">
      <header className="site-header sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3.5">
          <a href="#/search" className="flex items-center gap-3 text-lg font-bold tracking-tight text-slate-950">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand-600 to-brand-900 text-white shadow-lg shadow-brand-900/20" aria-hidden="true">
              N
            </span>
            <span>NeuralSeek</span>
            <span className="hidden border-l border-slate-200 pl-3 text-sm font-medium tracking-normal text-slate-500 sm:inline">Semantic Image Retrieval</span>
          </a>
          <nav aria-label="Main">
            <ul className="flex gap-1 rounded-xl border border-slate-200/80 bg-slate-50/80 p-1">
              {PAGES.map((p) => (
                <li key={p.id}>
                  <a
                    href={`#/${p.id}`}
                    aria-current={page === p.id ? "page" : undefined}
                    className={`rounded-lg px-3.5 py-2 text-sm font-semibold transition-colors ${
                      page === p.id
                        ? "bg-slate-900 text-white shadow-sm"
                        : "text-slate-600 hover:bg-white hover:text-slate-950"
                    }`}
                  >
                    {p.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8 sm:py-10">
        {page === "search" && <SearchPage />}
        {page === "models" && <ModelsPage />}
        {page === "evaluation" && <EvaluationPage />}
        {page === "about" && <AboutPage />}
      </main>
      <footer className="mx-auto mt-8 max-w-7xl border-t border-slate-200/80 px-5 py-6 text-xs text-slate-500">
        Scores are cosine similarities used for ranking, not probabilities.
      </footer>
    </div>
  );
}
