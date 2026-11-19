import { AlertTriangle, Info, XCircle } from "lucide-react";
import type { ReactNode } from "react";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
      <span
        className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-brand-600"
        aria-hidden="true"
      />
      {label}
    </span>
  );
}

type Tone = "error" | "warning" | "info";

const TONES: Record<Tone, { box: string; icon: ReactNode; role: "alert" | "status" }> = {
  error: {
    box: "border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100",
    icon: <XCircle className="h-5 w-5 shrink-0" aria-hidden="true" />,
    role: "alert",
  },
  warning: {
    box: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100",
    icon: <AlertTriangle className="h-5 w-5 shrink-0" aria-hidden="true" />,
    role: "status",
  },
  info: {
    box: "border-sky-300 bg-sky-50 text-sky-900 dark:border-sky-800 dark:bg-sky-950 dark:text-sky-100",
    icon: <Info className="h-5 w-5 shrink-0" aria-hidden="true" />,
    role: "status",
  },
};

export function Banner({ tone, title, children }: { tone: Tone; title?: string; children?: ReactNode }) {
  const t = TONES[tone];
  return (
    <div role={t.role} className={`flex gap-3 rounded-lg border p-4 text-sm ${t.box}`}>
      {t.icon}
      <div>
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={title ? "mt-1" : ""}>{children}</div>}
      </div>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center dark:border-slate-700">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-2 text-sm text-slate-600 dark:text-slate-400">{children}</div>}
    </div>
  );
}

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section
      className={`rounded-2xl border border-slate-200/80 bg-white/90 p-5 shadow-[0_8px_28px_rgb(15_23_42/4%)] backdrop-blur-sm sm:p-6 ${className}`}
    >
      {title && <h2 className="mb-3 text-base font-semibold">{title}</h2>}
      {children}
    </section>
  );
}
