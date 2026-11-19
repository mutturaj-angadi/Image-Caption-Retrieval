// Small dependency-free SVG charts. Colours come from the Okabe-Ito colour-blind-safe palette.
const PALETTE = ["#147779", "#D89432", "#3974B8", "#718096", "#BA6B72", "#4F9A83"];

export interface BarSeries {
  label: string;
  values: number[];
}

export function Legend({ labels }: { labels: string[] }) {
  return (
    <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600 dark:text-slate-300">
      {labels.map((l, i) => (
        <li key={l} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: PALETTE[i % PALETTE.length] }} />
          {l}
        </li>
      ))}
    </ul>
  );
}

export function GroupedBarChart({
  title,
  categories,
  series,
  max = 100,
  unit = "%",
}: {
  title: string;
  categories: string[];
  series: BarSeries[];
  max?: number;
  unit?: string;
}) {
  const W = 560;
  const H = 260;
  const left = 44;
  const right = 12;
  const top = 16;
  const bottom = 36;
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const groupW = plotW / Math.max(categories.length, 1);
  const barW = Math.min(36, (groupW * 0.8) / Math.max(series.length, 1));
  const y = (v: number) => top + plotH - (Math.max(0, Math.min(v, max)) / max) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
        <title>{title}</title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={W - right} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={0.12} />
            <text x={left - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.7}>
              {Math.round(t)}
            </text>
          </g>
        ))}
        {categories.map((cat, ci) => {
          const gx = left + ci * groupW + groupW / 2 - (barW * series.length) / 2;
          return (
            <g key={cat}>
              {series.map((s, si) => {
                const v = s.values[ci];
                if (v === undefined || Number.isNaN(v)) return null;
                return (
                  <g key={s.label}>
                    <rect x={gx + si * barW} y={y(v)} width={barW - 3} height={top + plotH - y(v)} rx={2} fill={PALETTE[si % PALETTE.length]}>
                      <title>{`${s.label}, ${cat}: ${v.toFixed(1)}${unit}`}</title>
                    </rect>
                    <text x={gx + si * barW + (barW - 3) / 2} y={y(v) - 3} textAnchor="middle" fontSize={9} fill="currentColor">
                      {v.toFixed(1)}
                    </text>
                  </g>
                );
              })}
              <text x={left + ci * groupW + groupW / 2} y={H - 14} textAnchor="middle" fontSize={11} fill="currentColor">
                {cat}
              </text>
            </g>
          );
        })}
      </svg>
      <Legend labels={series.map((s) => s.label)} />
    </figure>
  );
}

export interface LineSeries {
  label: string;
  values: Array<number | null>;
}

export function LineChart({ title, xs, series, yLabel }: { title: string; xs: number[]; series: LineSeries[]; yLabel: string }) {
  const W = 560;
  const H = 260;
  const left = 48;
  const right = 12;
  const top = 16;
  const bottom = 36;
  const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null && Number.isFinite(v));
  if (xs.length === 0 || all.length === 0) {
    return <p className="text-sm text-slate-500">No curve data available.</p>;
  }
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = hi === lo ? 1 : (hi - lo) * 0.08;
  const yMin = lo - pad;
  const yMax = hi + pad;
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const px = (x: number) => left + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - left - right);
  const py = (v: number) => top + (1 - (v - yMin) / (yMax - yMin)) * (H - top - bottom);
  const ticks = [0, 0.5, 1].map((f) => yMin + f * (yMax - yMin));
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
        <title>{title}</title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={W - right} y1={py(t)} y2={py(t)} stroke="currentColor" strokeOpacity={0.12} />
            <text x={left - 6} y={py(t) + 4} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.7}>
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {series.map((s, si) => {
          const pts = s.values
            .map((v, i) => (v === null || !Number.isFinite(v) ? null : `${px(xs[i])},${py(v)}`))
            .filter((p): p is string => p !== null);
          return (
            <polyline key={s.label} points={pts.join(" ")} fill="none" stroke={PALETTE[si % PALETTE.length]} strokeWidth={2} />
          );
        })}
        <text x={(left + W - right) / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="currentColor">
          epoch
        </text>
        <text x={12} y={top + 4} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          {yLabel}
        </text>
      </svg>
      <Legend labels={series.map((s) => s.label)} />
    </figure>
  );
}
