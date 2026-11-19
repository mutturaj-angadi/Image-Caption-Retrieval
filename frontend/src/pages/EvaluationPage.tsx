import { api } from "../api";
import { GroupedBarChart, LineChart } from "../components/Charts";
import { Banner, Card, Spinner } from "../components/Feedback";
import { useAsync } from "../hooks";
import type { EvalResult } from "../types";

const f1 = (v: number | undefined) => (v === undefined ? "—" : v.toFixed(1));

function ResultsTable({ rows }: { rows: EvalResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-left text-sm">
        <caption className="sr-only">Measured retrieval metrics per model</caption>
        <thead className="border-b border-slate-200 text-xs uppercase text-slate-500 dark:border-slate-700">
          <tr>
            {["Model", "Split / protocol", "Images", "T2I R@1", "T2I R@5", "T2I R@10", "T2I mean rank", "I2T R@1", "I2T R@10", "Train s", "Latency ms"].map((h) => (
              <th key={h} scope="col" className="px-2 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.file} className="border-b border-slate-100 dark:border-slate-800">
              <th scope="row" className="px-2 py-2 font-medium">
                {r.name}
                {r.is_subset && <span className="ml-1 rounded bg-amber-100 px-1 text-[10px] text-amber-900">subset</span>}
              </th>
              <td className="px-2 py-2">
                {r.split} / {r.protocol}
              </td>
              <td className="px-2 py-2 tabular-nums">{r.num_images}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["t2i_R@1"])}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["t2i_R@5"])}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["t2i_R@10"])}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["t2i_mean_rank"])}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["i2t_R@1"])}</td>
              <td className="px-2 py-2 tabular-nums">{f1(r.metrics["i2t_R@10"])}</td>
              <td className="px-2 py-2 tabular-nums">{r.training_seconds ?? "n/a"}</td>
              <td className="px-2 py-2 tabular-nums">{r.latency_ms_median ?? "n/a"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EvaluationPage() {
  const evaluation = useAsync(() => api.evaluation());
  const stats = useAsync(() => api.stats());
  const ev = evaluation.data;
  const chartSplit = ev?.results.some((r) => r.split === "test") ? "test" : "val";
  const chartRows = (ev?.results ?? []).filter((r) => r.split === chartSplit && r.protocol === "full");
  const curves = Object.entries(ev?.histories ?? {});

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Evaluation</h1>

      {evaluation.loading && <Spinner label="Loading evaluation…" />}
      {evaluation.error && (
        <Banner tone="error" title="Cannot load evaluation results">
          {evaluation.error}
        </Banner>
      )}
      {ev && !ev.available && (
        <Banner tone="warning" title="No evaluation results yet">
          {ev.message} Results are shown here only when they have actually been measured; nothing is estimated.
        </Banner>
      )}
      {ev?.synthetic_warning && (
        <Banner tone="warning" title="Synthetic data">
          Some results were computed on generated smoke-test data. They verify the pipeline only and must not be reported as experimental results.
        </Banner>
      )}

      {ev?.available && (
        <>
          <Card title="Measured results">
            <ResultsTable rows={ev.results} />
            <p className="mt-3 text-xs text-slate-500">
              T2I = text→image retrieval (caption is the query), I2T = image→text. Recall values are percentages; ties are ranked
              pessimistically. Latency is a single-query median under the setup recorded in each JSON file.
            </p>
          </Card>
          {chartRows.length > 0 && (
            <Card title={`Text→image Recall@K (${chartSplit} split)`}>
              <GroupedBarChart
                title={`Text to image recall at K on the ${chartSplit} split`}
                categories={["R@1", "R@5", "R@10"]}
                series={chartRows.map((r) => ({
                  label: r.name,
                  values: [r.metrics["t2i_R@1"], r.metrics["t2i_R@5"], r.metrics["t2i_R@10"]],
                }))}
              />
            </Card>
          )}
        </>
      )}

      {curves.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {curves.map(([name, hist]) => (
            <Card key={name} title={`${name}: training and validation loss`}>
              <LineChart
                title={`${name} loss curves`}
                xs={hist.map((h) => h.epoch)}
                yLabel="ranking loss"
                series={[
                  { label: "train", values: hist.map((h) => h.train_loss) },
                  { label: "validation", values: hist.map((h) => h.val_loss) },
                ]}
              />
            </Card>
          ))}
        </div>
      )}

      <Card title="Dataset statistics">
        {stats.loading && <Spinner label="Loading statistics…" />}
        {stats.error && <Banner tone="error">{stats.error}</Banner>}
        {stats.data && !stats.data.available && <Banner tone="warning">{stats.data.message}</Banner>}
        {stats.data?.available && (
          <>
            <p className="mb-3 text-sm">
              <strong>{stats.data.dataset.name}</strong> — {stats.data.dataset.split_source}
              {stats.data.dataset.is_subset ? " — development SUBSET" : ""}. Vocabulary: {stats.data.stats.vocab_size} words.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-sm">
                <thead className="border-b border-slate-200 text-xs uppercase text-slate-500 dark:border-slate-700">
                  <tr>
                    <th className="px-2 py-2">Split</th>
                    <th className="px-2 py-2">Images</th>
                    <th className="px-2 py-2">Captions</th>
                    <th className="px-2 py-2">Mean length</th>
                    <th className="px-2 py-2">95th pct</th>
                    <th className="px-2 py-2">OOV rate</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.data.stats.splits ?? {}).map(([split, s]) => (
                    <tr key={split} className="border-b border-slate-100 dark:border-slate-800">
                      <th scope="row" className="px-2 py-2 font-medium">
                        {split}
                      </th>
                      <td className="px-2 py-2 tabular-nums">{s.num_images}</td>
                      <td className="px-2 py-2 tabular-nums">{s.num_captions}</td>
                      <td className="px-2 py-2 tabular-nums">{f1(s.caption_len_mean)}</td>
                      <td className="px-2 py-2 tabular-nums">{f1(s.caption_len_p95)}</td>
                      <td className="px-2 py-2 tabular-nums">{s.oov_rate === undefined ? "—" : `${(s.oov_rate * 100).toFixed(2)}%`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>

      {ev?.reference && (
        <Card title="Reference values from the paper (not measured here)">
          <p className="text-sm text-slate-700 dark:text-slate-300">
            The paper reports {ev.reference.model}: Recall@1 {ev.reference["t2i_R@1"]}% and Recall@10 {ev.reference["t2i_R@10"]}% under its own
            experimental setup. {ev.reference.note} Differences from the measured numbers above are expected and are discussed in the report.
          </p>
        </Card>
      )}
    </div>
  );
}
