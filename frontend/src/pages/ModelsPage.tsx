import { api } from "../api";
import { Banner, Card, EmptyState, Spinner } from "../components/Feedback";
import { useAsync } from "../hooks";
import type { ModelInfo } from "../types";

const EXPLAINERS = [
  {
    title: "Linear-regression baseline",
    body: "Averages the GloVe vectors of a caption's words and learns one linear map (ridge-regularised least squares) from that 300-d vector to the 4,096-d VGG-19 fc7 feature of the image. Retrieval compares the predicted feature with every image feature by cosine similarity.",
  },
  {
    title: "GRU + GloVe",
    body: "Words are embedded with GloVe-initialised vectors and read by a gated recurrent unit; its final hidden state is projected into the shared space. Image fc7 features are projected into the same space. A margin-based ranking loss pulls matching pairs together and pushes mismatched pairs apart.",
  },
  {
    title: "LSTM + GloVe",
    body: "Same pipeline as the GRU model with a long short-term-memory cell. It is evaluated with the identical protocol so the two recurrent cells can be compared fairly.",
  },
];

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function ModelCard({ m }: { m: ModelInfo }) {
  const arch = m.metadata.architecture ?? {};
  const ck = m.metadata.checkpoint;
  const training = m.metadata.training ?? {};
  const evals = Object.entries(m.metadata.evaluation ?? {});
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-lg font-semibold">{m.name}</h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            m.available ? "bg-emerald-100 text-emerald-900" : "bg-red-100 text-red-900"
          }`}
        >
          {m.available ? "available" : "unavailable"}
        </span>
      </div>
      {!m.available && <p className="mt-2 text-sm text-red-700 dark:text-red-300">{m.unavailable_reason ?? "Unknown reason"}</p>}
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-slate-500">Type</dt>
        <dd>{m.type}</dd>
        <dt className="text-slate-500">Embedding dimension</dt>
        <dd>{fmt(m.embedding_dim)}</dd>
        <dt className="text-slate-500">Hidden size / layers</dt>
        <dd>
          {fmt(arch["hidden_dim"])} / {fmt(arch["num_layers"])}
        </dd>
        <dt className="text-slate-500">Pretrained word vectors</dt>
        <dd>{fmt(arch["pretrained_embeddings"])}</dd>
        <dt className="text-slate-500">Vocabulary size</dt>
        <dd>{fmt(arch["vocab_size"])}</dd>
        <dt className="text-slate-500">Image features</dt>
        <dd>{fmt(arch["image_feature_extractor"])}</dd>
        <dt className="text-slate-500">Indexed images</dt>
        <dd>
          {m.num_images} ({m.index_backend ?? "no index"})
        </dd>
        <dt className="text-slate-500">Dataset</dt>
        <dd>
          {fmt(m.metadata.dataset?.name)}
          {m.metadata.dataset?.is_subset ? " (subset)" : ""}
        </dd>
        <dt className="text-slate-500">Checkpoint</dt>
        <dd className="break-all font-mono text-xs">
          {fmt(ck?.file)} · {ck?.sha256 ? ck.sha256.slice(0, 12) : "no hash"}
        </dd>
        <dt className="text-slate-500">Training</dt>
        <dd>
          {fmt(training["epochs_run"])} epochs (best {fmt(training["best_epoch"])}), {fmt(training["train_seconds"])} s on {fmt(training["device"])}
        </dd>
      </dl>
      <h4 className="mt-4 text-sm font-semibold">Recorded evaluations</h4>
      {evals.length === 0 ? (
        <p className="mt-1 text-sm text-slate-500">None yet — run scripts/evaluate.py.</p>
      ) : (
        <ul className="mt-1 space-y-1 text-sm">
          {evals.map(([key, e]) => (
            <li key={key}>
              <span className="font-mono text-xs">{key}</span>: text→image R@1 {e.metrics["t2i_R@1"]?.toFixed(1)}%, R@10{" "}
              {e.metrics["t2i_R@10"]?.toFixed(1)}%, mean rank {e.metrics["t2i_mean_rank"]?.toFixed(1)}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function ModelsPage() {
  const models = useAsync(() => api.models());
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Models</h1>
      <div className="grid gap-4 md:grid-cols-3">
        {EXPLAINERS.map((e) => (
          <Card key={e.title} title={e.title}>
            <p className="text-sm text-slate-700 dark:text-slate-300">{e.body}</p>
          </Card>
        ))}
      </div>
      <Card title="Metrics used">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
          <li>
            <strong>Recall@K</strong> — percentage of queries whose correct item is ranked within the top K (higher is better).
          </li>
          <li>
            <strong>Mean rank</strong> — average position of the correct item (lower is better).
          </li>
          <li>
            <strong>Text→image</strong>: each caption is a query and the image it describes is the target (what this app does).{" "}
            <strong>Image→text</strong>: each image is a query and its five captions are targets.
          </li>
        </ul>
      </Card>
      <h2 className="text-xl font-semibold">Trained models in this installation</h2>
      {models.loading && <Spinner label="Loading models…" />}
      {models.error && (
        <Banner tone="error" title="Cannot load models">
          {models.error}
        </Banner>
      )}
      {models.data && models.data.models.length === 0 && (
        <EmptyState title="No trained models found">Run scripts/train.py or scripts/smoke_test.py.</EmptyState>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        {models.data?.models.map((m) => (
          <ModelCard key={m.name} m={m} />
        ))}
      </div>
    </div>
  );
}
