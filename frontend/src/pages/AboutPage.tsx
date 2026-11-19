import { Card } from "../components/Feedback";
import { COURSE, INSTITUTION, TEAM } from "../team";

const REFERENCES = [
  "Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014.",
  "Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015 (VGG-19).",
  "Pennington, Socher & Manning, GloVe: Global Vectors for Word Representation, EMNLP 2014.",
  "Cho et al., Learning Phrase Representations using RNN Encoder-Decoder, EMNLP 2014 (GRU).",
  "Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation 1997.",
  "Karpathy & Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Descriptions, CVPR 2015 (split).",
  "Kiros, Salakhutdinov & Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, 2014.",
  "Faghri et al., VSE++: Improving Visual-Semantic Embeddings with Hard Negatives, BMVC 2018.",
];

export function AboutPage() {
  const filled = TEAM.filter((m) => m.name.trim());
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">About NeuralSeek</h1>
      <Card title="Objective">
        <p className="text-sm text-slate-700 dark:text-slate-300">
          Retrieve and rank images from a natural-language description by aligning captions and images in a shared embedding space,
          and evaluate the approach against a linear-regression baseline using Recall@K and mean rank.
        </p>
      </Card>
      <Card title="Architecture">
        <ol className="list-decimal space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
          <li>Images → pretrained VGG-19 fc7 features (4,096-d, cached once).</li>
          <li>Captions → tokens → GloVe-initialised embeddings → GRU/LSTM → projection to the shared space.</li>
          <li>Training with a bidirectional margin ranking loss on matching vs. mismatching pairs.</li>
          <li>Image embeddings are indexed (exact cosine search; FAISS optional for large corpora).</li>
          <li>A FastAPI service embeds each query with the same model and returns the nearest images; this React app displays them.</li>
        </ol>
      </Card>
      <Card title="Research inspiration and honesty notes">
        <p className="text-sm text-slate-700 dark:text-slate-300">
          The methodology follows the reference paper on neural caption–image retrieval (GloVe + GRU/LSTM, contrastive ranking loss,
          Recall@K and mean rank on MS-COCO). Modern engineering choices (torchvision VGG-19 weights, Adam optimiser, validation-based early
          stopping) differ from the original setup, and this project does not claim to reproduce the paper's numbers.
        </p>
      </Card>
      <Card title="Limitations">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
          <li>Only images present in the index can be returned; the index is built from one dataset split.</li>
          <li>Single-sentence queries; no attention, multi-turn context or fine-grained region reasoning.</li>
          <li>Words outside the training vocabulary are ignored; counting, negation and spatial relations are often wrong.</li>
          <li>COCO reflects the social and photographic biases of its sources; results are not neutral.</li>
          <li>Scores are cosine similarities used for ranking, not calibrated probabilities.</li>
        </ul>
      </Card>
      <Card title="Run it locally">
        <pre className="overflow-x-auto rounded-md bg-slate-100 p-3 text-xs dark:bg-slate-800">{`# backend (repo root)
uvicorn backend.main:app --port 8000
# frontend
cd frontend; npm install; npm run dev`}</pre>
      </Card>
      <Card title="Team">
        <p className="mb-2 text-sm text-slate-600 dark:text-slate-400">
          {COURSE} · {INSTITUTION}
        </p>
        {filled.length === 0 ? (
          <p className="text-sm text-amber-700 dark:text-amber-300">
            Team details not filled in yet. Edit <code>frontend/src/team.ts</code> with real names, SRNs and contributions.
          </p>
        ) : (
          <ul className="space-y-1 text-sm">
            {filled.map((m) => (
              <li key={m.srn || m.name}>
                <strong>{m.name}</strong> {m.srn && `(${m.srn})`} — {m.contribution || "contribution not specified"}
              </li>
            ))}
          </ul>
        )}
      </Card>
      <Card title="References">
        <ul className="list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-400">
          {REFERENCES.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
