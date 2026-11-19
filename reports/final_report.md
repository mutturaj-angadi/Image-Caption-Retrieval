# Neural Caption–Image Retrieval: Text-to-Image Search Using Deep Learning

**Course:** UE24CS352A – Machine Learning Mini-Project · **Team:** [TO COMPLETE: names, SRNs, section] · **Guide:** [TO COMPLETE]

## 1. Problem statement
Given a free-form sentence, return the most relevant images from a collection. Keyword matching fails on paraphrases and uncaptioned images, so we learn a joint embedding space where sentences and images are directly comparable, following the neural caption–image retrieval approach of the reference paper.

## 2. Objectives
(i) Build a reproducible pipeline for image–caption data; (ii) implement a linear-regression baseline and GRU/LSTM models with GloVe initialisation and a margin ranking loss; (iii) evaluate with Recall@K and mean rank under a leakage-free protocol; (iv) deliver a working search application (API + web UI) with tests and documentation.

## 3. Dataset and preprocessing
MS-COCO 2014 captions (five human captions per image) with the Karpathy split: 113,287 training, 5,000 validation and 5,000 test images when run in full. **Actual data used in this report: [TO COMPLETE: full split or labelled subset, with image counts].** Captions are lower-cased and tokenised on alphanumeric runs; the vocabulary (minimum frequency 5) is built from training captions only; unknown words map to `<unk>`. Images with missing/corrupt files, duplicate ids or fewer than five captions are removed, and a check asserts that no image appears in two splits.

## 4. Methodology and model architecture
**Image features.** Pretrained VGG-19 (torchvision weights, eval mode, 256→224 centre crop); the 4,096-d fc7 activation is cached. **Baseline (Model A).** Mean 300-d GloVe vector of the caption → ridge-regularised multi-response linear regression to the fc7 feature; images are ranked by cosine similarity with the predicted feature. **GRU + GloVe (Model B)** and **LSTM + GloVe (Model C).** Tokens → GloVe-initialised embeddings → recurrent layer (1,024 units) → linear projection → L2-normalisation; image fc7 → linear projection (4,096→1,024) → L2-normalisation. With unit vectors and cosine score *s*(*i*,*t*) = *i*·*t*, the loss for a batch of *B* matching pairs is

*L* = (1/*B*) Σ<sub>k</sub> Σ<sub>j≠k</sub> [ max(0, α − *s*(*i*<sub>k</sub>,*t*<sub>k</sub>) + *s*(*i*<sub>k</sub>,*t*<sub>j</sub>)) + max(0, α − *s*(*i*<sub>k</sub>,*t*<sub>k</sub>) + *s*(*i*<sub>j</sub>,*t*<sub>k</sub>)) ],  α = 0.2,

where pairs sharing an image are never used as negatives. Training uses Adam, gradient clipping and early stopping on validation Recall; the test split is touched only for the final evaluation.

## 5. Implementation overview
Python/PyTorch library (`src/`) with configs, scripts and automated tests; retrieval over L2-normalised embeddings with exact cosine search (FAISS optional); a FastAPI service (`/api/search`, `/api/models`, `/api/evaluation`, …) loading models once at startup; and a React/TypeScript interface with Search, Models, Evaluation and About pages. A synthetic smoke-test mode exercises the pipeline without downloading COCO (its numbers are never reported as results). See the README for the exact checks executed in this workspace.

## 6. Experimental setup
Embedding size 1,024, margin 0.2, Adam (learning rate 2×10<super>-4</super>), batch size 128, seed 42, patience 5. Metrics: text→image Recall@1/5/10 and mean rank (each caption is a query; the single described image is the target) and the reverse direction; ties are ranked pessimistically. **Protocol, hardware and software: [TO COMPLETE from `reports/metrics/*.json`: full or 1k folds, CPU/GPU, Python/PyTorch versions].**

## 7. Results
{{RESULTS_TABLE}}

{{RESULTS_NOTE}}

{{FIGURES}}

## 8. Qualitative analysis and failure cases
**[TO COMPLETE after experiments: two or three example queries with their top results, and at least three failure cases (e.g. counting, negation, rare words) with a short explanation, plus the effect of query length if analysed.]**

## 9. Conclusion
The repository implements the text-to-image retrieval pipeline and its synthetic smoke demo. Real-data experiments and model comparisons remain pending. **[TO COMPLETE after experiments: state which model performed best, by how much, how the results compare with the paper's reference values (Recall@1 37.0 %, Recall@10 86.8 % for GRU + GloVe under the paper's own setup), and the likely causes of any gap.]** No claim of reproducing the paper's numbers is made.

## 10. Limitations
Only indexed images can be retrieved; queries are single sentences without attention or context; unknown words are ignored and counting, negation and spatial relations are weak; torchvision VGG-19 weights and the training recipe differ from the original; COCO carries social and photographic biases; scores are cosine similarities for ranking, not probabilities. Dynamic attention and context-aware retrieval (the paper's future work) are not implemented.

## 11. References
[1] Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014. [2] Karpathy & Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Descriptions, CVPR 2015. [3] Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015. [4] Pennington, Socher & Manning, GloVe: Global Vectors for Word Representation, EMNLP 2014. [5] Cho et al., Learning Phrase Representations using RNN Encoder–Decoder, EMNLP 2014. [6] Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation 1997. [7] Kiros et al., Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, 2014. [8] Faghri et al., VSE++: Improving Visual-Semantic Embeddings with Hard Negatives, BMVC 2018. [9] *Neural Caption–Image Retrieval* (course reference paper): [TO COMPLETE: authors, venue, year].
