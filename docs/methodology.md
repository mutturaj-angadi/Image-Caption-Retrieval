# Methodology

The reference paper's setup was taken from the project brief (MS-COCO with five captions per image, 113,287 / 5,000 / 5,000
split, VGG-19 fc7 4,096-d features, 300-d GloVe, multi-response linear regression baseline, GRU/LSTM text encoders, margin
ranking loss with negatives, 1,024-d shared space, Recall@K and mean rank). **Cross-check every statement below against the
paper PDF before submission** — the PDF itself was not available to the author of this code.

## 1. Data

* Images and five captions per image from MS-COCO 2014 with the Karpathy split (restval merged into train).
* Captions are lower-cased and tokenised on alphanumeric runs (`src/data/text.py`). The vocabulary is built from **training** captions only (`min_freq = 5` by default); words not in it map to `<unk>`.
* Images with fewer than five captions are dropped; extra captions are truncated so every image has exactly five.
* Validation removes missing/corrupted images and duplicate ids; optional MD5 hashing finds byte-identical images across splits (the evaluation copy is kept). A leakage check asserts that no image id occurs in two splits.

## 2. Image features

Pretrained VGG-19 (torchvision `IMAGENET1K_V1`), eval mode, `torch.inference_mode`; resize 256 → centre-crop 224 → ImageNet normalisation. The 4,096-d activation after the second fully connected layer and its ReLU (**fc7**) is stored.
Differences from the paper: torchvision weights (not the original Oxford/Caffe model), a single centre crop (the original may have used multi-crop averaging), different resize/interpolation details. Expect small numerical differences.

## 3. Models

**A. Baseline (multi-response linear regression).** Caption vector `x` = mean of its word vectors (300-d). Learn `W` minimising `‖Y − [X,1]W‖² + λ‖W_{w}‖²` (closed form, intercept unpenalised) where `Y` are the fc7 features of the described images. A caption query is mapped to a predicted fc7 vector and compared with all image features by cosine similarity. λ is selected on the validation split (λ = 0 would be plain least squares; the ridge term is an engineering addition for conditioning).

**B. GRU + GloVe.** `tokens → Embedding(300, GloVe-initialised) → GRU(1024) → final hidden state → Linear(1024→1024) → L2-normalise`. Images: `fc7 → Linear(4096→1024) → L2-normalise`. Variable-length captions use packed sequences so padding never affects the result.

**C. LSTM + GloVe.** Identical, with an LSTM cell (hidden state `h_n` used).

Embeddings can be frozen (`model.freeze_embeddings`) or fine-tuned (default) and GloVe can be replaced by random initialisation (`configs/gru_no_glove.yaml`).

## 4. Training objective

With unit vectors `i_k`, `t_k` (image/caption of pair *k*), cosine score `s(i, t) = i·t` and margin `α` (0.2):

```
L = 1/B Σ_k Σ_{j≠k} [ max(0, α − s(i_k,t_k) + s(i_k,t_j)) + max(0, α − s(i_k,t_k) + s(i_j,t_k)) ]
```

Negatives are the other pairs in the mini-batch. Because a batch may contain several captions of the same image, pairs that share an image are **masked out** and never treated as negatives. `max_violation: true` keeps only the hardest negative (VSE++ variant). Optimiser: Adam (configurable), gradient clipping 2.0.
The loss is checked against a loop-based NumPy reference in `tests/test_models.py`.

## 5. Model selection and leakage control

Per-epoch evaluation, early stopping (patience 5) and checkpoint selection use the **validation** split only (`train.selection_metric`, default text→image R@1+R@5+R@10). `evaluate.py` refuses to touch the test split without `--allow-test`, and the vocabulary and ridge λ are also chosen without test data.

## 6. Evaluation

* **Text → image** (what the web app does): every caption is a query against all images; target = its image. Recall@K = % of queries with the target in the top K; mean rank = average 1-based rank.
* **Image → text**: every image is a query against all captions; the best-ranked of its five captions defines the rank.
* Ties are counted against the model (pessimistic), so constant scores earn no credit.
* Protocols: `full` (all images of the split) and `1k` (mean over consecutive 1,000-image folds; needs ≥ 1,000 images).
* Latency: median over 50 single-caption queries (encode + score all images), with the hardware recorded in the result JSON.

## 7. Retrieval index

All image embeddings are L2-normalised once; cosine similarity is an inner product. Exact search is an `N×D` matrix product followed by top-k; FAISS `IndexFlatIP` is optional. The index is tied to a model by dimension and checkpoint hash; rebuild it after retraining or after adding images (`ImageIndex.add`).

## 8. Differences from the original research (summary)

| Aspect | Paper (per brief) | This project |
|---|---|---|
| VGG-19 weights | original model | torchvision ImageNet weights, centre crop |
| Optimiser / schedule | as in paper | Adam, early stopping on validation |
| Baseline | linear regression | ridge-regularised linear regression |
| Loss negatives | negative examples with margin | in-batch negatives with margin, same-image pairs masked |
| Evaluation | Recall@K, mean rank | same, plus median rank and both protocols |
| Future work in paper | dynamic attention, context-aware retrieval | not implemented |

## 9. Optional extension

A pretrained CLIP model could be added as a fourth retriever (`BaseRetriever` interface) and reported separately, clearly labelled as a modern pretrained system rather than the paper's method. It is **not implemented** in this version.
