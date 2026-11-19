# Viva questions and answers

Answers refer to this project's implementation. Each has a short *plain* answer and, where useful, a *technical* one.
Check numbers (split sizes, paper results) against the paper before the viva.

## A. Problem and concepts

**1. What problem does the project solve?**
Given a sentence, return the most relevant images from a collection. *Technical:* cross-modal retrieval via a shared embedding space; query and images are embedded and ranked by cosine similarity.

**2. How is image retrieval different from image classification?**
Classification picks one label from a fixed set; retrieval ranks an open collection for an arbitrary query. *Technical:* retrieval is judged by ranking metrics (Recall@K, rank), classification by accuracy; queries are free text and unseen at training time.

**3. Why map images and text into a *shared* space?**
So that "a dog in snow" and a photo of that scene end up close, making comparison a simple distance. *Technical:* two encoders project to the same D-dimensional space (1,024) where cosine similarity is meaningful; training with a ranking loss enforces matching pairs > mismatching pairs.

**4. Why not just search the caption text of each image?**
Keyword search fails on paraphrases ("puppy" vs "dog") and on images without captions. The embedding approach generalises through learned semantics; captions are used only for training and evaluation.

**5. What is the role of the baseline?**
It shows how far a simple method goes (mean word vector → linear map), so the benefit of sequence modelling and ranking loss can be measured fairly.

## B. Image features

**6. What is VGG-19 and what is fc7?**
A 19-layer CNN trained on ImageNet. fc7 is the second fully-connected layer (4,096-d) after ReLU, used as a generic image descriptor. *In code:* `model.classifier[:5]` in eval mode.

**7. Why freeze the CNN and cache features?**
It matches the research setup, makes training fast (only small layers train) and lets every model share the same features. Downside: no fine-tuning of the visual side.

**8. How do torchvision's VGG-19 features differ from the paper's?**
Different weights/training framework, single centre crop vs. possible multi-crop, different resize details; small numerical differences are expected and reported as a limitation.

**9. How do you make sure features match the right image?**
Features are stored with an id list in the same order (`<split>_ids.json`); tests check row *i* belongs to `ids[i]`, and `build_split_data` raises if ids and records disagree.

## C. Text processing and embeddings

**10. What is GloVe?**
Pretrained 300-d word vectors learned from word co-occurrence statistics, so related words have nearby vectors. We initialise the embedding layer with them.

**11. Why initialise with GloVe instead of random?**
It transfers knowledge from a huge corpus, helps rare words and speeds convergence; the `gru_no_glove.yaml` ablation measures the effect. *Fine-tune or freeze?* Default is fine-tune; `freeze_embeddings: true` freezes.

**12. How are out-of-vocabulary words handled?**
They map to `<unk>`; GloVe-missing in-vocabulary words get small random vectors; at query time the API lists ignored words, and refuses a query where *no* word is known.

**13. Why is the vocabulary built from the training split only?**
Using validation/test words would leak information about the evaluation data.

**14. How are variable-length captions batched?**
Padded to a common length with id 0 and a `lengths` vector; the RNN uses packed sequences so padding cannot influence the result (a test checks this).

## D. Models

**15. What is a GRU and how does it differ from an LSTM?**
Both are gated RNNs that mitigate vanishing gradients. GRU has two gates (update, reset) and no separate cell state; LSTM has three gates and a cell state. GRU has fewer parameters and is often similar in accuracy.

**16. Why a recurrent encoder rather than averaging word vectors?**
It can use word order and context ("man bites dog" ≠ "dog bites man"), which averaging discards.

**17. How is the sentence vector produced?**
Final hidden state of the top layer → linear projection → L2 normalisation.

**18. How is the baseline trained?**
Closed-form ridge regression from mean GloVe vectors to image fc7 features; the ridge strength is chosen on validation. Retrieval compares the *predicted* fc7 with real fc7 by cosine.

**19. Why L2-normalise embeddings?**
Then the dot product equals cosine similarity, scores are bounded in [-1, 1] and not dominated by vector length.

**20. How many parameters/what size is the shared space?**
Embedding dim 1,024 (config); text encoder hidden 1,024; image projection 4,096→1,024. Dimensions are configurable.

## E. Loss and training

**21. Explain the ranking loss.**
For each matching pair, every mismatching pair in the batch must score at least a margin α (0.2) lower: `max(0, α − s(i,t) + s(i,t'))`, summed over wrong captions and wrong images. Pairs already separated by α contribute zero.

**22. What are positive and negative samples here?**
Positive: an image with one of its own captions. Negatives: all other images/captions in the same mini-batch.

**23. What if two captions of the same image are in one batch?**
They are not negatives; the loss masks pairs that share an image (via `image_index`).

**24. Contrastive vs. ranking (hinge) loss?**
Both pull positives together and push negatives apart. Hinge/margin losses stop caring once the margin is met; InfoNCE-style contrastive losses use a softmax over all negatives with a temperature. We use the hinge form (as in the research setup).

**25. What does `max_violation` do?**
Uses only the hardest negative per query (VSE++), which often trains better but can be unstable early; default is the sum over negatives.

**26. How do you detect overfitting?**
Train loss keeps falling while validation loss/recall stops improving or worsens; early stopping with patience 5 halts training and `best.pt` keeps the best validation epoch.

**27. What is early stopping and what does it monitor?**
Stops when the validation metric (`t2i_rsum` = R@1+R@5+R@10) has not improved for `patience` epochs.

**28. How is data leakage prevented?**
Official split, id-level leakage check, optional content-hash duplicate removal (evaluation copy kept), train-only vocabulary, validation-only model selection, and `evaluate.py` blocks the test split unless `--allow-test` is given.

**29. Why Adam and gradient clipping?**
Adam adapts step sizes per parameter and converges reliably without much tuning; clipping (2.0) prevents exploding gradients in RNNs.

**30. How does resuming work?**
`last.pt` stores weights, optimiser, history and the early-stopping counter; `--resume` restores them and fast-forwards the shuffling RNG.

## F. Evaluation

**31. Define Recall@K.**
The percentage of queries whose correct item is in the top K results. R@1 is the strictest, R@10 the most forgiving.

**32. Define mean rank and why report it too.**
Average position of the correct item (lower is better). It reflects the whole ranking, not just the top K, and is sensitive to bad outliers.

**33. Text-to-image versus image-to-text?**
T2I: each caption queries all images (one correct image). I2T: each image queries all captions (five correct captions; the best-ranked counts). The web app is T2I; both are reported and labelled.

**34. What are the `full` and `1k` protocols?**
`full`: candidates = every image of the split. `1k`: average over consecutive 1,000-image folds (the usual COCO 5k convention). Recalls are higher on 1k because fewer distractors; never compare numbers across protocols.

**35. How are ties handled?**
Pessimistically: an equal score counts as ranked ahead, so degenerate constant embeddings score at chance rather than 100 %.

**36. How do you verify the metrics are right?**
Unit tests with hand-computed ranks, perfect embeddings (R@1 = 100), random embeddings (≈ chance), ties and fold averaging.

**37. How do your results compare with the paper's 37.0 % R@1 / 86.8 % R@10?**
*Answer with your measured numbers.* The paper's values are a reference under its own setup; differences can come from a subset, torchvision weights, crops, optimiser, early-stopping, protocol (full vs 1k), vocabulary and GloVe coverage. We do not claim to reproduce them.

## G. Retrieval system

**38. How does search work at query time?**
Validate text → tokenise with the model vocabulary → encode → normalise → inner product with all image embeddings → top-k → attach ids, scores and captions.

**39. Why exact search instead of an approximate index?**
For ≤ 10⁵ vectors a matrix product is milliseconds. FAISS `IndexFlatIP` is also exact; approximate indexes (IVF/HNSW) trade some recall for speed on millions of items and are not needed here.

**40. Time and memory complexity?**
Query: O(N·D) multiply-adds to score N images (+ O(N) top-k selection). Memory: N·D·4 bytes (5,000 × 1,024 ≈ 20 MB). Training: O(B²·D) per batch for the similarity matrix.

**41. What happens when new images are added?**
Extract their features, embed them with the same model and `ImageIndex.add`/rebuild; the model need not be retrained.

**42. How do you guarantee a result id maps to the correct file?**
The index stores ids in embedding-row order; the image store maps id → file via metadata under a fixed root; tests check row↔feature equality and path-escape safety.

**43. What does the score mean?**
Cosine similarity in [-1, 1]; useful for ranking only. It is not a calibrated probability and values are not comparable across different models.

## H. Engineering, deployment, ethics

**44. Describe the API architecture and error handling.**
FastAPI with Pydantic schemas, a service layer (`SearchService`) loaded once at startup, consistent `{error:{code,message}}` bodies, status codes 404/422/503, CORS limited to the dev origin, images served only by known ids.

**45. How is the project reproducible?**
Fixed seeds, YAML configs, cached features, recorded environment and checkpoint SHA-256 in `model_metadata.json`, deterministic subsetting, pinned dependency ranges and instructions.

**46. What dataset biases and licensing issues exist?**
COCO reflects its sources' social and photographic biases and contains identifiable people; annotations are CC BY 4.0, images follow Flickr terms and must not be redistributed; GloVe is PDDL. Results should not be treated as neutral and the system must not be used for identification.

**47. What are the main limitations and how would you extend the work?**
Weak with counting/negation/spatial relations, ignores unknown words, single sentence, frozen CNN. Extensions: attention over image regions, context-aware/multi-turn queries (paper's future work), fine-tuning the CNN, hard-negative mining, modern pretrained models such as CLIP (kept separate from the paper's methods).

**48. Why did you choose a margin of 0.2 and 1,024 dimensions?**
They are common values in this literature and the dimension matches the reported experiments; both are configurable and could be tuned on the validation set.

**49. What would you do with limited compute?**
Use a labelled subset, cache features once, train on CPU (small layers), reduce batch size, and report results as subset results.

**50. What did the smoke test show and not show?**
It shows the pipeline works end to end on synthetic shapes; it says nothing about real-world quality, so its numbers are never reported as results.
