# Live demonstration script (≈ 4 minutes)

**Before the demo (do these 10 minutes earlier):** activate the venv, confirm `Invoke-RestMethod http://localhost:8000/health` says `ok`, start the frontend, open the Search page, check that the models you will show are `available`, and keep screenshots of each step as a fallback in case the network/hardware misbehaves. Run the demo on the real (COCO) workspace, not the synthetic one.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Search page, point at the search box. | "NeuralSeek retrieves images from a sentence. Nothing here uses labels or tags: the sentence and each image are embedded in one space and we rank by cosine similarity." |
| 0:20 | Click the example **"A dog running through snow"**; scroll the grid. | "These are the top 10 test images in rank order. The bar and number are cosine scores — a ranking signal, not a probability. Hover/click a card to see the five human captions of that image." |
| 1:00 | Open the lightbox on result #1, then #2 (arrow key). | "We can inspect each hit and its ground-truth captions to judge relevance." |
| 1:20 | Type **"three teddy bears lying together on a bed"**, change *Model* to the baseline, repeat with the GRU model. | "The baseline averages word vectors, so it ignores word order; the GRU reads the sentence in order. Compare the two rankings." *(Only say which is better if the grid actually shows it.)* |
| 2:10 | **Models** page. | "Architecture, embedding size 1024, checkpoint hash, vocabulary size and training time are recorded for every model." |
| 2:30 | **Evaluation** page. | "These numbers come from `scripts/evaluate.py` on the held-out test split: Recall@1/5/10 and mean rank. The paper's historical values appear only as a labelled reference. Our numbers differ because of [subset size / torchvision weights / …]." |
| 3:15 | Failure case: search **"two people playing tennis at night"** (or any query that fails; prepare one in advance from your failure analysis). | "Here it returns daytime images / wrong count. Retrieval is semantic but imperfect: counting, negation and rare words are weak, and out-of-vocabulary words are ignored (the app lists them)." |
| 3:40 | Optional: type a blank query. | "Inputs are validated and errors are explained instead of crashing." |
| 3:50 | Close. | "Code, tests, report and the exact commands are in the repository." |

## If something goes wrong
* *No results / Cannot reach the API:* show the `/docs` page and the screenshots; explain the command that starts the API.
* *Images missing:* set `NCIR_IMAGES_ROOT` and restart the backend.
* *Slow first query:* the first request warms the model; do one query before the demo starts.
