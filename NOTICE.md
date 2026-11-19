# Third-party data, models and licensing notes

This repository contains **code only**. Datasets, word vectors and pretrained weights are
downloaded by you and remain under their own licences. Please verify the current terms on the
official pages before redistributing anything.

| Resource | Used for | Licence / terms (verify at source) |
|---|---|---|
| Microsoft COCO captions + images (https://cocodataset.org) | Training / evaluation data | Annotations: CC BY 4.0. Images: Flickr terms of use; do not redistribute the images. |
| Karpathy & Fei-Fei caption split (`dataset_coco.json`) | Standard 113,287 / 5,000 / 5,000 split | Distributed with the "Deep Visual-Semantic Alignments" project page; cite the paper. |
| GloVe 6B vectors (https://nlp.stanford.edu/projects/glove/) | Word embeddings | Public Domain Dedication and License (PDDL) v1.0. |
| torchvision VGG-19 ImageNet weights | Image features | Weights follow torchvision's model licence; trained on ImageNet, which has research-use terms. |
| OpenAI CLIP (optional comparison only) | Not bundled | MIT code; check the model-card terms before use. |

## Code licence

No code licence has been chosen. The team should decide (for a graded course project, "all rights
reserved" until submission is a normal default) and add a `LICENSE` file if the repository is
made public.

## Ethical notes

COCO captions and images contain social biases (e.g. gender, occupation and scene stereotypes)
and may include identifiable people. Retrieval results should not be treated as neutral or
authoritative. Do not use this system for identifying or surveilling individuals.
