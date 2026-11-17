# 00 – Project Overview

## Problem

Ripeness grades are **ordered**: unripe < partially ripe < ripe < rotten.
A normal softmax classifier treats them as unrelated labels, so predicting
"unripe" for a rotten fruit costs exactly the same as predicting "ripe" for it.
That ignores useful structure. Ordinal regression uses the order.

## Method: ordinal regression with cumulative (CORAL-style) targets

With K = 4 grades, the model answers K−1 = 3 yes/no questions:

| Question | unripe (0) | partially ripe (1) | ripe (2) | rotten (3) |
|---|---|---|---|---|
| grade > 0 ? | 0 | 1 | 1 | 1 |
| grade > 1 ? | 0 | 0 | 1 | 1 |
| grade > 2 ? | 0 | 0 | 0 | 1 |

* Head: one shared linear weight vector → a single score `s`, plus 3 learned
  biases `b_k`. Logit for question k is `s + b_k`.
* Loss: binary cross-entropy summed over the 3 questions.
* Prediction: number of questions whose probability > 0.5.
* Sharing the weight vector keeps the answers consistent (rank-consistent):
  "> 2" can never be more likely than "> 1".

Reference: Cao, Mirjalili, Raschka (2020), *Rank consistent ordinal regression
for neural networks* (CORAL).

## Experiments (controlled comparison)

| Model | Backbone | Head / loss |
|---|---|---|
| A. Baseline CNN | small CNN from scratch | softmax + cross-entropy |
| B. Transfer softmax | pretrained ImageNet backbone | softmax + cross-entropy |
| C. Transfer ordinal | same pretrained backbone | ordinal head + BCE |

B vs C differ **only** in the head and loss: same data split, augmentation,
optimiser, schedule, and seeds. Each runs with 3 seeds.

## Metrics

* **MAE** (mean absolute grade error) – main metric, it measures how far off.
* Accuracy.
* **QWK** (quadratic weighted kappa) – standard ordinal agreement score.
* **Severe error rate** – share of predictions 2 or more grades off.
* Confusion matrix.

## Syllabus mapping (CSE471)

| Project part | Unit |
|---|---|
| Image representation, resizing, normalisation, colour | I |
| Loss functions, activation, optimisation | I |
| CNN layers, pooling, BatchNorm, dropout, augmentation, LR scheduling, early stopping | II |
| Pretrained backbone, freezing/unfreezing, layer-wise learning rates, fine-tuning | III |
| Bounding boxes (dataset is labelled in YOLO box format; boxes are used to crop fruits) | IV |
| Grad-CAM interpretability | VI |

Out of syllabus (allowed): ordinal regression (CORAL), QWK metric.
