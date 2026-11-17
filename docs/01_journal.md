# 01 – Process Journal

Each entry records: what was done, the exact command, the real output, and the decision taken.

---

## 2026-11-17 · Step 0 – Environment check

**Command**
```
nvidia-smi ; python --version ; python -c "import torch, torchvision; ..."
```
**Output**
```
NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB
Python 3.14.0
torch 2.14.1+cu126, CUDA available: True
torchvision 0.29.1+cu126
scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.7, numpy 2.3.5
```
**Decision:** framework is PyTorch on the local GPU. 4 GB of VRAM means small
backbones (ResNet18 / EfficientNet-B0) at 224×224 with mixed precision (AMP).

## 2026-11-17 · Step 1 – Dataset choice

Candidates looked at:
* Kaggle "Fruit Ripeness: Unripe, Ripe, and Rotten" – only 3 stages, and needs a Kaggle login.
* Mendeley "Fresh and Rotten Fruits" – only 2 classes (fresh/rotten), so not ordinal.
* **Mendeley "Ripening Stages of Strawberries and Avocados"** – real photos, 2 fruits,
  **4 ordered stages**, CC BY 4.0, direct download. **Chosen.**

**Command**
```
curl -L -o data/raw/dataset.zip https://data.mendeley.com/public-api/zip/zysvgmxcyz/download/1
```

**Output:** HTTP 200, 2,110,769,149 bytes (= 1.97 GiB, matches the listed size).

## 2026-11-17 · Step 2 – Inspecting the raw data

Findings (script: zip listing + label parsing):
* The zip holds `images/` (758 JPG photos), `labels/` (1,332 YOLO `.txt` files) and a copy of the
  `yolov5` code repo (not needed). Only 758 labels have a matching photo; the other 574 label files
  have no image in the download. We use the 758 photos that have both.
* Labels are **object-detection boxes** (`class cx cy w h`), not one-class-per-image folders.
  Most label files list one fruit; some list up to 12 fruits in one photo (924 boxes in total
  across the 758 usable photos).
* 8 class ids, **no names file**. We cropped 8 random boxes per class and looked at them:

| id | fruit | stage | grade |
|---|---|---|---|
| 0 | strawberry | unripe (green/white) | 0 |
| 1 | strawberry | partially ripe (pale, part red) | 1 |
| 2 | strawberry | ripe (red) | 2 |
| 3 | strawberry | rotten (mould, shrivelled) | 3 |
| 4 | avocado | unripe (hard green) | 0 |
| 5 | avocado | partially ripe (green, spots) | 1 |
| 6 | avocado | ripe (dark purple/black) | 2 |
| 7 | avocado | rotten (brown, wrinkled) | 3 |

  So `fruit = id // 4` and `grade = id % 4`. The crops also showed **some label noise**
  (a flower and one avocado inside strawberry classes). We keep the labels as given and report it.
* Many photos are edited copies (same fruit pasted on a coloured background) or burst shots.

## 2026-11-17 · Step 3 – Crops and leak-free split (`src/prepare_data.py`)

**Command:** `python src/prepare_data.py`

What it does: crops every labelled box (+8% border) → `data/crops/` (924 crops), then groups crops
that must stay together, and assigns folds.

Iterations (kept honest – each failure taught us something):
1. *8×8 perceptual hash of the whole crop, Hamming ≤ 6* → chained **93 different dark avocados**
   into one group (verified by viewing them). Too loose.
2. *16×16 hash of the crop centre only* → found real duplicate pairs, but true burst-shot pairs
   (IMG_1147/1148, distance 66) and different fruits (distance 63) overlap. A hash cannot separate them.
3. **Final rule:** same source photo, **or** same class with photo numbers in one block of 10
   (catches burst shots), **or** centre-hash distance ≤ 40 → same group. 159 groups.
4. A single 70/15/15 split left only **2 ripe avocados in test** – too few to measure anything.
   **Switched to 5-fold grouped cross-validation**: run k tests on fold k, validates (early stopping)
   on fold k+1, trains on the rest. Every crop is tested exactly once; results are pooled.
   Fold assignment chosen from 200 random seeds by most even class mix (label counts only, no model).

**Output**
```
photos: 758  crops: 924  near-duplicate pairs merged: 3  groups: 159
fold 0 189 {straw-0:47, straw-1:11, straw-2:21, straw-3:4,  avoca-0:35, avoca-1:25, avoca-2:6,  avoca-3:40}
fold 1 239 {straw-0:24, straw-1:22, straw-2:8,  straw-3:68, avoca-0:35, avoca-1:18, avoca-2:6,  avoca-3:58}
fold 2 179 {straw-0:12, straw-1:16, straw-2:21, straw-3:3,  avoca-0:30, avoca-1:25, avoca-2:19, avoca-3:53}
fold 3 148 {straw-0:9,  straw-1:9,  straw-2:9,  straw-3:3,  avoca-0:23, avoca-1:24, avoca-2:6,  avoca-3:65}
fold 4 169 {straw-0:20, straw-1:21, straw-2:10, straw-3:10, avoca-0:33, avoca-1:20, avoca-2:11, avoca-3:44}
leakage check passed: no group appears in two folds
```
**Known limitation:** most rotten strawberries come from one shooting session, so they sit in fold 1.
When fold 1 is the test fold, training sees only ~20 rotten strawberries. This is the price of
preventing leakage; it makes the scores *harder*, not inflated.

## 2026-11-17 · Step 4 – Model code (`src/common.py`, `src/train.py`)

* **Models**
  * `cnn` – small CNN from scratch: 4 blocks of [conv-BN-ReLU ×2 + max-pool], global average pool,
    dropout 0.3, linear → 4 logits (softmax).
  * `tl_softmax` – ImageNet-pretrained **ResNet18**, last layer replaced by linear → 4 logits.
  * `tl_ordinal` – same ResNet18, last layer replaced by the **CORAL ordinal head**
    (one shared weight vector → 1 score, + 3 threshold biases → 3 logits "grade > k").
* **Why ResNet18:** best-known small backbone, fits 4 GB with batch 32 + AMP, ~11 M params.
* **Augmentation:** random resized crop, flips, rotation ±20°, brightness/contrast ±15%.
  **No hue/saturation change** – colour is the label signal.
* **Training:** AdamW (lr 1e-3, wd 1e-4), cosine LR schedule, max 30 epochs, early stopping
  (patience 7) on validation MAE. Transfer models: 3 epochs with the backbone **frozen**, then
  **unfrozen** with the backbone at 10× lower LR (layer-wise learning rates).

## 2026-11-17 · Step 5 – Sanity checks before training (`src/sanity_checks.py`)

Following Karpathy's "recipe for training neural networks": check the plumbing before trusting any number.

```
1. ordinal targets OK: [[0,0,0],[1,0,0],[1,1,0],[1,1,1]]
2. ordinal probs sum to 1, predictions [2,0,3] OK
4. batch: x (64, 3, 224, 224), range [-2.12, 2.64], labels [20, 11, 13, 20]
3. initial loss cnn         1.388  (theory for a clueless model 1.386 = ln 4)
3. initial loss tl_softmax  1.386  (theory 1.386)
3. initial loss tl_ordinal  2.079  (theory 2.079 = 3·ln 2)
```

**Overfit-one-batch test** (`python src/train.py --model X --fold 0 --overfit_one_batch`):
a model that cannot memorise 32 images has a bug.

| model | step 0 | step 50 | step 300 |
|---|---|---|---|
| cnn | 1.353 | 0.004 | 0.0003 |
| tl_softmax | 1.814 | 0.0003 | 0.0000 |
| tl_ordinal (first version) | 1.859 | 0.536 | **0.470 – stuck** |
| tl_ordinal (fixed) | 2.094 | 0.202 | 0.088 (still falling) |

**Bug found and fixed:** the 3 ordinal thresholds started only 0.33 apart and Adam moves each by
about one learning-rate step per update, so they could not spread apart. Fix: start them at
[2, 0, −2] and give them a 10× learning rate (no weight decay). The ordinal model still learns more
slowly than softmax, which is expected: every image must be squeezed into **one** ripeness score.

## 2026-11-17 · Step 6 – First real run (fold 0, seed 0)

```
cnn         TEST mae 0.386  acc 0.693  qwk 0.832  severe 0.079
tl_softmax  TEST mae 0.228  acc 0.794  qwk 0.904  severe 0.016
tl_ordinal  TEST mae 0.254  acc 0.757  qwk 0.905  severe 0.011
```
One fold, one seed = too noisy to conclude anything. Transfer learning clearly beats the scratch CNN.

## 2026-11-17 · Step 7 – Full experiment grid (`src/run_all.py`)

3 models × 5 folds × 3 seeds = 45 runs. Per-run logs in `runs/<model>_f<fold>_s<seed>/`.

## 2026-11-17 · Step 8 – Grad-CAM (`src/gradcam.py`)

Written by hand with PyTorch hooks on ResNet18 `layer4`. A first try used a module backward hook,
which never fired for the ordinal score; it was replaced by a hook on the activation tensor itself.
For the ordinal model we explain the shared **ripeness score** (w·x), so red = "evidence for riper".

Observations on fold-0 test images (`results/gradcam_tl_ordinal.png`):
* Ripe and rotten fruit: heat sits on the fruit surface (dark skin, mould, red flesh) – the model
  looks at the right thing.
* Unripe fruit: heat goes to the image corners or a neighbouring red fruit. That makes sense:
  an unripe fruit has *no* evidence for "riper", so the little positive evidence there is comes from
  elsewhere. It also exposes a real weakness: background and neighbouring fruit can influence the grade.
* The two avocado mistakes (unripe/partially ripe confused) are the visually hardest pair –
  both are green.

## 2026-11-17 · Step 9 – Seed-0 results (all 5 folds pooled, n = 924)

`python src/evaluate.py` after the 15 seed-0 runs (seeds 1 and 2 still training):

| Model | MAE | Accuracy | QWK | ≥2 grades off |
|---|---|---|---|---|
| Small CNN (scratch) | 0.437 | 64.3% | 0.810 | 7.03% |
| ResNet18 + softmax | 0.192 | 82.3% | 0.926 | 1.08% |
| ResNet18 + ordinal | 0.208 | 80.1% | 0.926 | 0.76% |

Honest reading so far: transfer learning is the big win (MAE 0.44 → ~0.2). Softmax and ordinal
are close; ordinal makes **fewer severe (2+ grade) errors** but slightly more 1-grade errors.
Ordinal is better on avocado, softmax better on strawberry. Final verdict waits for 3 seeds.

## 2026-11-17 · Step 10 – Web app (`webapp/`)

* `webapp/server.py` – FastAPI. Loads the 5 fold models of each kind (seed 0).
  * Uploaded photo → average of all 5 models (ensemble).
  * Gallery sample → **only the model whose test fold contains it**, so the demo never shows a
    memorised answer. (Caught before release: the first version averaged all 5 models, 3 of which
    had trained on the sample.)
  * Returns ordinal probabilities, P(grade > k), softmax comparison, and a Grad-CAM overlay.
* `webapp/static/` – one page styled after the "Digital Brainstorm" template (cream paper,
  torn grid strips, sticky notes, tape, dashed arrows). Design rules from the *unslop* skill:
  no gradients/glows/emoji, motion only where it explains (pointer slides to the grade,
  bars fill to their probability, arrows draw on scroll, numbers count up).
* Bugs found while testing in the browser: result panel visible before any prediction
  (CSS overrode `hidden`); Grad-CAM hook crashed during normal no-grad inference; NaN std
  with one seed broke the JSON; mind-map arrows hidden under the notes. All fixed.

## 2026-11-17 · Step 11 – Presentation page (`/presentation`)

Replaced the "Syllabus map" section of the demo page with a link card to a 17-slide deck built on
the same template (`webapp/static/presentation.html/.css/.js`). Slides: title, agenda, problem,
dataset (real crop grid, `src/make_figures.py` → `results/dataset_grid.png`), boxes → crops diagram,
leakage "wrong ideas / best ideas" board with fold diagram, softmax vs ordinal, CORAL head
architecture diagram, model mind map, sanity checks (overfit chart from the numbers in Step 5),
results, error distances + 3 confusion matrices, training curves, Grad-CAM, findings, syllabus map,
references. Results and the findings text are filled from `/api/results`, so they update when the
last seeds finish – nothing is hard-coded about which model wins.
Navigation: arrow keys / space / dots. `?slide=N` opens one slide without animation
(used to screenshot every slide with headless Edge for checking).
Also removed the "CSE471 · Deep Learning for CV" chip from the pages (user request).

## 2026-11-17 · Step 12 – Final results (all 45 runs: 3 models × 5 folds × 3 seeds)

Pooled 5-fold grouped CV, n = 924 crops, mean ± std over 3 seeds (`results/summary.md`):

| Model | MAE ↓ | Accuracy ↑ | QWK ↑ | ≥2 grades off ↓ |
|---|---|---|---|---|
| Small CNN (scratch) | 0.422 ± 0.019 | 66.1 ± 2.0% | 0.811 ± 0.005 | 7.40 ± 0.31% |
| ResNet18 + softmax | 0.228 ± 0.045 | 79.0 ± 3.8% | 0.910 ± 0.022 | 1.37 ± 0.50% |
| ResNet18 + ordinal (CORAL) | 0.214 ± 0.030 | 79.5 ± 2.5% | 0.924 ± 0.013 | 0.79 ± 0.38% |

Reading: transfer learning roughly halves the error. The ordinal head is ahead of softmax on every
metric on average and is more stable across seeds (smaller std), with the biggest relative gain on
severe errors (0.79% vs 1.37%) and on strawberries (MAE 0.281 vs 0.309). The gaps are about one
standard deviation, so this is a consistent small edge, not a statistically proven win.

## 2026-11-17 · Step 13 – Deployment (Hugging Face, free)

* First attempt: Docker Space running the FastAPI server. **Refused (HTTP 402)**: Docker/Gradio
  Spaces now need a paid PRO plan; only static Spaces are free.
* Switched to a **static Space with the models running in the visitor's browser**
  (ONNX Runtime Web). `deploy/export_onnx.py` exports the fold-4 ordinal + softmax models.
  Fold 4 because its test fold has ≥10 crops of every fruit × grade and its training folds hold the
  most rotten strawberries; the 8 gallery samples are drawn from fold 4's test fold, so "the model
  never saw this photo" is still true.
* Grad-CAM moved into the ONNX graph without backprop: for s = w·GAP(A), dS/dA_c = w_c/49, so
  Grad-CAM = ReLU(Σ w_c A_c) after min-max normalisation.
* Parity checks: PyTorch vs ONNX logits max diff 3.8e-06; PyTorch Grad-CAM vs ONNX CAM 9.1e-06.
  Browser (canvas resize) vs PyTorch on the 8 samples: same grade 8/8, probabilities within 0.025.
* `deploy/build_static.py` generates the static site from `webapp/static` (paths made relative,
  `infer.js` added), so the local FastAPI version keeps working unchanged.
* Live: https://huggingface.co/spaces/AD074890/ripe-or-not
  (direct: https://ad074890-ripe-or-not.static.hf.space)
