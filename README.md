<p align="center">
  <img src="docs/assets/banner.svg" alt="Ripe or Not – fruit ripeness grading via ordinal regression" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/AD074890/ripe-or-not"><img alt="Live demo" src="https://img.shields.io/badge/live%20demo-try%20it%20now-EC0B9C?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://ad074890-ripe-or-not.static.hf.space/presentation.html"><img alt="Slides" src="https://img.shields.io/badge/slides-13%20slide%20deck-F9C34B?style=for-the-badge&logoColor=black"></a>
  <a href="docs/01_journal.md"><img alt="Process journal" src="https://img.shields.io/badge/process-journal-A3B8FA?style=for-the-badge"></a>
</p>

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.14-1A1A1A?logo=pytorch">
  <img alt="ResNet18" src="https://img.shields.io/badge/backbone-ResNet18-1A1A1A">
  <img alt="CORAL" src="https://img.shields.io/badge/head-CORAL%20ordinal-1A1A1A">
  <img alt="ONNX" src="https://img.shields.io/badge/runs%20in-your%20browser%20(ONNX)-1A1A1A?logo=onnx">
  <img alt="Data" src="https://img.shields.io/badge/data-CC%20BY%204.0-1A1A1A">
</p>

<p align="center"><b>Show it a strawberry or an avocado. It grades ripeness on an ordered scale<br>
<code>0 unripe → 1 partially ripe → 2 ripe → 3 rotten</code> and shows where it looked.</b></p>

---

## 🍓 Live demos

| | Link | What you can do |
|---|---|---|
| **Demo** | [huggingface.co/spaces/AD074890/ripe-or-not](https://huggingface.co/spaces/AD074890/ripe-or-not) | Upload a fruit photo or click a test photo, then watch the ripeness ruler, the 3 yes/no questions and the Grad-CAM heatmap |
| **Slides** | [presentation.html](https://ad074890-ripe-or-not.static.hf.space/presentation.html) | 13 slides covering the whole project. Use ← → keys or the dots |

> The models run **inside your browser** (ONNX Runtime Web), so the demo needs no server.
> The first prediction downloads about 90 MB of model files, which your browser then caches.

<p align="center">
  <a href="https://huggingface.co/spaces/AD074890/ripe-or-not"><img src="docs/assets/demo_result.jpg" alt="Demo: a rotten avocado graded 3/3 with the three yes/no answers and Grad-CAM" width="62%"></a>
</p>

---

## 💡 The idea: ripeness is a ladder, not a list

Normal classification (softmax) treats the four stages as **unrelated boxes**. Calling a rotten fruit
*unripe* costs the same as calling it *ripe*. But the stages are **ordered**, and a mistake of 3 grades
is much worse than a mistake of 1.

**Ordinal regression (CORAL)** turns one 4-way choice into **3 yes/no questions** that all share a single
ripeness score:

<p align="center"><img src="docs/assets/ruler.svg" alt="Animated: the pointer moves along the ripeness ruler and the matching yes/no questions turn green" width="85%"></p>

| | grade 0 | grade 1 | grade 2 | grade 3 |
|---|:-:|:-:|:-:|:-:|
| grade > 0 ? | 0 | **1** | **1** | **1** |
| grade > 1 ? | 0 | 0 | **1** | **1** |
| grade > 2 ? | 0 | 0 | 0 | **1** |

```text
s = w · features              one shared "ripeness score" (shared w → answers can never contradict)
P(grade > k) = σ(s + b_k)     three learned thresholds b0 > b1 > b2
loss = Σ_k BCE                predicted grade = number of answers above 0.5
```

<p align="center"><img src="docs/assets/slide_8.jpg" alt="Slide: the CORAL head architecture" width="80%"></p>

---

## 🛠️ Pipeline

```mermaid
flowchart LR
    A["758 real photos<br/>YOLO boxes, 8 classes"] --> B["924 fruit crops<br/>+8% border, 224×224"]
    B --> C["Leak-free grouping<br/>same photo · burst shots · near-duplicates"]
    C --> D["5-fold grouped CV<br/>each crop tested once"]
    D --> E1["Small CNN<br/>from scratch"]
    D --> E2["ResNet18<br/>+ softmax"]
    D --> E3["ResNet18<br/>+ ordinal (CORAL)"]
    E1 & E2 & E3 --> F["MAE · accuracy · QWK<br/>severe errors · Grad-CAM"]
    E3 --> G["ONNX → runs in the browser<br/>(Hugging Face Space)"]
```

<details>
<summary><b>Training recipe</b> (click to open)</summary>

| Setting | Value | Syllabus |
|---|---|---|
| Backbone | ResNet18, ImageNet weights | Unit III |
| Phase 1 | 3 epochs, backbone **frozen**, only the head learns | Unit III |
| Phase 2 | everything **unfrozen**, backbone LR 10× lower than the head (layer-wise LR) | Unit III |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule | Unit II |
| Early stopping | on validation MAE, patience 7, max 30 epochs | Unit II |
| Augmentation | random crop, flips, ±20° rotation, brightness/contrast. **Never hue**, because colour is the label | Unit II |
| Ordinal thresholds | start at [2, 0, −2], 10× learning rate (a bug fix, see the journal) | beyond |
| Hardware | RTX 3050 Laptop, 4 GB, mixed precision (AMP) | – |

</details>

<details>
<summary><b>Keeping the test honest</b> (click to open)</summary>

The dataset contains burst shots and copies of the same fruit pasted on new backgrounds. If one copy lands
in training and its twin in testing, the score is inflated. So crops are **grouped** first:

* same source photo, **or**
* same class with photo numbers in one block of 10 (burst shots), **or**
* centre-crop perceptual hash distance ≤ 40 of 256 bits

That gives 159 groups, split into 5 folds with `StratifiedGroupKFold`. Run *k* tests on fold *k*,
validates on fold *k+1* and trains on the other three. **Every crop is tested exactly once, by a model
that never saw it.** The two ideas that failed (a whole-crop hash that merged 93 different avocados, and a
70/15/15 split with only 2 ripe avocados in the test set) are written up in the [journal](docs/01_journal.md).

</details>

---

## 📊 Results

All 45 runs (3 models × 5 folds × 3 seeds). 924 crops pooled. Mean ± std over seeds.

| Model | MAE ↓ | Accuracy ↑ | QWK ↑ | ≥ 2 grades off ↓ |
|---|:-:|:-:|:-:|:-:|
| Small CNN (from scratch, softmax) | 0.422 ± 0.019 | 66.1 ± 2.0 % | 0.811 ± 0.005 | 7.40 ± 0.31 % |
| ResNet18 + softmax | 0.228 ± 0.045 | 79.0 ± 3.8 % | 0.910 ± 0.022 | 1.37 ± 0.50 % |
| **ResNet18 + ordinal (CORAL)** | **0.214 ± 0.030** | **79.5 ± 2.5 %** | **0.924 ± 0.013** | **0.79 ± 0.38 %** |

* **Transfer learning roughly halves the error** (MAE 0.42 → 0.21).
* **The ordinal head is ahead of softmax on every metric** and steadier across seeds. Its biggest gain is
  on severe mistakes (0.79 % vs 1.37 %) and on strawberries (MAE 0.281 vs 0.309).
* The gaps are about one standard deviation, so read this as a **consistent small edge**, not a proven win.

<p align="center">
  <img src="results/error_distance.png" alt="Share of predictions that are 1, 2 or 3 grades off, per model" width="49%">
  <img src="results/confusion_tl_ordinal.png" alt="Confusion matrix of the ordinal model" width="34%">
</p>

<details>
<summary><b>Per-fruit results and training curves</b></summary>

| Fruit | Model | MAE | Accuracy | QWK | ≥ 2 off |
|---|---|:-:|:-:|:-:|:-:|
| avocado | CNN | 0.367 | 68.9 % | 0.856 | 5.3 % |
| avocado | ResNet18 + softmax | 0.178 | 82.9 % | 0.940 | 0.7 % |
| avocado | ResNet18 + ordinal | 0.173 | 83.0 % | 0.945 | 0.3 % |
| strawberry | CNN | 0.513 | 61.5 % | 0.698 | 10.9 % |
| strawberry | ResNet18 + softmax | 0.309 | 72.6 % | 0.840 | 2.5 % |
| strawberry | ResNet18 + ordinal | 0.281 | 73.9 % | 0.876 | 1.6 % |

<img src="results/curves_tl_ordinal.png" alt="Training curves of the ordinal model, fold 0" width="70%">

</details>

---

## 🔍 Where does it look? Grad-CAM

Written by hand with PyTorch hooks ([`src/gradcam.py`](src/gradcam.py)). For the ordinal model we explain
the shared ripeness score, so red means *evidence for riper*.

<p align="center"><img src="results/gradcam_tl_ordinal.png" alt="Grad-CAM of the ordinal model on one test image per fruit and grade" width="100%"></p>

* Ripe and rotten fruit: the heat sits on dark skin, mould and red flesh, which is the right evidence.
* Unripe fruit: there is little "riper" evidence, so the map drifts to the background. That is a real weakness.

> In the browser version, Grad-CAM is computed inside the ONNX graph without backprop. For
> s = w·GAP(A), ∂s/∂A_c = w_c/49, so Grad-CAM = ReLU(Σ w_c A_c) after normalising. The browser version
> matches PyTorch to within 1e-5.

---

## 🎞️ The presentation

<p align="center">
  <a href="https://ad074890-ripe-or-not.static.hf.space/presentation.html"><img src="docs/assets/slides_grid.jpg" alt="Six slides from the 13-slide deck" width="100%"></a>
</p>

---

## 📚 Syllabus map (CSE471 – Deep Learning for Computer Vision)

| Unit | Where it shows up in this project |
|---|---|
| **I** Foundations | image representation, resizing, normalisation, colour as the signal, loss functions, optimisation |
| **II** CNNs & training | CNN from scratch, BatchNorm, dropout, augmentation, LR scheduling, early stopping |
| **III** Transfer learning | pretrained ResNet18, freeze → unfreeze, layer-wise learning rates |
| **IV** Detection | YOLO-format bounding boxes, used to crop every fruit |
| **VI** Interpretability | Grad-CAM, written by hand |
| **Beyond** | ordinal regression (CORAL), QWK metric, grouped cross-validation, in-browser ONNX inference |

---

## 🗂️ Repository

```text
src/            prepare_data · common (models, CORAL head, metrics) · train · run_all
                evaluate · gradcam · sanity_checks · make_figures
webapp/         FastAPI server + the notebook-style site (demo + presentation)
deploy/         export_onnx (ONNX + parity check) · build_static (free HF Space)
data/crops/     924 fruit crops     data/splits/manifest.csv   leak-free folds
runs/           per-run config, epoch logs and test predictions (weights not committed)
results/        tables, figures, web_results.json
docs/           00_project_overview · 01_journal (every step, command, output and decision)
```

<details>
<summary><b>Reproduce everything</b></summary>

```bash
pip install torch torchvision fastapi uvicorn python-multipart pillow pandas scikit-learn matplotlib onnx onnxruntime onnxscript

# 1. data: download the Mendeley zip into data/raw/ and extract images/ + labels/ to data/raw/extracted/
python src/prepare_data.py     # crops + leak-free 5 folds
python src/sanity_checks.py    # plumbing checks before training
python src/run_all.py          # 45 training runs (resumable)
python src/evaluate.py         # tables + figures -> results/
python src/gradcam.py          # Grad-CAM figures
python webapp/server.py        # local demo at http://127.0.0.1:8000

# free deployment (models in the browser)
python deploy/export_onnx.py
python deploy/build_static.py --push USER/SPACE
python deploy/build_pptx.py <folder of slide captures>   # PowerPoint version
```

Model weights (`runs/**/best.pt`, 45 × 45 MB) are not committed. Retrain with `src/run_all.py`, or use the
ONNX models served by the [Space](https://huggingface.co/spaces/AD074890/ripe-or-not/tree/main/models).

</details>

---

## 📖 Process journal

Everything was recorded as it happened, including the failures: [`docs/01_journal.md`](docs/01_journal.md).
Highlights:

* the class ids had no names file, so they were decoded by looking at crops;
* two leak-prevention ideas failed before the final grouping;
* the ordinal head could not memorise one batch, and the fix was wider thresholds with a 10× LR;
* paid Docker hosting was refused (HTTP 402), so the app moved to free in-browser ONNX inference.

## Credits

**Data:** Kamat, P., Chandekar, H., Dlima, L., Gite, S., Pradhan, B., Alamri, A. (2024). *Comprehensive Dataset on
Ripening Stages of Strawberries and Avocados: From Unripe to Rotten.* Mendeley Data, V1.
DOI [10.17632/zysvgmxcyz.1](https://doi.org/10.17632/zysvgmxcyz.1). CC BY 4.0. The crops in `data/crops/` are derived from it.

**Method:** Cao, Mirjalili, Raschka (2020). *Rank consistent ordinal regression for neural networks.* ·
Selvaraju et al. (2017). *Grad-CAM.* · He et al. (2016). *ResNet.*

**Design:** styled after a "Digital Brainstorm" notebook template.
