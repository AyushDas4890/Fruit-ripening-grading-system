"""Grad-CAM: where does the model look?  [Unit VI – interpretability]

Grad-CAM in 4 steps (written by hand, no library):
  1. forward an image, keep the activations A of the last conv block (layer4: 512 maps of 7x7);
  2. backpropagate the score we want to explain, keep the gradients dS/dA;
  3. weight of map c = average of its gradient over the 7x7 grid;
  4. heatmap = ReLU( sum_c weight_c * A_c ), upsampled to the image size.

Score explained: for the ordinal model the single 'ripeness score' s = w·x (shared by all
thresholds), so the heatmap shows what pushes the fruit towards 'more ripe/rotten'.
For softmax it is the logit of the predicted class.

    python src/gradcam.py            -> results/gradcam_<model>.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CROPS, GRADES, ROOT, build_model, predict, transforms_for


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        model.layer4.register_forward_hook(self._keep)

    def _keep(self, module, inp, out):
        # keep the activations, and ask autograd to hand us their gradient on backward
        self.act = out
        if out.requires_grad:  # plain inference (no_grad) passes through untouched
            out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, kind):
        out = self.model(x)
        if kind == "tl_ordinal":
            score = self.model.fc.fc(F.adaptive_avg_pool2d(self.act, 1).flatten(1)).sum()
        else:
            score = out[0, out.argmax(1)].sum()
        self.model.zero_grad()
        score.backward()
        w = self.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), int(predict(kind, out.detach())[0])


def load(kind, fold=0, seed=0):
    m = build_model(kind)
    m.load_state_dict(torch.load(os.path.join(ROOT, "runs", f"{kind}_f{fold}_s{seed}", "best.pt"), map_location="cpu"))
    return m


def main():
    preds = pd.read_csv(os.path.join(ROOT, "runs", "tl_ordinal_f0_s0", "test_preds.csv"))
    # one example per (fruit, grade) from the fold-0 test set -> the model never saw these
    ex = preds.groupby(["fruit", "grade"]).head(1).sort_values(["fruit", "grade"]).reset_index(drop=True)
    tf = transforms_for(False)
    for kind in ["tl_ordinal", "tl_softmax"]:
        cam = GradCAM(load(kind))
        fig, axes = plt.subplots(2, len(ex), figsize=(2.1 * len(ex), 4.6))
        for j, r in ex.iterrows():
            img = Image.open(os.path.join(CROPS, r.crop)).convert("RGB")
            x = tf(img).unsqueeze(0).requires_grad_(True)
            heat, p = cam(x, kind)
            show = np.asarray(img.resize((224, 224))) / 255
            axes[0, j].imshow(show); axes[0, j].set_title(f"{r.fruit}\ntrue: {GRADES[r.grade]}", fontsize=7)
            axes[1, j].imshow(show); axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
            axes[1, j].set_title(f"pred: {GRADES[p]}", fontsize=7, color="green" if p == r.grade else "red")
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle(f"Grad-CAM – {kind}", fontsize=10); fig.tight_layout()
        out = os.path.join(ROOT, "results", f"gradcam_{kind}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
