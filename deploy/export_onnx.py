"""Export the fold-4 (seed 0) ordinal + softmax models to ONNX for in-browser inference,
and check they match PyTorch.

    python deploy/export_onnx.py

Why fold 4: its test fold has >= 10 crops of every fruit x grade, and its training folds hold
the most rotten strawberries. The web demo's gallery samples all come from fold 4's test fold,
so "the model never saw this photo" stays true with a single model.

Grad-CAM inside the graph: for the ordinal model the explained score is s = w . GAP(A).
dS/dA_c = w_c / 49 everywhere, so the Grad-CAM channel weight is w_c / 49 and
CAM = ReLU(sum_c w_c A_c) up to a constant, which min-max normalisation removes.
No backprop is needed, so the browser can compute it in the forward pass.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import CROPS, build_model, transforms_for  # noqa: E402
from gradcam import GradCAM  # noqa: E402

FOLD = 4
OUT = os.path.join(ROOT, "deploy", "onnx")
os.makedirs(OUT, exist_ok=True)


def load(kind):
    m = build_model(kind)
    m.load_state_dict(torch.load(os.path.join(ROOT, "runs", f"{kind}_f{FOLD}_s0", "best.pt"), map_location="cpu"))
    return m.eval()


class OrdinalWithCAM(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        a = n.layer4(n.layer3(n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))))
        logits = n.fc(torch.flatten(F.adaptive_avg_pool2d(a, 1), 1))       # (1,3)
        w = n.fc.fc.weight.view(1, -1, 1, 1)                                  # (1,512,1,1)
        cam = F.relu((a * w).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
        lo = cam.amin(dim=(1, 2), keepdim=True)
        hi = cam.amax(dim=(1, 2), keepdim=True)
        return logits, (cam - lo) / (hi - lo + 1e-8)


def export(module, path, names):
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(module, (x,), path, input_names=["input"], output_names=names, opset_version=17, dynamo=False)
    print("exported", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


ordm, softm = load("tl_ordinal"), load("tl_softmax")
export(OrdinalWithCAM(ordm).eval(), os.path.join(OUT, "ordinal.onnx"), ["logits", "cam"])
export(softm, os.path.join(OUT, "softmax.onnx"), ["logits"])

# ---- parity check on real held-out crops
so = ort.InferenceSession(os.path.join(OUT, "ordinal.onnx"))
ss = ort.InferenceSession(os.path.join(OUT, "softmax.onnx"))
cam_ref = GradCAM(load("tl_ordinal"))
tf = transforms_for(False)
m = pd.read_csv(os.path.join(ROOT, "data", "splits", "manifest.csv"))
crops = m[m.fold == FOLD].sample(6, random_state=0).crop
worst = dict(logit=0.0, cam=0.0)
for c in crops:
    x = tf(Image.open(os.path.join(CROPS, c)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        t_ord, t_soft = ordm(x).numpy(), softm(x).numpy()
    o_ord, o_cam = so.run(None, {"input": x.numpy()})
    o_soft = ss.run(None, {"input": x.numpy()})[0]
    ref, _ = cam_ref(x.clone().requires_grad_(True), "tl_ordinal")
    worst["logit"] = max(worst["logit"], np.abs(t_ord - o_ord).max(), np.abs(t_soft - o_soft).max())
    worst["cam"] = max(worst["cam"], np.abs(ref - o_cam[0]).max())
print(f"max |torch - onnx| logits: {worst['logit']:.2e}   max |Grad-CAM(torch) - CAM(onnx)|: {worst['cam']:.2e}")
assert worst["logit"] < 1e-3 and worst["cam"] < 1e-2, "parity check failed"
print("parity OK")
