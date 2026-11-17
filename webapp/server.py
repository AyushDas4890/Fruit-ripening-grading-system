"""Local web app server.

    python webapp/server.py        then open http://127.0.0.1:8000

Model used for the demo: the 5 cross-validation models of each kind (seed 0) are
averaged (an ensemble). Each one was trained on 3/5 of the data, so together they
have seen every training image. Grad-CAM is shown for the fold-0 ordinal model.
"""
import base64
import io
import json
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import GRADES, K, build_model, to_probs, transforms_for  # noqa: E402
from gradcam import GradCAM  # noqa: E402

torch.set_grad_enabled(False)
TF = transforms_for(False)


def load_kind(kind):
    """List indexed by fold. Needs all 5 folds trained, so index == fold is guaranteed."""
    nets = []
    for f in range(5):
        p = os.path.join(ROOT, "runs", f"{kind}_f{f}_s0", "best.pt")
        if not os.path.exists(p):
            sys.exit(f"missing {p} – run src/run_all.py first")
        m = build_model(kind)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        nets.append(m.eval())
    return nets


MODELS = {k: load_kind(k) for k in ("tl_ordinal", "tl_softmax")}
CAMS = [GradCAM(m) for m in MODELS["tl_ordinal"]]
print({k: len(v) for k, v in MODELS.items()}, "models loaded")

# Which fold is each crop's TEST fold? A sample picked from the gallery is graded only by
# the model that never trained on it, so the demo cannot show off memorised answers.
import pandas as pd  # noqa: E402

FOLD_OF = dict(pd.read_csv(os.path.join(ROOT, "data", "splits", "manifest.csv"))[["crop", "fold"]].values)

app = FastAPI()


def heat_png(img, heat):
    """Blend the Grad-CAM map onto the photo with a warm template-coloured ramp."""
    base = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255
    h = heat[..., None]
    warm = np.array([0.93, 0.08, 0.60])          # template magenta
    cool = np.array([0.64, 0.73, 0.98])          # template periwinkle
    col = cool * (1 - h) + warm * h
    out = base * (1 - 0.55 * h) + col * 0.55 * h
    buf = io.BytesIO()
    Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), crop: str = ""):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    img.thumbnail((512, 512))
    x = TF(img).unsqueeze(0)
    fold = FOLD_OF.get(crop)  # gallery sample -> its own held-out model; upload -> ensemble of all 5
    res = {"used": f"fold-{fold} model (never saw this photo)" if fold is not None else "ensemble of 5 models"}
    for kind, all_nets in MODELS.items():
        nets = [all_nets[fold]] if fold is not None and fold < len(all_nets) else all_nets
        if not nets:
            continue
        outs = [n(x) for n in nets]
        probs = torch.stack([to_probs(kind, o) for o in outs]).mean(0)[0]
        r = {"probs": [round(float(p), 4) for p in probs]}
        if kind == "tl_ordinal":
            p_gt = torch.stack([torch.sigmoid(o) for o in outs]).mean(0)[0]
            r["p_gt"] = [round(float(p), 4) for p in p_gt]          # P(grade > 0), > 1, > 2
            r["grade"] = int((p_gt > 0.5).sum())
            r["expected"] = round(float(p_gt.sum()), 3)                  # continuous ripeness 0..3
        else:
            r["grade"] = int(probs.argmax())
        res[kind] = r
    if CAMS:
        cam = CAMS[fold] if fold is not None and fold < len(CAMS) else CAMS[0]
        with torch.enable_grad():
            heat, _ = cam(x.clone().requires_grad_(True), "tl_ordinal")
        res["gradcam"] = heat_png(img, heat)
    res["grades"] = GRADES
    return JSONResponse(res)


@app.get("/api/results")
def results():
    p = os.path.join(ROOT, "results", "web_results.json")
    return JSONResponse(json.load(open(p)) if os.path.exists(p) else {})


app.mount("/results", StaticFiles(directory=os.path.join(ROOT, "results")), name="results")
app.mount("/crops", StaticFiles(directory=os.path.join(ROOT, "data", "crops")), name="crops")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/presentation")
def presentation():
    return FileResponse(os.path.join(HERE, "static", "presentation.html"))


if __name__ == "__main__":
    # local default 127.0.0.1:8000; Hugging Face Spaces sets HOST=0.0.0.0 PORT=7860
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
