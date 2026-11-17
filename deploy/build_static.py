"""Build the FREE static Hugging Face Space (no server: models run in the browser).

    python deploy/export_onnx.py                          # once: ONNX models + parity check
    python deploy/build_static.py                         # build deploy/static_space/
    python deploy/build_static.py --push USER/SPACE       # build + upload (needs HF_TOKEN)

Pages are generated from webapp/static (no hand-made copies): absolute server paths become
relative, infer.js + onnxruntime-web are added, results load from results/web_results.json.
Gallery samples are drawn from fold 4's TEST fold, because the shipped models are fold 4's.
"""
import argparse
import json
import os
import re
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "webapp", "static")
OUT = os.path.join(ROOT, "deploy", "static_space")
FOLD = 4
ORT = '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>'
README = """---
title: Ripe or Not
colorFrom: pink
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
short_description: Fruit ripeness grading with ordinal regression (CORAL)
---

Strawberry and avocado ripeness grading (unripe < partially ripe < ripe < rotten) with an
ordinal (CORAL) head on ResNet18. CSE471 course project.

The models run in your browser (ONNX Runtime Web), so this Space needs no server.
Live demo: `index.html` · slides: `presentation.html`.

Data: Kamat, Chandekar, Dlima, Gite, Pradhan, Alamri (2024). Ripening Stages of Strawberries
and Avocados. Mendeley Data, DOI 10.17632/zysvgmxcyz.1, CC BY 4.0.
"""


def rewrite(text):
    text = text.replace('"/static/', '"').replace("'/static/", "'")
    text = text.replace('"/results/', '"results/').replace("/crops/${", "crops/${")
    text = text.replace('href="/presentation"', 'href="presentation.html"').replace('href="/"', 'href="index.html"')
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "models"))

    for f in ["style.css", "presentation.css", "app.js", "presentation.js", "infer.js"]:
        open(os.path.join(OUT, f), "w", encoding="utf-8").write(rewrite(open(os.path.join(SRC, f), encoding="utf-8").read()))
    cfg = '<script>window.RESULTS_URL = "results/web_results.json";</script>'
    idx = rewrite(open(os.path.join(SRC, "index.html"), encoding="utf-8").read())
    idx = idx.replace('<script src="app.js"></script>', f'{cfg}\n{ORT}\n<script src="infer.js"></script>\n<script src="app.js"></script>')
    idx = idx.replace("or pick a test photo the model never trained on", "or pick a test photo the model never trained on · the model runs in your browser")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(idx)
    pres = rewrite(open(os.path.join(SRC, "presentation.html"), encoding="utf-8").read())
    pres = pres.replace('<script src="presentation.js"></script>', f'{cfg}\n<script src="presentation.js"></script>')
    open(os.path.join(OUT, "presentation.html"), "w", encoding="utf-8").write(pres)
    for name in ("index.html", "presentation.html", "app.js", "presentation.js"):
        left = re.findall(r'["\'`](/(?:static|results|crops|api)[^"\'`]*)', open(os.path.join(OUT, name), encoding="utf-8").read())
        # server fallbacks, unused here: RESULTS_URL is set and infer.js defines localPredict
        left = [x for x in left if x not in ("/api/results", "/api/predict")]
        assert not left, f"server path left in {name}: {left}"

    # results (figures + JSON), with gallery samples from fold 4's test fold
    shutil.copytree(os.path.join(ROOT, "results"), os.path.join(OUT, "results"),
                    ignore=shutil.ignore_patterns("*.csv", "*.md"))
    m = pd.read_csv(os.path.join(ROOT, "data", "splits", "manifest.csv"))
    pick = m[m.fold == FOLD].sample(frac=1, random_state=3).groupby(["fruit", "grade"]).head(1)
    pick = pick.sort_values(["fruit", "grade"])[["crop", "fruit", "grade"]].to_dict("records")
    for s in pick:
        s["grade"] = int(s["grade"])
    rj = os.path.join(OUT, "results", "web_results.json")
    R = json.load(open(rj))
    R["samples"] = pick
    json.dump(R, open(rj, "w"), indent=1)
    os.makedirs(os.path.join(OUT, "crops"))
    for s in pick:
        shutil.copy2(os.path.join(ROOT, "data", "crops", s["crop"]), os.path.join(OUT, "crops"))
    for f in ("ordinal.onnx", "softmax.onnx"):
        shutil.copy2(os.path.join(ROOT, "deploy", "onnx", f), os.path.join(OUT, "models"))
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"built {OUT} ({size / 1e6:.0f} MB), samples: {[s['crop'] for s in pick]}")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=OUT, repo_id=a.push, repo_type="space", commit_message="Deploy Ripe or Not (static, in-browser models)")
        print(f"pushed -> https://huggingface.co/spaces/{a.push}")


if __name__ == "__main__":
    main()
