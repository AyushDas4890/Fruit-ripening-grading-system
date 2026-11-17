"""Stage exactly what the Hugging Face Space needs into deploy/space/, then (optionally) upload.

    python deploy/prepare_space.py                      # stage only
    python deploy/prepare_space.py --push USER/SPACE    # stage + create/upload the Docker Space

Staged: src/common.py + gradcam.py, webapp/, the seed-0 model weights (10 x ~45 MB),
data/crops (for the sample gallery), data/splits/manifest.csv, results/.
Not staged: the 2 GB raw dataset, other seeds' weights, logs.
"""
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deploy", "space")

DOCKERFILE = """FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \\
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HOST=0.0.0.0 PORT=7860 TORCH_HOME=/app/.torch
EXPOSE 7860
CMD ["python", "webapp/server.py"]
"""
REQS = "fastapi==0.115.6\nuvicorn==0.34.0\npython-multipart==0.0.20\npillow==11.0.0\nnumpy==2.1.3\npandas==2.2.3\nscikit-learn==1.6.0\nmatplotlib==3.9.3\n"
README = """---
title: Ripe or Not
emoji: 🍓
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
---

Fruit ripeness grading (strawberry, avocado) via ordinal regression (CORAL) on ResNet18.
CSE471 course project. Live demo at `/`, slides at `/presentation`.
Data: Kamat et al. 2024, Mendeley Data, DOI 10.17632/zysvgmxcyz.1, CC BY 4.0.
"""


def cp(rel):
    src, dst = os.path.join(ROOT, rel), os.path.join(OUT, rel)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "artifact"))
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", help="USER/SPACE to create and upload to")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for rel in ["src/common.py", "src/gradcam.py", "webapp/server.py", "webapp/static", "data/crops",
                "data/splits/manifest.csv", "results"]:
        cp(rel)
    for kind in ("tl_ordinal", "tl_softmax"):
        for f in range(5):
            cp(f"runs/{kind}_f{f}_s0/best.pt")
    # the server builds ResNet18 with ImageNet weights before loading ours; ship that file too
    # so the Space never downloads at startup
    hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "resnet18-f37072fd.pth")
    dst = os.path.join(OUT, ".torch", "hub", "checkpoints")
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(hub, dst)
    open(os.path.join(OUT, "Dockerfile"), "w").write(DOCKERFILE)
    open(os.path.join(OUT, "requirements.txt"), "w").write(REQS)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"staged {OUT}  ({size / 1e6:.0f} MB)")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=OUT, repo_id=a.push, repo_type="space",
                          commit_message="Deploy Ripe or Not")
        print(f"pushed -> https://huggingface.co/spaces/{a.push}")


if __name__ == "__main__":
    main()
