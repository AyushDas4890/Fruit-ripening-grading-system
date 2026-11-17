"""Step 2: turn the raw YOLO-format dataset into per-fruit crops + a leak-free split.

Raw data: full photos (data/raw/extracted/images) + YOLO box labels
(data/raw/extracted/labels, one line per fruit: class cx cy w h, all relative 0..1).

Class ids (worked out by viewing crops, the dataset ships no names file):
    0..3 = strawberry  unripe, partially ripe, ripe, rotten
    4..7 = avocado     unripe, partially ripe, ripe, rotten
So   fruit = cls // 4   and   ripeness grade = cls % 4   (0 < 1 < 2 < 3).

Leakage guard: the dataset contains augmented copies (same fruit pasted on a new
background, flipped, ...). If a copy lands in train and its twin in test, test
accuracy is inflated. We therefore
  1. put all crops from the same photo in one group, and
  2. merge groups whose crops look near-identical (perceptual hash, flip-aware),
then split by group with StratifiedGroupKFold.

Usage:  python src/prepare_data.py
"""
import csv
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "extracted")
CROPS = os.path.join(ROOT, "data", "crops")
SPLITS = os.path.join(ROOT, "data", "splits")
FRUITS = ["strawberry", "avocado"]
STAGES = ["unripe", "partially_ripe", "ripe", "rotten"]
PAD = 0.08          # extra border around each box (fraction of box size)
CROP_SIZE = 256     # saved crop size (longest side)
HASH_SIZE = 16      # dHash grid -> 16*16 = 256-bit fingerprint
HASH_THRESH = 40    # max Hamming distance (of 256 bits) to call two crops duplicates
BLOCK = 10          # photo-number block size for session grouping
N_TRIES = 200       # candidate fold assignments; keep the one with the most even class mix
N_FOLDS = 5
SEED = 42


def dhash(img, size=HASH_SIZE):
    """Difference hash of the crop's centre (the fruit, not the background).

    A first version hashed the whole crop at 8x8; it chained 93 *different* dark
    avocados into one group, so we hash only the central 60% at 16x16."""
    W, H = img.size
    img = img.crop((int(W * .2), int(H * .2), int(W * .8), int(H * .8)))
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    os.makedirs(CROPS, exist_ok=True)
    os.makedirs(SPLITS, exist_ok=True)
    rows, hashes = [], []
    images = sorted(os.listdir(os.path.join(RAW, "images")))
    for fname in images:
        stem = os.path.splitext(fname)[0]
        img = ImageOps.exif_transpose(Image.open(os.path.join(RAW, "images", fname))).convert("RGB")
        W, H = img.size
        lines = open(os.path.join(RAW, "labels", stem + ".txt")).read().split("\n")
        for k, line in enumerate(l.split() for l in lines):
            if len(line) != 5:
                continue
            cls = int(line[0])
            cx, cy, w, h = map(float, line[1:])
            w, h = w * (1 + 2 * PAD), h * (1 + 2 * PAD)
            box = (max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H)),
                   min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H)))
            if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                continue  # tiny box, unusable
            crop = img.crop(box)
            crop.thumbnail((CROP_SIZE, CROP_SIZE))
            out = f"{stem}_{k}.jpg"
            crop.save(os.path.join(CROPS, out), quality=92)
            rows.append(dict(crop=out, source=fname, cls=cls,
                             fruit=FRUITS[cls // 4], grade=cls % 4))
            # flip-invariant fingerprints (original + horizontal mirror)
            hashes.append((dhash(crop), dhash(ImageOps.mirror(crop))))

    # --- group: same source photo, same shooting session, or near-identical crop ---
    n = len(rows)
    uf = UnionFind(n)
    first = {}
    for i, r in enumerate(rows):
        uf.union(i, first.setdefault(r["source"], i))
        # Burst shots of one fruit have nearby photo numbers (seen: IMG_1147/1148,
        # IMG_1222/1234). Hashing cannot separate them from different fruits, so we
        # conservatively keep each block of BLOCK consecutive numbers of one class together.
        num = int("".join(ch for ch in r["source"] if ch.isdigit()))
        uf.union(i, first.setdefault(("block", r["cls"], num // BLOCK), i))
    H0 = np.stack([h[0] for h in hashes])
    H1 = np.stack([h[1] for h in hashes])
    dup_pairs = 0
    for i in range(n):
        d = np.minimum((H0[i] != H0).sum(1), (H0[i] != H1).sum(1))
        for j in np.where(d <= HASH_THRESH)[0]:
            if j > i and rows[j]["cls"] == rows[i]["cls"]:
                uf.union(i, j)
                dup_pairs += 1
    groups = [uf.find(i) for i in range(n)]

    # --- split by group, stratified by class: ~70 / 15 / 15 ---
    y = np.array([r["cls"] for r in rows])
    g = np.array(groups)
    idx = np.arange(n)
    target = np.bincount(y, minlength=8) / n

    # 5-fold grouped cross-validation. With only ~900 crops a single 15% test set
    # left some classes with 2 examples, so instead every crop is tested exactly once:
    # run k uses fold k as test, fold (k+1)%5 as validation (early stopping), rest train.
    def imbalance(fold):
        # how far each fold's class mix is from the overall mix (label counts only,
        # no model involved, so choosing on this cannot bias the test results)
        return sum(np.abs(np.bincount(y[fold == k], minlength=8) / max(1, (fold == k).sum()) - target).sum()
                   for k in range(N_FOLDS))

    best = None
    for seed in range(SEED, SEED + N_TRIES):
        fold = np.empty(n, dtype=int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, y, g)):
            fold[te] = k
        score = imbalance(fold)
        if best is None or score < best[0]:
            best = (score, seed, fold)
    _, split_seed, fold = best
    print(f"chosen fold seed {split_seed} (class-mix deviation {best[0]:.3f} over {N_TRIES} tries)")

    with open(os.path.join(SPLITS, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["crop", "source", "cls", "fruit", "grade", "group", "fold"])
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({**r, "group": groups[i], "fold": int(fold[i])})

    # --- report + leakage self-check ---
    print(f"photos: {len(images)}  crops: {n}  near-duplicate pairs merged: {dup_pairs}  groups: {len(set(groups))}")
    for k in range(N_FOLDS):
        c = Counter((rows[i]["fruit"], rows[i]["grade"]) for i in range(n) if fold[i] == k)
        print(f"fold {k}", sum(c.values()), {f"{fr[:5]}-{gr}": c[(fr, gr)] for fr in FRUITS for gr in range(4)})
    for a in range(N_FOLDS):
        for b in range(a + 1, N_FOLDS):
            assert not (set(g[fold == a]) & set(g[fold == b]))
    print("leakage check passed: no group appears in two folds")


if __name__ == "__main__":
    main()
