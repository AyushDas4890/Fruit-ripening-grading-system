"""Extra figure for the presentation: real example crops for every fruit x grade.

    python src/make_figures.py   -> results/dataset_grid.png
"""
import os

import pandas as pd
from PIL import Image, ImageDraw

from common import CROPS, GRADES, MANIFEST, ROOT

m = pd.read_csv(MANIFEST)
S, PER = 180, 4                      # tile size, examples per cell
fruits = ["strawberry", "avocado"]
W, H = len(GRADES) * PER * S, len(fruits) * S
canvas = Image.new("RGB", (W, H), (250, 246, 239))
for r, fr in enumerate(fruits):
    for g in range(len(GRADES)):
        pick = m[(m.fruit == fr) & (m.grade == g)].sample(PER, random_state=7)
        for k, crop in enumerate(pick.crop):
            im = Image.open(os.path.join(CROPS, crop)).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((S - 8, S - 8))
            canvas.paste(im, ((g * PER + k) * S + 4, r * S + 4))
ImageDraw.Draw(canvas)
out = os.path.join(ROOT, "results", "dataset_grid.png")
canvas.save(out)
print("saved", out, canvas.size)
