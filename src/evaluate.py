"""Pool test predictions of all runs and produce the final tables + figures.

    python src/evaluate.py

For each model and seed, the 5 test folds together cover every crop exactly once.
We compute metrics on that pooled set (one number per seed), then report the
mean ± std over the 3 seeds.

Outputs in results/:
    summary.csv / summary.md      main table (MAE, accuracy, QWK, severe-error rate)
    per_fruit.csv                 same metrics split by fruit
    confusion_<model>.png         confusion matrices (seed 0, pooled folds)
    curves_<model>.png            training curves of one example run
    error_distance.png            how far off the wrong predictions are
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from common import GRADES, ROOT, metrics

RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")
NAMES = {"cnn": "Small CNN (scratch, softmax)", "tl_softmax": "ResNet18 transfer + softmax",
         "tl_ordinal": "ResNet18 transfer + ordinal (CORAL)"}
os.makedirs(OUT, exist_ok=True)


def pooled(model, seed):
    files = sorted(glob.glob(os.path.join(RUNS, f"{model}_f*_s{seed}", "test_preds.csv")))
    return pd.concat([pd.read_csv(f) for f in files]) if len(files) == 5 else None


rows, fruit_rows, pools = [], [], {}
for model in NAMES:
    for seed in range(3):
        df = pooled(model, seed)
        if df is None:
            continue
        pools[(model, seed)] = df
        rows.append(dict(model=model, seed=seed, n=len(df), **metrics(df.grade, df.pred)))
        for fr, d in df.groupby("fruit"):
            fruit_rows.append(dict(model=model, seed=seed, fruit=fr, **metrics(d.grade, d.pred)))

res = pd.DataFrame(rows)
summ = res.groupby("model")[["mae", "acc", "qwk", "severe"]].agg(["mean", "std"]).reindex(list(NAMES))
summ.to_csv(os.path.join(OUT, "summary.csv"))
fr = pd.DataFrame(fruit_rows).groupby(["fruit", "model"])[["mae", "acc", "qwk", "severe"]].mean()
fr.to_csv(os.path.join(OUT, "per_fruit.csv"))

with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
    f.write(f"Pooled 5-fold grouped CV, mean ± std over {res.seed.nunique()} seeds, n = {res.n.iloc[0]} crops\n\n")
    f.write("| Model | MAE ↓ | Accuracy ↑ | QWK ↑ | Severe errors (≥2 grades) ↓ |\n|---|---|---|---|---|\n")
    for m in NAMES:
        if m not in summ.index or summ.loc[m].isna().all():
            continue
        s = summ.loc[m]
        f.write(f"| {NAMES[m]} | {s[('mae','mean')]:.3f} ± {s[('mae','std')]:.3f} | "
                f"{s[('acc','mean')]*100:.1f} ± {s[('acc','std')]*100:.1f}% | "
                f"{s[('qwk','mean')]:.3f} ± {s[('qwk','std')]:.3f} | "
                f"{s[('severe','mean')]*100:.2f} ± {s[('severe','std')]*100:.2f}% |\n")
    f.write("\nPer fruit (mean over seeds)\n\n" + fr.round(3).to_markdown() + "\n")
print(open(os.path.join(OUT, "summary.md"), encoding="utf-8").read())

# confusion matrices (seed 0)
for m in NAMES:
    if (m, 0) not in pools:
        continue
    d = pools[(m, 0)]
    cm = confusion_matrix(d.grade, d.pred, labels=range(4))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="RdPu")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), GRADES, rotation=30); ax.set_yticks(range(4), GRADES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(NAMES[m], fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"confusion_{m}.png"), dpi=150); plt.close(fig)

# error distance histogram (all seeds)
fig, ax = plt.subplots(figsize=(6, 3.5))
w = 0.27
for k, m in enumerate(NAMES):
    ds = [pools[(m, s)] for s in range(3) if (m, s) in pools]
    if not ds:
        continue
    err = np.abs(pd.concat(ds).grade - pd.concat(ds).pred)
    share = [(err == e).mean() * 100 for e in (1, 2, 3)]
    ax.bar(np.arange(3) + (k - 1) * w, share, w, label=NAMES[m])
ax.set_xticks(range(3), ["1 grade off", "2 grades off", "3 grades off"])
ax.set_ylabel("% of all predictions"); ax.legend(fontsize=7); ax.set_title("How wrong are the wrong answers?")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_distance.png"), dpi=150); plt.close(fig)

# training curves for one example run per model
for m in NAMES:
    p = os.path.join(RUNS, f"{m}_f0_s0", "log.csv")
    if not os.path.exists(p):
        continue
    lg = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(lg.epoch, lg.train_loss, label="train"); ax[0].plot(lg.epoch, lg.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(lg.epoch, lg.val_mae, color="C3"); ax[1].set_title("validation MAE")
    unf = lg[lg.phase == "full"].epoch.min()
    if m != "cnn":
        for a in ax:
            a.axvline(unf - 0.5, ls="--", c="gray")
        ax[1].text(unf - 0.4, ax[1].get_ylim()[1] * 0.95, "unfreeze", fontsize=7, va="top")
    fig.suptitle(NAMES[m], fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"curves_{m}.png"), dpi=150); plt.close(fig)
print("figures saved to", OUT)

# ---- compact JSON for the web app
import json  # noqa: E402

web = {
    "n": int(res.n.iloc[0]), "seeds": int(res.seed.nunique()),
    "summary": {m: {k: [float(summ.loc[m, (k, "mean")]), float(np.nan_to_num(summ.loc[m, (k, "std")]))]
                    for k in ("mae", "acc", "qwk", "severe")} for m in NAMES if m in res.model.values},
    "error_dist": {}, "confusion": {},
    "per_fruit": [dict(fruit=f, model=m, **{k: float(v) for k, v in r.items()}) for (f, m), r in fr.iterrows()],
}
for m in NAMES:
    ds = [pools[(m, s)] for s in range(3) if (m, s) in pools]
    if ds:
        d = pd.concat(ds)
        err = np.abs(d.grade - d.pred)
        web["error_dist"][m] = [float((err == e).mean() * 100) for e in (1, 2, 3)]
        web["confusion"][m] = confusion_matrix(pools[(m, 0)].grade, pools[(m, 0)].pred, labels=range(4)).tolist()
# demo samples: one held-out crop per fruit x grade (from the ordinal model's seed-0 test folds)
if ("tl_ordinal", 0) in pools:
    p0 = pools[("tl_ordinal", 0)].sample(frac=1, random_state=3)
    web["samples"] = p0.groupby(["fruit", "grade"]).head(1).sort_values(["fruit", "grade"])[
        ["crop", "fruit", "grade"]].to_dict("records")
    for s in web["samples"]:
        s["grade"] = int(s["grade"])
json.dump(web, open(os.path.join(OUT, "web_results.json"), "w"), indent=1)
print("web_results.json written")
