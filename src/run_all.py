"""Run the full experiment grid: 3 models x 5 folds x 3 seeds = 45 training runs.
Skips runs that already finished (test_preds.csv exists), so it can be resumed.

    python src/run_all.py
"""
import os
import subprocess
import sys

from common import ROOT, N_FOLDS

MODELS = ["cnn", "tl_softmax", "tl_ordinal"]
SEEDS = [0, 1, 2]
here = os.path.dirname(os.path.abspath(__file__))
for seed in SEEDS:
    for fold in range(N_FOLDS):
        for m in MODELS:
            run = os.path.join(ROOT, "runs", f"{m}_f{fold}_s{seed}")
            if os.path.exists(os.path.join(run, "test_preds.csv")):
                continue
            print(f">>> {m} fold {fold} seed {seed}", flush=True)
            with open(os.path.join(ROOT, "runs", "stdout.log"), "a") as log:
                subprocess.run([sys.executable, os.path.join(here, "train.py"), "--model", m,
                                "--fold", str(fold), "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
print("ALL DONE")
