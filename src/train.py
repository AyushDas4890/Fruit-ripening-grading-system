"""Train one model on one cross-validation fold with one seed.

    python src/train.py --model tl_ordinal --fold 0 --seed 0

Writes runs/<model>_f<fold>_s<seed>/ :
    config.json      all settings
    log.csv          per-epoch train loss, val loss, val MAE, val accuracy, learning rate
    best.pt          weights of the epoch with the lowest validation MAE (early stopping)
    test_preds.csv   predictions + per-grade probabilities on the held-out test fold

Training recipe:
  [Unit III] transfer models: phase 1 trains only the new head with the backbone FROZEN,
             phase 2 UNFREEZES everything with layer-wise learning rates
             (backbone 10x smaller than head), so pretrained features are not destroyed.
  [Unit II]  AdamW + weight decay, cosine learning-rate schedule, early stopping on val MAE.
  [Unit I]   mixed precision (AMP) to fit the 4 GB GPU.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from common import ROOT, K, CropDataset, build_model, fold_frames, loss_fn, metrics, predict, to_probs


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_epoch(model, kind, loader, dev, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, ys, ps, outs, idxs = 0.0, [], [], [], []
    for x, y, i in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = model(x)
            loss = loss_fn(kind, out.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot += loss.item() * len(y)
        ys.append(y.cpu()); ps.append(predict(kind, out.float()).cpu())
        outs.append(out.float().detach().cpu()); idxs.append(i)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return tot / len(y), y, p, torch.cat(outs), torch.cat(idxs).numpy()


def param_groups(model, kind, lr, frozen):
    if kind == "cnn":
        return [{"params": model.parameters(), "lr": lr}]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    for p in body:
        p.requires_grad = not frozen
    if kind == "tl_ordinal":
        # thresholds are a handful of scalars that must travel several units: give them 10x LR
        groups = [{"params": model.fc.fc.parameters(), "lr": lr},
                  {"params": [model.fc.bias], "lr": lr * 10, "weight_decay": 0.0}]
    else:
        groups = [{"params": model.fc.parameters(), "lr": lr}]
    if not frozen:
        groups.append({"params": body, "lr": lr * 0.1})  # layer-wise LR: backbone 10x smaller
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "tl_softmax", "tl_ordinal"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--overfit_one_batch", action="store_true", help="sanity check: memorise 1 batch")
    args = ap.parse_args()
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(ROOT, "runs", f"{args.model}_f{args.fold}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    tr_df, va_df, te_df = fold_frames(args.fold)
    tr, va, te = CropDataset(tr_df, True), CropDataset(va_df, False), CropDataset(te_df, False)
    g = torch.Generator().manual_seed(args.seed)
    tr_dl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, generator=g, drop_last=True)
    va_dl = torch.utils.data.DataLoader(va, 64)
    te_dl = torch.utils.data.DataLoader(te, 64)

    model = build_model(args.model).to(dev)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    if args.overfit_one_batch:  # Karpathy sanity check: loss on one batch must go to ~0
        x, y, _ = next(iter(tr_dl))
        x, y = x.to(dev), y.to(dev)
        opt = torch.optim.AdamW(param_groups(model, args.model, args.lr, frozen=False), weight_decay=0)
        model.train()
        for step in range(301):
            loss = loss_fn(args.model, model(x).float(), y)
            if step == 0 or step % 50 == 0:
                print(f"step {step:3d} loss {loss.item():.4f}")
            opt.zero_grad(); loss.backward(); opt.step()
        return

    frozen = args.model != "cnn" and args.freeze_epochs > 0
    opt = torch.optim.AdamW(param_groups(model, args.model, args.lr, frozen), weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, bad = 1e9, 0
    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "phase", "train_loss", "val_loss", "val_mae", "val_acc", "lr", "sec"])
    for ep in range(args.epochs):
        if frozen and ep == args.freeze_epochs:  # phase 2: unfreeze + layer-wise LR
            frozen = False
            opt = torch.optim.AdamW(param_groups(model, args.model, args.lr, frozen), weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs - ep)
        t = time.time()
        trl, *_ = run_epoch(model, args.model, tr_dl, dev, opt, scaler)
        val, y, p, *_ = run_epoch(model, args.model, va_dl, dev)
        m = metrics(y, p)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        phase = "frozen" if frozen else "full"
        wr.writerow([ep, phase, f"{trl:.4f}", f"{val:.4f}", f"{m['mae']:.4f}", f"{m['acc']:.4f}", f"{lr:.2e}",
                     f"{time.time() - t:.1f}"])
        log.flush()
        print(f"ep {ep:2d} [{phase}] train {trl:.3f} | val loss {val:.3f} mae {m['mae']:.3f} acc {m['acc']:.3f}")
        score = m["mae"] + 1e-3 * val  # early stopping on val MAE (val loss breaks ties)
        if score < best - 1e-4:
            best, bad = score, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        elif not frozen:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    log.close()

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev))
    _, y, p, out, idx = run_epoch(model, args.model, te_dl, dev)
    probs = to_probs(args.model, out).numpy()
    df = te_df.reset_index(drop=True).iloc[idx][["crop", "fruit", "grade"]].copy()
    df["pred"] = p
    for k in range(K):
        df[f"p{k}"] = probs[:, k].round(4)
    df.to_csv(os.path.join(out_dir, "test_preds.csv"), index=False)
    print("TEST", json.dumps({k: round(v, 4) for k, v in metrics(y, p).items()}))


if __name__ == "__main__":
    main()
