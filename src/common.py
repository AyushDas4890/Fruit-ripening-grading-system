"""Shared building blocks: dataset, models, ordinal head, losses, metrics.

Syllabus links are marked [Unit X].
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = os.path.join(ROOT, "data", "crops")
MANIFEST = os.path.join(ROOT, "data", "splits", "manifest.csv")
K = 4  # ripeness grades: 0 unripe < 1 partially ripe < 2 ripe < 3 rotten
GRADES = ["unripe", "partially ripe", "ripe", "rotten"]
N_FOLDS = 5
IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics [Unit I: normalisation]


# ----------------------------------------------------------------------------- data
def transforms_for(train):
    """[Unit I] resize + normalise.  [Unit II] augmentation for training only.

    No hue / saturation jitter on purpose: colour IS the ripeness signal, changing
    it would change the correct label."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),  # lighting only
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CropDataset(torch.utils.data.Dataset):
    """All crops are small (≤256 px), so they are loaded into RAM once."""

    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.imgs = [Image.open(os.path.join(CROPS, f)).convert("RGB") for f in self.df.crop]
        self.tf = transforms_for(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.df.grade[i]), i


def fold_frames(fold):
    """Run `fold`: test = fold, val = fold+1, train = the other three."""
    m = pd.read_csv(MANIFEST)
    val = (fold + 1) % N_FOLDS
    return m[~m.fold.isin([fold, val])], m[m.fold == val], m[m.fold == fold]


# ----------------------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """Baseline trained from scratch.  [Unit II] conv, BatchNorm, ReLU, pooling, dropout."""

    def __init__(self, out_dim):
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                       nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, out_dim))

    def forward(self, x):
        return self.head(self.features(x))


class OrdinalHead(nn.Module):
    """CORAL ordinal head.

    One shared weight vector w gives a single 'ripeness score' s = w·x.
    K-1 separate biases b_k turn it into K-1 logits:  logit_k = s + b_k,
    meaning P(grade > k).  Because w is shared, the K-1 answers cannot contradict
    each other (rank consistency)."""

    def __init__(self, in_dim, k=K):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        # start ordered and well separated: b = [2, 0, -2].  (A first version used
        # [1, .67, .33]; the overfit-one-batch test got stuck at loss ~0.47 because Adam
        # moves each bias only ~lr per step, so the thresholds could not spread apart.)
        self.bias = nn.Parameter(torch.linspace(2, -2, k - 1))

    def forward(self, x):
        return self.fc(x) + self.bias  # (B, K-1)


def build_model(kind):
    """kind ∈ {cnn, tl_softmax, tl_ordinal}.  [Unit III] pretrained ResNet18 backbone."""
    if kind == "cnn":
        return SmallCNN(K)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_dim = net.fc.in_features
    net.fc = OrdinalHead(in_dim) if kind == "tl_ordinal" else nn.Linear(in_dim, K)
    return net


# ----------------------------------------------------------------------------- ordinal maths
def ordinal_targets(y, k=K):
    """grade y -> K-1 binary targets [y>0, y>1, ..., y>K-2].   e.g. y=2 -> [1,1,0]."""
    return (y.unsqueeze(1) > torch.arange(k - 1, device=y.device)).float()


def loss_fn(kind, out, y):
    if kind == "tl_ordinal":
        return F.binary_cross_entropy_with_logits(out, ordinal_targets(y), reduction="sum") / len(y)
    return F.cross_entropy(out, y)  # [Unit I] softmax + cross-entropy


def to_probs(kind, out):
    """Return per-grade probabilities (B, K) for any head."""
    if kind == "tl_ordinal":
        p_gt = torch.sigmoid(out)                       # P(y>0), P(y>1), P(y>2)
        ones, zeros = torch.ones_like(p_gt[:, :1]), torch.zeros_like(p_gt[:, :1])
        cum = torch.cat([ones, p_gt, zeros], 1)          # P(y>-1)=1 ... P(y>K-1)=0
        return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)   # P(y=k) = P(y>k-1) - P(y>k)
    return torch.softmax(out, 1)


def predict(kind, out):
    if kind == "tl_ordinal":
        return (torch.sigmoid(out) > 0.5).sum(1)  # count of "yes" answers
    return out.argmax(1)


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    from sklearn.metrics import cohen_kappa_score
    y, p = np.asarray(y), np.asarray(p)
    return dict(
        mae=float(np.abs(y - p).mean()),
        acc=float((y == p).mean()),
        qwk=float(cohen_kappa_score(y, p, weights="quadratic")),
        severe=float((np.abs(y - p) >= 2).mean()),  # share of errors 2+ grades off
    )
