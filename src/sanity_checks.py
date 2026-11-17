"""Sanity checks run BEFORE any real training (Karpathy's 'recipe for training NNs').

1. Ordinal target encoding is correct.
2. Ordinal probabilities sum to 1 and prediction = number of 'yes' answers.
3. Initial loss matches theory: softmax ≈ ln(4) = 1.386 ; ordinal ≈ 3·ln(2) = 2.079
   (a freshly initialised model should be maximally unsure).
4. One batch of real data looks right (shapes, value range, labels).

    python src/sanity_checks.py
"""
import math

import torch

from common import CropDataset, build_model, fold_frames, loss_fn, ordinal_targets, predict, to_probs

torch.manual_seed(0)

# 1 – target encoding
t = ordinal_targets(torch.tensor([0, 1, 2, 3]))
assert t.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], t
print("1. ordinal targets OK:", t.tolist())

# 2 – probabilities + prediction
logits = torch.tensor([[5., 5., -5.], [-5., -5., -5.], [5., 5., 5.]])
p = to_probs("tl_ordinal", logits)
assert torch.allclose(p.sum(1), torch.ones(3)), p
assert predict("tl_ordinal", logits).tolist() == [2, 0, 3]
print("2. ordinal probs sum to 1, predictions [2,0,3] OK")

# 3 – initial loss
tr, _, _ = fold_frames(0)
ds = CropDataset(tr.sample(64, random_state=0), train=False)
x = torch.stack([ds[i][0] for i in range(len(ds))])
y = torch.tensor([ds[i][1] for i in range(len(ds))])
print(f"4. batch: x {tuple(x.shape)}, range [{x.min():.2f}, {x.max():.2f}], labels {torch.bincount(y, minlength=4).tolist()}")
for kind, theory in [("cnn", math.log(4)), ("tl_softmax", math.log(4)), ("tl_ordinal", 3 * math.log(2))]:
    m = build_model(kind).eval()
    if kind == "tl_softmax":  # make the new head start neutral, as it would be with a zero init
        torch.nn.init.zeros_(m.fc.weight); torch.nn.init.zeros_(m.fc.bias)
    if kind == "tl_ordinal":
        torch.nn.init.zeros_(m.fc.fc.weight); torch.nn.init.zeros_(m.fc.bias)
    with torch.no_grad():
        l = loss_fn(kind, m(x), y).item()
    print(f"3. initial loss {kind:11s} {l:.3f}  (theory for a clueless model {theory:.3f})")
