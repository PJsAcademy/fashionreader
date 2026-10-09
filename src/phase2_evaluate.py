"""Phase 2 — FashionReader evaluation.

Loads the shipped model.pt and the test_subset.npz, computes per-class
accuracy + 10×10 confusion matrix. Writes metrics.parquet (per-class) and
confusion.parquet (long-form matrix, 100 rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
from model import FashionCNN, CLASSES

CHECKPOINTS = HERE / "checkpoints"


def main() -> None:
    model = FashionCNN()
    # Trusted artifact: produced by this repo's own src/phase1_train.py in the
    # same build; load_state_dict only accepts tensor dicts, no code.
    model.load_state_dict(torch.load(CHECKPOINTS / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    data = np.load(CHECKPOINTS / "test_subset.npz")
    xs = torch.from_numpy(data["images"])
    ys = data["labels"]
    print(f"[phase2] eval set: {len(xs)} images")

    with torch.no_grad():
        logits = model(xs)
        preds = logits.argmax(1).numpy()

    acc = float((preds == ys).mean())
    print(f"[phase2] overall accuracy: {acc:.4f}")

    # Per-class metrics
    rows = []
    for cls_id, cls_name in enumerate(CLASSES):
        mask = ys == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = float((preds[mask] == cls_id).mean())
        rows.append({"class_id": cls_id, "class_name": cls_name,
                     "support": int(mask.sum()),
                     "accuracy": cls_acc,
                     "errors": int(mask.sum() - (preds[mask] == cls_id).sum())})
    pcm = pd.DataFrame(rows)
    pcm.to_parquet(CHECKPOINTS / "metrics.parquet", index=False)
    print(f"[phase2] per-class metrics (n={len(pcm)}):")
    print(pcm.to_string(index=False))

    # 10×10 confusion matrix, long form
    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(ys, preds):
        conf[t, p] += 1
    cols = []
    for t in range(10):
        for p in range(10):
            cols.append({
                "true_id": t, "true": CLASSES[t],
                "pred_id": p, "pred": CLASSES[p],
                "n": int(conf[t, p]),
            })
    pd.DataFrame(cols).to_parquet(CHECKPOINTS / "confusion.parquet", index=False)
    print(f"[phase2] wrote confusion.parquet ({10*10} rows)")


if __name__ == "__main__":
    main()
