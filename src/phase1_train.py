"""Phase 1 — FashionReader train.

Trains the small CNN on Fashion-MNIST for 2 epochs. Deliberately short — the
goal is a shipped-in-repo model that fits in <1 MB, not SOTA accuracy.
Writes checkpoints/model.pt + checkpoints/test_subset.npz (first 500 test
images + labels, ~400 KB — all the UI needs to demo inference without
re-downloading the 30 MB Fashion-MNIST tarball on cold start).

Runs locally once; the .pt + .npz get checked into the repo (that's the
point — Streamlit Cloud doesn't train, only infers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
from model import FashionCNN

CHECKPOINTS = HERE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)
DATA_DIR = HERE / "data"


def main(epochs: int = 2, batch_size: int = 128, lr: float = 1e-3, seed: int = 42) -> None:
    torch.manual_seed(seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(str(DATA_DIR), train=True,  download=True, transform=tfm)
    test_ds  = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    model = FashionCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[phase1] model params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running = 0.0; n = 0
        for i, (x, y) in enumerate(train_dl):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
            if i % 100 == 0:
                print(f"  epoch {epoch+1}  batch {i:4d}  loss={loss.item():.3f}")
        print(f"  epoch {epoch+1} avg loss: {running/n:.3f}")

    # Final test accuracy for reporting
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"[phase1] test accuracy: {correct/total:.4f}  ({correct}/{total})")

    torch.save(model.state_dict(), CHECKPOINTS / "model.pt")
    size_kb = (CHECKPOINTS / "model.pt").stat().st_size / 1024
    print(f"[phase1] wrote model.pt ({size_kb:.1f} KB)")

    # Save a small test subset for the deployed app
    xs, ys = [], []
    for x, y in test_dl:
        xs.append(x.numpy()); ys.append(y.numpy())
    xs = np.concatenate(xs)[:500]
    ys = np.concatenate(ys)[:500]
    np.savez_compressed(CHECKPOINTS / "test_subset.npz", images=xs, labels=ys)
    size_kb = (CHECKPOINTS / "test_subset.npz").stat().st_size / 1024
    print(f"[phase1] wrote test_subset.npz ({size_kb:.1f} KB, 500 images)")


if __name__ == "__main__":
    main()
