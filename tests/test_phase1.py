"""Phase 1 invariants (6) — model + shipped artifacts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import FashionCNN, CLASSES  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"


def test_classes_shape():
    assert len(CLASSES) == 10
    assert len(set(CLASSES)) == 10, "duplicate class name"


def test_model_param_count_small():
    model = FashionCNN()
    n = sum(p.numel() for p in model.parameters())
    assert 10_000 < n < 200_000, (
        f"model has {n:,} params — too big to ship small, probably wrong architecture"
    )


def test_model_forward_shape():
    model = FashionCNN()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10), f"expected (4, 10), got {out.shape}"


def test_model_file_exists_and_loads():
    p = CHECKPOINTS / "model.pt"
    if not p.exists():
        pytest.skip("model.pt missing — run `python src/phase1_train.py` first")
    model = FashionCNN()
    # weights_only=True: only tensor dict, no arbitrary code execution.
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))


def test_test_subset_exists():
    p = CHECKPOINTS / "test_subset.npz"
    if not p.exists():
        pytest.skip("test_subset.npz missing")
    data = np.load(p)
    assert data["images"].shape == (500, 1, 28, 28)
    assert data["labels"].shape == (500,)
    assert data["labels"].min() >= 0 and data["labels"].max() < 10


def test_test_subset_small_enough_to_ship():
    p = CHECKPOINTS / "test_subset.npz"
    if not p.exists():
        pytest.skip("test_subset.npz missing")
    size_kb = p.stat().st_size / 1024
    assert size_kb < 500, f"test subset too large ({size_kb:.0f} KB) — shouldn't ship"
