"""Phase 3 invariants (5) — Grad-CAM."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import FashionCNN  # noqa: E402
from phase3_gradcam import get_cam  # noqa: E402


@pytest.fixture(scope="module")
def model():
    m = FashionCNN()
    p = ROOT / "checkpoints" / "model.pt"
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    m.eval()
    return m


def test_cam_shape(model):
    cam = get_cam(model, torch.rand(1, 1, 28, 28), target_class=0)
    assert cam.shape == (28, 28)


def test_cam_in_unit_interval(model):
    cam = get_cam(model, torch.rand(1, 1, 28, 28), target_class=3)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6


def test_cam_nonzero_for_random_image(model):
    """Grad-CAM shouldn't be identically zero — if it is, the gradient didn't flow."""
    cam = get_cam(model, torch.rand(1, 1, 28, 28), target_class=5)
    assert cam.max() > 1e-4


def test_different_targets_produce_different_cams(model):
    img = torch.rand(1, 1, 28, 28)
    cam_a = get_cam(model, img, target_class=0)
    cam_b = get_cam(model, img, target_class=7)
    # They may both be low-signal on random noise, but they shouldn't be identical
    assert not np.allclose(cam_a, cam_b, atol=1e-5), (
        "same CAM for different target classes — gradient didn't vary by target"
    )


def test_cam_runs_fast_enough(model):
    """CPU inference + one backward pass should be sub-500ms."""
    import time
    img = torch.rand(1, 1, 28, 28)
    start = time.perf_counter()
    get_cam(model, img, target_class=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"get_cam took {elapsed*1000:.0f} ms — too slow for live UI"
