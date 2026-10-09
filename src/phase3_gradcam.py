"""Phase 3 — FashionReader Grad-CAM.

Pure PyTorch implementation of Grad-CAM (Selvaraju et al, 2017). No external
captum/tf-explain dependency.

Mechanic: for a chosen class c, compute gradient of score[c] w.r.t. the last
conv activations A. Average the gradient spatially to get per-channel weights
alpha_c. Weighted-sum A along channels, ReLU → class-discriminative heatmap.
Upsample to the input resolution.

Usage:
    cam = get_cam(model, image_tensor, target_class)   # (H, W) in [0, 1]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_cam(model, image: torch.Tensor, target_class: int) -> np.ndarray:
    """Return a (28, 28) heatmap in [0, 1] showing what pixels mattered for
    the model's score on target_class. image is (1, 1, 28, 28)."""
    model.eval()
    image = image.clone().requires_grad_(False)

    # Forward pass; model stashes last conv activations
    logits = model(image)
    score = logits[0, target_class]

    # Backprop to the stashed activations
    activations = model._last_conv_activations
    grads = torch.autograd.grad(score, activations, retain_graph=False)[0]

    # Pool gradients spatially → per-channel importance weights
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    cam = (weights * activations).sum(dim=1).squeeze(0)  # (H', W')
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample from 7×7 → 28×28
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                        size=(28, 28), mode="bilinear", align_corners=False)
    return cam.squeeze().detach().cpu().numpy()


if __name__ == "__main__":
    # Smoke test: ensure imports + shape work when run standalone.
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(HERE / "src"))
    from model import FashionCNN

    m = FashionCNN()
    img = torch.rand(1, 1, 28, 28)
    cam = get_cam(m, img, target_class=0)
    print(f"[phase3] CAM shape: {cam.shape}  range: [{cam.min():.3f}, {cam.max():.3f}]")
