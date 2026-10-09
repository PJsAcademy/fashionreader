# FashionReader

Small CNN on Fashion-MNIST + Grad-CAM saliency. DL capstone of
[Bits to Builds](https://bitstobuilds.com). 3 conv + 2 fc layers, 33k parameters,
133 KB `model.pt`, pure-PyTorch Grad-CAM in 30 lines.

**Live demo:** (fill in after `publish.sh`)
**Source:** <https://github.com/PJsAcademy/fashionreader>

---

## What it does

| Phase | Deliverable |
|-------|-------------|
| 1. Train    | FashionCNN (3 conv + 2 fc), 2 epochs on 60k Fashion-MNIST images, ~86% test acc |
| 2. Evaluate | per-class accuracy, 10×10 confusion matrix (orange = correct, red = confused) |
| 3. Grad-CAM | gradient-weighted saliency, one forward + one backward, sub-100ms CPU |

## Run locally

```bash
pip install -r requirements.txt
python src/phase1_train.py     # ~3 min first time (downloads Fashion-MNIST)
python src/phase2_evaluate.py  # ~2s
python src/phase3_gradcam.py   # smoke test, ~1s
pytest tests/ -q               # 17 invariants should pass
streamlit run streamlit_app.py
```

Note: `checkpoints/model.pt` and `checkpoints/test_subset.npz` are committed to
the repo (they're the shipped artifacts). On Streamlit Cloud deploy, bootstrap
loads them directly — **no training at cold start**.

## Deploy to Streamlit Community Cloud (free)

```bash
REPO=fashionreader ../portfolio/publish.sh
```

Then at <https://share.streamlit.io>: Create app → pick this repo → main file
`streamlit_app.py` → Deploy. First build takes ~3 min (torch CPU wheel install
is the slow part); subsequent cold starts are seconds.

## Honest limits

- **86% test accuracy, not 95%.** 2 epochs trades 5% accuracy for a clear
  demonstration of Fashion-MNIST's hard-class confusion (Coat ≈ Pullover ≈ Shirt).
- **No calibration, no OOD rejection.** Upload a cat photo and the model
  confidently says "T-shirt". Flagged in Methodology.
- **Auto-invert on upload** is a shortcut for background-dark Fashion-MNIST vs
  background-light user photos. Not robust to mixed lighting.
- **500 test images shipped** out of 10,000 total — full-test metrics ~1% apart.

## Credits

Fashion-MNIST: [Zalando Research, 2017, MIT license](https://github.com/zalandoresearch/fashion-mnist).
Grad-CAM: Selvaraju et al, 2017. Built from the
[Bits to Builds](https://bitstobuilds.com) curriculum.
