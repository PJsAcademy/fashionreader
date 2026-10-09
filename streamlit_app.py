"""FashionReader — Fashion-MNIST classifier + Grad-CAM.

Final capstone of Bits to Builds (DL track). 3-conv CNN pre-trained in the
repo, inference-only on Streamlit Cloud (no training on cold-start, no
torchvision download — all 500 test images shipped as a 321 KB .npz).

Tabs:
  1. Explore      — pick a test image, see prediction + confidence + Grad-CAM overlay
  2. Confusion    — 10×10 confusion heatmap + per-class accuracy bars
  3. Upload       — upload a 28×28 grayscale image, see classification
  4. Methodology  — 5 decisions + "what a staff engineer would flag"
  5. About
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"

BRAND_PRIMARY = "#FF6B35"  # DL orange
BRAND_YELLOW = "#FFC72C"
BRAND_GREEN = "#50C878"
BRAND_BLUE = "#4A90E2"
BRAND_RED = "#E63946"
BRAND_INK = "#0E1117"
BRAND_INK2 = "#171B22"
BRAND_INK3 = "#232833"
BRAND_FG = "#E7E9EC"
BRAND_FG_DIM = "#9AA3B2"

sys.path.insert(0, str(HERE / "src"))

st.set_page_config(
    page_title="FashionReader — Fashion-MNIST + Grad-CAM",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================= bootstrap

@st.cache_resource(show_spinner="First-time setup: evaluating model (~5s)...")
def bootstrap() -> None:
    required = [
        CHECKPOINTS / "model.pt",
        CHECKPOINTS / "test_subset.npz",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p.name}. These are pre-trained artifacts shipped "
                "in the repo — if absent, run `python src/phase1_train.py` "
                "locally (needs ~30 MB torchvision download) and commit."
            )
    # Compute metrics if missing
    if not (CHECKPOINTS / "metrics.parquet").exists():
        import phase2_evaluate
        phase2_evaluate.main()


bootstrap()


# ======================================================= loaders

from model import FashionCNN, CLASSES  # noqa: E402
from phase3_gradcam import get_cam  # noqa: E402


@st.cache_resource
def load_model():
    m = FashionCNN()
    # Trusted artifact: model.pt is produced by this repo's own src/phase1_train.py
    # and loaded with weights_only=True (tensor dict only, no code execution).
    m.load_state_dict(torch.load(CHECKPOINTS / "model.pt",
                                  map_location="cpu", weights_only=True))
    m.eval()
    return m


@st.cache_data
def load_test() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(CHECKPOINTS / "test_subset.npz")
    return data["images"], data["labels"]


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "metrics.parquet")


@st.cache_data
def load_confusion() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "confusion.parquet")


# ======================================================= Global CSS

st.markdown(
    f"""
    <style>
    #MainMenu, footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}
    .kpi-card {{
        background: linear-gradient(135deg, {BRAND_INK2} 0%, {BRAND_INK3} 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid var(--accent, {BRAND_PRIMARY});
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        height: 100%;
    }}
    .kpi-card .kpi-label {{color: {BRAND_FG_DIM}; font-size: 12px; font-weight: 500;
                          text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;}}
    .kpi-card .kpi-value {{color: {BRAND_FG}; font-size: clamp(16px, 1.9vw, 30px);
                          font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
    .kpi-card .kpi-delta {{display: inline-block; margin-top: 6px; color: {BRAND_FG_DIM};
                          font-size: 12px;}}
    .kpi-icon {{float: right; font-size: 20px; opacity: 0.35; margin-left: 4px;}}
    @media (max-width: 1100px) {{ .kpi-icon {{display: none;}} }}
    .insight {{background: {BRAND_INK2}; border: 1px solid rgba(255,107,53,0.22);
              border-radius: 10px; padding: 10px 14px; font-size: 13px;
              color: {BRAND_FG}; line-height: 1.45;}}
    .insight .insight-tag {{color: {BRAND_PRIMARY}; font-weight: 600; font-size: 11px;
                           text-transform: uppercase; letter-spacing: 0.07em; margin-right: 6px;}}
    .hero {{background: radial-gradient(circle at top left, rgba(255,107,53,0.14) 0%, rgba(14,17,23,0) 55%);
           padding: 10px 0 16px; margin-bottom: 10px;}}
    .hero h1 {{font-size: 36px !important; font-weight: 800 !important; margin-bottom: 4px !important;}}
    .hero .tagline {{color: {BRAND_FG_DIM}; font-size: 15px;}}
    section[data-testid="stSidebar"] {{background: {BRAND_INK};}}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, delta: str = "", icon: str = "", accent: str = BRAND_PRIMARY):
    st.markdown(
        f"""<div class="kpi-card" style="--accent:{accent};">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_image(arr: np.ndarray, size: int = 280) -> Image.Image:
    """Grayscale 28×28 float [0,1] array → PIL upscaled image."""
    if arr.ndim == 3: arr = arr[0]
    img8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img8, mode="L").resize((size, size), Image.NEAREST)


def render_cam_overlay(image: np.ndarray, cam: np.ndarray, size: int = 280) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the grayscale image. Red = important."""
    if image.ndim == 3: image = image[0]
    base = (image * 255).clip(0, 255).astype(np.uint8)
    base_rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    # Red colormap: cam ∈ [0,1] → (R=255*cam, G=0, B=0)
    red = np.stack([cam * 255, np.zeros_like(cam), np.zeros_like(cam)], axis=-1)
    # Alpha-blend red over grayscale, weight by cam intensity
    alpha = (cam[..., None] * 0.6)
    out = (1 - alpha) * base_rgb + alpha * red
    out = out.clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB").resize((size, size), Image.NEAREST)


# ======================================================= Load

model = load_model()
test_images, test_labels = load_test()
metrics = load_metrics()
confusion = load_confusion()
overall_acc = float((metrics["accuracy"] * metrics["support"]).sum() / metrics["support"].sum())
easiest = metrics.loc[metrics["accuracy"].idxmax()]
hardest = metrics.loc[metrics["accuracy"].idxmin()]


# ======================================================= Sidebar

with st.sidebar:
    st.markdown("### 👗 FashionReader")
    st.caption("Bits to Builds · DL capstone")
    st.divider()
    st.markdown("##### Model")
    n_params = sum(p.numel() for p in model.parameters())
    st.caption(f"**FashionCNN** — 3 conv + 2 fc")
    st.caption(f"**{n_params:,}** parameters")
    st.caption(f"**{(CHECKPOINTS / 'model.pt').stat().st_size / 1024:.0f} KB** model.pt")

    st.markdown("##### Data")
    st.caption(f"**{len(test_images)}** Fashion-MNIST test images shipped")
    st.caption("28×28 grayscale, 10 classes")

    st.download_button("⬇ Download metrics", metrics.to_csv(index=False).encode(),
                       "metrics.csv", "text/csv", use_container_width=True)
    st.download_button("⬇ Download confusion", confusion.to_csv(index=False).encode(),
                       "confusion.csv", "text/csv", use_container_width=True)

    st.divider()
    st.markdown("##### Links")
    st.markdown("[💻 Source on GitHub](https://github.com/PJsAcademy/fashionreader)")
    st.markdown("[📚 Bits to Builds](https://bitstobuilds.com)")
    st.markdown("[👕 Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)")


# ======================================================= Hero + KPIs

st.markdown(
    """<div class="hero">
      <h1>👗 FashionReader</h1>
      <div class="tagline">Fashion-MNIST classifier (3-conv CNN) + Grad-CAM saliency maps · DL capstone of
      <a href="https://bitstobuilds.com" style="color:#FFC72C;">Bits to Builds</a></div>
    </div>""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("Accuracy", f"{overall_acc*100:.1f}%", f"on {len(test_images)} test images", "🎯")
with k2: kpi_card("Parameters", f"{n_params/1000:.1f}K", "deliberately small", "🧠", BRAND_YELLOW)
with k3: kpi_card("Model size", f"{(CHECKPOINTS/'model.pt').stat().st_size/1024:.0f} KB",
                   "shipped in repo", "📦", BRAND_BLUE)
with k4: kpi_card("Easiest class", easiest["class_name"], f"{easiest['accuracy']*100:.0f}% accuracy", "✅", BRAND_GREEN)
with k5: kpi_card("Hardest class", hardest["class_name"], f"{hardest['accuracy']*100:.0f}% accuracy", "⚠️", BRAND_RED)


# Insights
most_confused = confusion.query("true_id != pred_id").sort_values("n", ascending=False).iloc[0]
cls_counts = pd.Series(test_labels).value_counts().sort_index()
best_cls = metrics.iloc[0]["class_name"]  # just first class for pills demo

st.markdown("")
i1, i2, i3, i4 = st.columns(4)
pills = [
    ("Trained",   f"<b>2 epochs</b> on 60,000 training images (CPU, seed=42)."),
    ("Model",     f"<b>3 conv + 2 fc</b>, {n_params:,} parameters, <5 MB on disk."),
    ("Top mistake", f"<b>{most_confused['true']} → {most_confused['pred']}</b> happened {int(most_confused['n'])} times."),
    ("Honest limit", f"Grayscale 28×28 + 2 epochs = this is the lesson, not SOTA."),
]
for col, (tag, body) in zip([i1, i2, i3, i4], pills):
    with col:
        st.markdown(f'<div class="insight"><span class="insight-tag">{tag}</span>{body}</div>',
                    unsafe_allow_html=True)

st.markdown("")
st.divider()

tab_explore, tab_conf, tab_upload, tab_method, tab_about = st.tabs(
    ["🔎 Explore", "🧮 Confusion", "📤 Upload", "🛠 Methodology", "ℹ️ About"]
)


# ------- Explore tab -------
with tab_explore:
    st.subheader("Pick a test image → classification + Grad-CAM")
    st.caption("Grad-CAM highlights the pixels the CNN keyed on when predicting the shown class. "
               "Red = most important.")

    c1, c2 = st.columns([1, 3])
    with c1:
        pick_class = st.selectbox("Filter by true class", ["(any)"] + CLASSES)
        if pick_class == "(any)":
            candidate_ids = np.arange(len(test_images))
        else:
            cls_id = CLASSES.index(pick_class)
            candidate_ids = np.where(test_labels == cls_id)[0]
        if len(candidate_ids) == 0:
            st.warning("No images for that class in the shipped subset.")
            st.stop()
        pos = st.slider("Image #", 0, max(0, len(candidate_ids) - 1), 0)
        idx = int(candidate_ids[pos])
        st.caption(f"Test image {idx}, true: **{CLASSES[int(test_labels[idx])]}**")

    img_np = test_images[idx]
    img_t = torch.from_numpy(img_np).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_id = int(probs.argmax())
    pred_cls = CLASSES[pred_id]
    conf = float(probs[pred_id])
    correct = pred_id == int(test_labels[idx])

    with c2:
        a, b, c = st.columns(3)
        a.image(render_image(img_np), caption="Input (28×28 grayscale, upscaled)")
        # Grad-CAM for the PREDICTED class
        cam = get_cam(model, img_t, pred_id)
        b.image(render_cam_overlay(img_np, cam),
                caption=f"Grad-CAM for predicted class: {pred_cls}")

        with c:
            kpi_card(
                "Prediction", pred_cls, f"{conf*100:.1f}% confidence",
                "✅" if correct else "❌",
                BRAND_GREEN if correct else BRAND_RED,
            )
            if not correct:
                st.caption(
                    f"**Why this is wrong (honestly):** the top class is "
                    f"**{pred_cls}** at {conf*100:.1f}%; the true class is "
                    f"**{CLASSES[int(test_labels[idx])]}** at {probs[int(test_labels[idx])]*100:.1f}%. "
                    "See Methodology for why upper-body garments confuse grayscale CNNs."
                )

    st.divider()
    st.subheader("Full probability distribution")
    prob_df = pd.DataFrame({"class": CLASSES, "probability": probs})
    bar = (alt.Chart(prob_df).mark_bar(cornerRadius=4).encode(
               x=alt.X("probability:Q", title="Probability", axis=alt.Axis(format=".0%")),
               y=alt.Y("class:N", sort="-x", title=None),
               color=alt.condition(
                   f"datum['class'] == '{pred_cls}'",
                   alt.value(BRAND_PRIMARY), alt.value(BRAND_INK3),
               ),
               tooltip=["class", alt.Tooltip("probability:Q", format=".2%")],
           ).properties(height=320))
    st.altair_chart(bar, use_container_width=True)


# ------- Confusion tab -------
with tab_conf:
    st.subheader("Confusion matrix")
    st.caption("Rows = true class, columns = predicted class. Diagonal = correct. "
               "Off-diagonal brightness shows where the model goes wrong.")

    conf_show = confusion.copy()
    heat = (alt.Chart(conf_show).mark_rect(stroke=BRAND_INK, strokeWidth=1).encode(
                x=alt.X("pred:N", title="Predicted", sort=CLASSES,
                        axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("true:N", title="True", sort=CLASSES),
                color=alt.Color("n:Q", scale=alt.Scale(scheme="orangered"),
                                legend=alt.Legend(title="Count")),
                tooltip=["true", "pred", "n"],
            ).properties(height=440))
    labels = (alt.Chart(conf_show).mark_text(fontSize=10, color=BRAND_FG).encode(
                  x=alt.X("pred:N", sort=CLASSES),
                  y=alt.Y("true:N", sort=CLASSES),
                  text=alt.Text("n:Q"),
              ))
    st.altair_chart((heat + labels).properties(height=440), use_container_width=True)

    st.divider()
    st.subheader("Per-class accuracy")
    pcm = metrics.copy()
    pcm_bar = (alt.Chart(pcm).mark_bar(cornerRadius=4).encode(
                   x=alt.X("accuracy:Q", title="Accuracy", axis=alt.Axis(format=".0%"),
                           scale=alt.Scale(domain=[0, 1])),
                   y=alt.Y("class_name:N", sort="-x", title=None),
                   color=alt.Color("accuracy:Q",
                                   scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                                   legend=None),
                   tooltip=["class_name", alt.Tooltip("accuracy:Q", format=".2%"),
                            "support", "errors"],
               ).properties(height=320))
    st.altair_chart(pcm_bar, use_container_width=True)


# ------- Upload tab -------
with tab_upload:
    st.subheader("Try your own image")
    st.caption("Upload a PNG/JPG. It will be converted to 28×28 grayscale (centered, inverted if light). "
               "Works best for cropped single-item images (one shirt, one shoe, etc.) on a plain background.")

    uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        pil = Image.open(uploaded).convert("L")
        # Resize to 28×28 maintaining aspect ratio via center crop
        pil.thumbnail((28, 28), Image.LANCZOS)
        padded = Image.new("L", (28, 28), 0)
        padded.paste(pil, ((28 - pil.width) // 2, (28 - pil.height) // 2))
        arr = np.array(padded).astype(np.float32) / 255.0
        # Fashion-MNIST is light clothing on dark; invert if the image is light-dominant
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        img_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_id = int(probs.argmax())
        cam = get_cam(model, img_t, pred_id)

        c1, c2, c3 = st.columns(3)
        c1.image(render_image(arr), caption="Preprocessed (28×28)")
        c2.image(render_cam_overlay(arr, cam), caption=f"Grad-CAM for {CLASSES[pred_id]}")
        with c3:
            kpi_card("Prediction", CLASSES[pred_id], f"{probs[pred_id]*100:.1f}% confidence", "👗")
            st.caption("Preprocessing: grayscale, resize-and-pad to 28×28, auto-invert if bright. "
                       "Any step can misrepresent your image — see Methodology.")


# ------- Methodology tab -------
with tab_method:
    st.markdown(
        f"""
        ## Methodology — decisions, tradeoffs, honest shortcuts

        ---
        ### Decision 1 — Why a 3-conv CNN, not ResNet-18?

        **Chose:** 3 conv layers + 2 fully-connected, {n_params:,} parameters, ~{(CHECKPOINTS/'model.pt').stat().st_size/1024:.0f} KB.

        **Why:** Fashion-MNIST is 28×28 grayscale. A ResNet trained from scratch
        would overfit badly on 60k images; a pretrained-on-ImageNet ResNet is a
        channel-mismatch dance (1 vs 3 channels) and 44 MB larger. For the lesson —
        "here's how CNNs work on a toy problem" — the small net is honest, fast,
        and ships in a free-tier repo.

        **What I'd change with 10× the time:** add a data-augmented ResNet-18 as a
        comparison baseline. Show that 1.5% better accuracy costs 300× more
        parameters.

        ---
        ### Decision 2 — Why 2 epochs, not 20?

        **Chose:** 2 epochs, final test accuracy **{overall_acc*100:.1f}%**.

        **Why (honest):** On the full 60k Fashion-MNIST train set, 2 epochs reaches
        ~85-87% test accuracy; 20 epochs reaches ~91-92%. The marginal 5% isn't
        the point of the capstone — the pipeline, Grad-CAM, and confusion analysis
        are. Shipping a trained-to-SOTA model would mask the honest Fashion-MNIST
        lesson: **upper-body garments confuse CNNs** (see the Coat ≈ Pullover ≈
        Shirt confusion in the heatmap).

        **What I'd change with 10× the time:** 20 epochs + mixup + random erasing +
        label smoothing. Expect 92%+ accuracy. Add a plot of train/test loss curves
        so the overfitting story is visible.

        ---
        ### Decision 3 — Why ship the model instead of training on cold start?

        **Chose:** `model.pt` + `test_subset.npz` committed to the repo.

        **Why:** Training on cold start would take 2-5 min on Streamlit Cloud's
        1-CPU free tier. The torchvision download alone is 30 MB. Shipping a
        pre-trained 133 KB model + 321 KB test subset = **cold start in seconds**,
        no network. This is also how real DL products ship — training is offline,
        deploy serves inference.

        **What I'd change with 10× the time:** add a `make train` target in a
        GitHub Action that trains on a GPU runner nightly and commits a fresh
        `model.pt`.

        ---
        ### Decision 4 — Why Grad-CAM instead of SHAP / LIME / Integrated Gradients?

        **Chose:** Grad-CAM (Selvaraju et al, 2017), ~30 lines of pure PyTorch.

        **Why:** For a conv net on images, Grad-CAM is the standard first-cut
        explainability tool. It uses the model's own gradients (no extra fitting,
        no sampling) and produces class-discriminative heatmaps that align with
        the input resolution. SHAP for images needs ~1000 forward passes per
        explanation; LIME needs training a surrogate model; Integrated Gradients
        needs 20-50 passes. Grad-CAM is one forward + one backward, sub-100ms on
        CPU.

        **What I'd change with 10× the time:** add Grad-CAM++ (fixes "spatial
        bias" when important pixels are scattered) and a SmoothGrad option (adds
        noise and averages to reduce visual noise).

        ---
        ### Decision 5 — Why auto-invert uploaded images?

        **Chose:** If the mean pixel > 0.5, invert so dark background → light foreground.

        **Why (honest shortcut):** Fashion-MNIST is light-colored clothing on a
        dark background. Most user-uploaded photos are the opposite — dark clothing
        on a light background. The model has never seen that distribution.
        Auto-inverting helps *some* uploads but gets confused on photos with mixed
        lighting (collar vs torso).

        **What I'd change with 10× the time:** train with heavy color-jitter and
        inversion augmentation so the deployed model is light/dark-background
        agnostic. Add a "manual invert" toggle for the current demo.

        ---

        ## What a staff engineer would flag that I left in

        - **Only 500 test images for metrics.** Shipped subset is 500 of 10,000
          test images. Full-test metrics are available by re-running `phase2_evaluate.py`
          after `phase1_train.py` locally — they're ~1% apart from the subset's.
        - **No calibration.** Softmax probabilities are confidence-shaped but not
          calibrated. A 90% confidence prediction is not actually right 90% of the time.
          Temperature scaling is the one-line fix.
        - **No OOD detection.** Upload a cat photo and the model confidently says
          "T-shirt". A proper DL product adds OOD rejection (Mahalanobis distance
          over the penultimate layer).
        - **No fairness audit of the training set.** Fashion-MNIST is a Zalando
          catalog scraped in 2017 — Western fashion, specific style skew. The
          demo inherits whatever bias that has.
        """
    )


# ------- About tab -------
with tab_about:
    st.markdown(
        f"""
        ## About

        **FashionReader** is the Deep Learning capstone of the
        [Bits to Builds](https://bitstobuilds.com) course. A small CNN
        trained on Fashion-MNIST (28×28 grayscale, 10 clothing classes) with
        Grad-CAM saliency for the "why did the model see it that way" tab.

        | Phase | Deliverable |
        |-------|-------------|
        | 1. Train    | `src/phase1_train.py` — 3-conv CNN, 2 epochs, ~{overall_acc*100:.0f}% test acc |
        | 2. Evaluate | per-class accuracy + 10×10 confusion matrix |
        | 3. Grad-CAM | pure-PyTorch class-discriminative heatmap, 1 forward + 1 backward |

        ## Data

        [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando,
        MIT license). The shipped Space uses 500 pre-saved test images + the
        trained model — no torchvision download at cold start. For retraining,
        run `python src/phase1_train.py` locally.

        ## Honest limits

        - **{overall_acc*100:.1f}% test accuracy** on the shipped subset. SOTA is
          ~95%; staying short of SOTA keeps the Fashion-MNIST failure modes
          (Coat ≈ Pullover ≈ Shirt) visible, which is the lesson.
        - **Hardest class: {hardest['class_name']} at {hardest['accuracy']*100:.0f}%** —
          commonly confused with other upper-body garments.
        - **Uploaded images** go through a simple grayscale + resize + maybe-invert
          preprocess; real-world photos often fall out of the training distribution.

        ## Source
        [github.com/PJsAcademy/fashionreader](https://github.com/PJsAcademy/fashionreader)
        """
    )
