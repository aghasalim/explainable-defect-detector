"""Streamlit demo for the Explainable Visual Defect Detector.

Loads an exported memory bank (models/<category>.pt), scores an uploaded image
against it, and shows the anomaly map beside the verdict.

The threshold shown is the one calibrated on held-out NORMAL training images at
a 1% false-alarm target - not tuned on the test set. The sidebar exposes it so
the trade-off is visible rather than hidden behind a single word.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "edd"))

from dataset import build_transform          # noqa: E402
from export import MODELS, predict           # noqa: E402
from patchcore import PatchFeatures          # noqa: E402

st.set_page_config(page_title="Explainable Defect Detector", page_icon="🔍", layout="wide")


@st.cache_resource
def get_model():
    dev = torch.device("cpu")   # Spaces free tier is CPU-only
    return PatchFeatures().to(dev), dev


@st.cache_resource
def get_artifact(category: str):
    a = torch.load(MODELS / f"{category}.pt", weights_only=False)
    a["bank"] = a["bank"].float()
    return a


def overlay(img: Image.Image, amap: np.ndarray, vmax: float, alpha: float = 0.5) -> Image.Image:
    """Colour the map on an ABSOLUTE scale anchored to the threshold.

    Per-image min-max normalisation is the intuitive choice and it is wrong
    here: it stretches whatever range an image happens to have to full
    brightness, so a perfectly clean part renders as a blaze of orange and the
    demo contradicts its own verdict. Fixing vmax to twice the decision
    threshold keeps colour comparable across images - normal parts stay cool,
    real defects saturate.
    """
    m = np.clip(amap / max(vmax, 1e-8), 0.0, 1.0)
    heat = (cm.inferno(m)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(img.resize(amap.shape[::-1]).convert("RGB")).astype(np.float32)
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))


st.title("🔍 Explainable Visual Defect Detector")
st.caption(
    "Trained on normal examples only — no defect was ever labelled for training. "
    "The heatmap shows which regions look unlike anything in the normal set."
)

available = sorted(p.stem for p in MODELS.glob("*.pt"))
if not available:
    st.error("No exported models found. Run `python src/edd/export.py` first.")
    st.stop()

with st.sidebar:
    category = st.selectbox("Object type", available)
    art = get_artifact(category)
    st.metric("Calibrated threshold", f"{art['threshold']:.3f}")
    st.caption(
        f"Set at the {1 - art['target_fpr']:.0%} quantile of {art['n_calib']} held-out "
        f"**normal** images — never on test data."
    )
    thr = st.slider("Decision threshold", 0.0, float(art["threshold"] * 2.5),
                    float(art["threshold"]), 0.01)
    st.divider()
    st.caption(
        f"Memory bank: {art['bank_size']:,} patches from {art['n_bank_images']} normal images "
        f"({art['coreset_frac']:.0%} coreset)."
    )

samples = sorted((ROOT / "assets" / "samples").glob(f"{category}__*.png"))
choice = st.radio(
    "Pick a sample", ["Upload my own"] + [p.stem.split("__", 1)[1] for p in samples],
    horizontal=True,
)

img = None
if choice == "Upload my own":
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    if up:
        img = Image.open(up).convert("RGB")
else:
    img = Image.open(next(p for p in samples if p.stem.endswith(f"__{choice}"))).convert("RGB")

if img is None:
    st.info("Upload an image or pick a sample above.")
    st.stop()

model, dev = get_model()
x = build_transform(art["size"], art["crop"])(img)
with st.spinner("Scoring…"):
    s, _flag, amap = predict(art, x, model, dev)

flag = s >= thr
c1, c2, c3 = st.columns(3)
c1.image(img, caption="input", use_container_width=True)
c2.image(overlay(img, amap, vmax=2 * thr),
         caption=f"anomaly map (absolute scale, 0 → {2 * thr:.1f})",
         use_container_width=True)
with c3:
    st.metric("Anomaly score", f"{s:.3f}", f"{s - thr:+.3f} vs threshold")
    # must be a statement: Streamlit's magic display renders the value of a
    # bare expression, so the ternary form prints a DeltaGenerator repr
    if flag:
        st.error("**DEFECT**")
    else:
        st.success("**OK**")
    st.progress(min(s / (thr * 2), 1.0))
    st.caption(
        "Score is the largest distance between any patch of this image and its nearest "
        "patch in the normal memory bank."
    )

with st.expander("How this works, and what it does not do"):
    st.markdown(
        f"""
**Method.** A frozen WideResNet50-2 turns the image into a 28×28 grid of 1536-d patch
embeddings. Every patch is compared to a memory bank built from **normal images only**
({art['bank_size']:,} patches kept by greedy k-center coreset selection). The image score
is the largest patch distance; the heatmap is the per-patch distance, upsampled and blurred.

**Honest limits.**
- Localisation quality is category-dependent. On `bottle` the hottest pixel lands inside
  the labelled defect 98% of the time; on `screw` it is 50%. Treat the heatmap as a
  pointer, not a segmentation.
- The bank encodes one object type under one lighting setup. Photograph a different
  object, or the same object under very different light, and everything looks anomalous.
- The threshold targets a 1% false-alarm rate on normal parts. Lowering it catches more
  defects and cries wolf more often — the slider is there so that trade-off is yours.
"""
    )
