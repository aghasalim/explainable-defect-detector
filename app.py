"""Streamlit demo for the Explainable Visual Defect Detector.

Loads an exported memory bank (models/<category>.pt), scores an uploaded image
against it, and shows the anomaly map beside the verdict.

The threshold shown is calibrated on NORMAL training images only - k-fold
cross-calibration plus a one-sided tolerance bound at a 1% false-alarm target,
never tuned on the test set. The sidebar exposes it so the trade-off is visible
rather than hidden behind a single word.

Everything user-facing here reads from the artefact or from assets/samples.json
rather than being written into the copy. Two separate captions in this file
went stale the moment the calibration changed, and each one kept confidently
describing a method the app was no longer using.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from matplotlib import cm
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "edd"))

from dataset import build_transform
from export import MODELS, predict
from patchcore import PatchFeatures

st.set_page_config(page_title="Explainable Defect Detector", page_icon="🔍", layout="wide")


@st.cache_resource
def get_model():
    dev = torch.device("cpu")   # Spaces free tier is CPU-only
    return PatchFeatures().to(dev), dev


@st.cache_data
def _recall(category: str) -> dict | None:
    """Per-category recall at the shipped threshold, from assets/samples.json."""
    f = ROOT / "assets" / "samples.json"
    if not f.exists():
        return None
    for row in json.loads(f.read_text()):
        if row.get("category") == category:
            return row
    return None


def _stamp(category: str) -> tuple[int, int]:
    p = MODELS / f"{category}.pt"
    st_ = p.stat()
    return st_.st_size, int(st_.st_mtime)


@st.cache_resource
def _load_artifact(category: str, _stamp_key: tuple[int, int]):
    a = torch.load(MODELS / f"{category}.pt", weights_only=False)
    a["bank"] = a["bank"].float()
    return a


def get_artifact(category: str):
    """Load an artefact, keyed on the file's size+mtime.

    st.cache_resource keys on the function's code and arguments, not on the
    file it happens to read. Re-exporting a model therefore left the deployed
    app serving the previous memory bank and threshold indefinitely - it showed
    the old calibration for hours after the new one was pushed, and only a
    manual reboot cleared it. Feeding the file stamp in as an argument makes a
    changed artefact a cache miss.
    """
    return _load_artifact(category, _stamp(category))


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


st.title("Explainable Visual Defect Detector")
st.caption(
    "Trained on normal examples only, no defect was ever labelled for training. "
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
    # Describe whichever method actually produced this artefact, rather than
    # hardcoding one - the calibration changed once already and the caption
    # silently kept claiming the old method.
    if art.get("threshold_method") == "tolerance_bound":
        guaranteed = art.get("guarantee_met")
        st.caption(
            f"A **tolerance bound**, not a quantile: at least "
            f"{1 - art['target_fpr']:.0%} of normal parts score below this"
            + (f", with {art.get('tolerance_confidence', 0.95):.0%} confidence."
               if guaranteed else
               f". {art['n_calib']} calibration images is short of the "
               f"{art.get('n_required_for_guarantee', 299)} needed for a formal "
               f"{art.get('tolerance_confidence', 0.95):.0%} guarantee, so this falls back "
               f"to their maximum.")
        )
    else:
        st.caption(
            f"Set at the {1 - art['target_fpr']:.0%} quantile of {art['n_calib']} "
            f"**normal** images, never on test data."
        )
    st.caption(
        f"Calibrated on {art['n_calib']} normal training images"
        + (f" ({art['k_folds']}-fold cross-calibration)." if art.get("k_folds") else ".")
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

# Samples are chosen by scoring the real test split with this exact artefact
# (src/edd/samples.py), so a "_MISSED" sample is a genuine false negative at
# the calibrated threshold rather than a broken demo. Saying so is the whole
# point of the project.
if choice.endswith("_MISSED"):
    # Recall is read from assets/samples.json, which samples.py writes from the
    # real test split. Hardcoding it here went stale the moment the calibration
    # changed, and the app cheerfully quoted the old numbers.
    rec = _recall(category)
    detail = (
        f"At this operating point it catches **{rec['recall_at_threshold']:.1%}** of "
        f"`{category}` defects ({rec['n_missed']} of {rec['n_defects']} missed). "
        if rec else ""
    )
    st.warning(
        f"**This is a known miss.** At the calibrated threshold this defect scores *below* "
        f"the line, so the detector reports OK, a false negative. {detail}"
        f"Drag the **Decision threshold** slider down and watch it flip to DEFECT, and "
        f"watch normal parts start tripping too. That trade-off is the actual engineering "
        f"problem, and it is not visible in an AUROC score."
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
  defects and cries wolf more often, so the slider is there and the trade-off is yours.
"""
    )
