# Explainable Visual Defect Detector

Anomaly detection for industrial visual inspection: flag defective items from a
photo, and show **where** the defect is — trained on normal examples only.

> **Status:** in progress. Milestones 1–2 complete (data pipeline, EDA, baseline).
> Milestone 3 (PatchCore) in progress. Live demo not up yet.
> Numbers below are current and un-cherry-picked, including the bad ones.

## Why anomaly detection instead of a defect classifier

In real inspection settings, defects are rare, diverse, and expensive to label.
Normal parts are abundant. So the deployable framing is: *learn what normal looks
like, flag deviations* — not *collect 500 examples of every defect type you hope
to see.*

MVTec AD enforces this: `train/` contains **only** normal images; every defect
lives in `test/`. Training a supervised classifier here requires moving defects
out of the test set, contaminating the only clean evaluation split the benchmark
has. Measured on `bottle`:

| split | class | n |
|---|---|---|
| train | good | 209 |
| test | good | 20 |
| test | broken_large / broken_small / contamination | 20 / 22 / 21 |

## Results

Image-level AUROC. `PatchCore (paper)` is Roth et al., CVPR 2022, as a reference
point — the gap is shown rather than omitted.

| category | method | image AUROC | acc @F1 | majority acc | PatchCore (paper) | gap |
|---|---|---|---|---|---|---|
| bottle | kNN baseline | **0.9937** | 0.988 | 0.759 | 1.000 | −0.006 |
| pill | kNN baseline | **0.7534** | 0.868 | 0.844 | 0.966 | −0.213 |
| screw | kNN baseline | **0.7639** | 0.788 | 0.744 | 0.981 | −0.217 |

**Read the `majority acc` column.** On `screw`, 78.8% accuracy sits against a
74.4% majority-class baseline — about 4 points of real signal. Evaluating only on
`bottle` would have produced a 0.9937 headline and hidden this entirely. Easy and
hard categories are both reported for exactly that reason.

### Why the baseline fails where it fails

The baseline pools each image into a single 512-d vector via global average
pooling. A `screw` defect is a few-pixel scratch on a small object against a
large dark background, so it is averaged away before any distance is computed.
The same collapse is why this baseline **cannot localise** — there is no spatial
map left to point at. Patch-level features address both.

## Decisions made on measurement, not convention

- **Resize-only, no CenterCrop.** The stock ImageNet transform
  (`Resize(256) → CenterCrop(224)`) discards the outer 12.5% of each image. On
  `bottle` that clips defect area in 4/63 anomalous images, worst case losing
  14.7% of a labelled defect. Zero cost to avoid, so avoided.
  (`src/edd/dataset.py`, run it to reproduce.)
- **224×224 input.** The smallest defect in `bottle` covers 0.58% of pixels ≈ 289
  pixels at 224², still resolvable. Chosen from the mask statistics, not by default.
- **AUROC / AP over accuracy.** Test splits run ~75–84% anomalous; accuracy alone
  rewards a constant prediction. Median defect covers 5.7% of pixels, so *pixel*
  accuracy is worse still — an all-negative mask scores 92.4%.

## A data bug worth documenting

The official mvtec.com download link is dead (404), so data comes from the
Voxel51 HuggingFace mirror. That mirror flattens every image into shared shards
and appends a dedup suffix — and **the suffix differs between an image and its
own mask**: `000-94.png` vs `000_mask-67.png`. Pairing them by filename yields
**zero** masks, silently. A pipeline built on that would report image-level
metrics while claiming pixel-level localisation.

`fetch_mvtec.py` strips the suffix and asserts every anomalous test image has a
matching mask, failing the fetch otherwise. Verified: 63/63 (bottle), 119/119
(screw), 141/141 (pill).

## Reproduce

```bash
uv sync
python src/edd/fetch_mvtec.py              # list categories
python src/edd/fetch_mvtec.py bottle       # download + validate one
uv run python src/edd/eda.py bottle        # -> reports/eda_bottle.{md,png}
uv run python src/edd/baseline.py --self-check
uv run python src/edd/baseline.py bottle   # -> reports/baseline_bottle.json
uv run python src/edd/summarize.py         # -> reports/results.md
```

Images are not committed (~150 MB/category); `fetch_mvtec.py` rebuilds them from
the tracked index into the canonical MVTec layout.

## Roadmap

- [x] **1 — Data.** Fetch + layout reconstruction + mask-pairing validation, EDA.
- [x] **2 — Baseline.** Frozen ResNet18 embeddings + kNN distance, no training loop.
- [ ] **3 — PatchCore.** Patch-level features, coreset subsampling, anomaly maps.
- [ ] **4 — Explainability.** Anomaly maps scored against ground-truth masks with
      pixel AUROC — measured, not eyeballed.
- [ ] **5 — Deployment.** Streamlit + Docker + Hugging Face Spaces.
- [ ] **6 — Docs.** Architecture diagram, full 15-category run, honest write-up.

## Stack

PyTorch (MPS), torchvision, scikit-learn, NumPy, Pillow, matplotlib. Managed with `uv`.

## Data & licence

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad),
CC BY-NC-SA 4.0 — research/non-commercial use.
