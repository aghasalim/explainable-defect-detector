# Explainable Visual Defect Detector

Anomaly detection for industrial visual inspection: flag defective items from a
photo, and show **where** the defect is — trained on normal examples only.

> **Status:** in progress. Milestones 1–3 complete (data pipeline, EDA, baseline,
> PatchCore + ablations). Milestone 4 (explainability verification) next.
> Live demo not up yet.
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

PatchCore, 1% coreset, paper preprocessing. Reference is Roth et al., CVPR 2022.

| category | image AUROC | paper | gap | pixel AUROC | paper | gap | runtime |
|---|---|---|---|---|---|---|---|
| bottle | **1.0000** | 1.000 | +0.000 | **0.9825** | 0.986 | −0.004 | 22 s |
| pill | **0.9569** | 0.966 | −0.009 | **0.9703** | 0.976 | −0.006 | 35 s |
| screw | **0.9412** | 0.981 | −0.040 | **0.9572** | 0.994 | −0.037 | 45 s |

Bottle and pill reproduce the paper within a point. `screw` remains 4 points short —
reported, not hidden. The likeliest remaining cause is the omitted score reweighting
(see *Deviations* below); no attempt was made to close the gap by tuning on the test set.

### Baseline → PatchCore

| category | kNN baseline | PatchCore | Δ |
|---|---|---|---|
| bottle | 0.9937 | **1.0000** | +0.006 |
| pill | 0.7534 | **0.9569** | +0.204 |
| screw | 0.7639 | **0.9412** | +0.177 |

The baseline was near-useless exactly where it mattered (`screw`: 78.8% accuracy against
a 74.4% majority-class baseline). Keeping the spatial grid instead of average-pooling it
away recovers ~18–20 AUROC points and makes localisation possible at all.

### Ablations

Coreset selection is not a formality — at an identical bank size on `screw`:

| sampling | bank | image AUROC |
|---|---|---|
| random 1% | 2,508 | 0.5518 |
| **greedy k-center 1%** | 2,508 | **0.8737** |

Random sampling collapses to near-chance. The image score is a *max* over patch
distances, so it is hostage to coverage: random sampling misses rare-but-normal patches,
those score as distant, and normal images get flagged. Note pixel AUROC barely moves
(0.9607 vs 0.9642) — it is dominated by easy background and hides the failure entirely.
Two metrics, opposite conclusions.

Bank size has real but sharply diminishing returns (`screw`, resize preprocessing):

| coreset | bank | image AUROC | runtime |
|---|---|---|---|
| 1% | 2,508 | 0.8737 | 44 s |
| 5% | 12,544 | 0.9182 | 179 s |
| 10% | 25,088 | 0.9289 | 343 s |

**Preprocessing beat all of it.** `screw` with the paper's `Resize(256)+CenterCrop(224)`
scores 0.9412 at 1% in 45 s — better than 10% coreset at 1/7 the compute. The crop zooms
in on a centred object, and `screw` defects are tiny. This directly contradicts the
resize-only choice justified on `bottle`: the right preprocessing is category-dependent,
and the measurement that settled it for one object did not transfer.

## Decisions made on measurement, not convention

- **Preprocessing, measured then revised.** The stock ImageNet transform
  (`Resize(256) → CenterCrop(224)`) discards the outer 12.5% of each image. On `bottle`
  that clips defect area in 4/63 anomalous images, worst case losing 14.7% of a labelled
  defect — so resize-only looked strictly better. On `screw` the same crop is worth
  **+6.8 AUROC** because it magnifies a tiny defect. Both are reported rather than
  picking the winner per category, which would be tuning on the test set.
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

## Deviations from the paper, stated plainly

- **Image score is a plain max** over patch distances. PatchCore additionally reweights
  that max by how isolated the matched bank point is. This is the most likely source of
  the remaining `screw` gap.
- **Greedy k-center runs in a 128-d Johnson–Lindenstrauss projection** (as the paper
  does) for speed; the bank itself keeps full 1536-d vectors.
- **No test-set tuning.** MVTec ships no validation split. The headline row is fixed to
  the paper's protocol (1% coreset, paper preprocessing) for comparability; every other
  configuration is reported as an ablation, not selected as a result.
- **Pixel AUROC is not comparable across preprocessing rows** — under `crop` it is
  computed over a different, zoomed pixel set.

## Reproduce

```bash
uv sync
python src/edd/fetch_mvtec.py              # list categories
python src/edd/fetch_mvtec.py bottle       # download + validate one
uv run python src/edd/eda.py bottle        # -> reports/eda_bottle.{md,png}
uv run python src/edd/baseline.py --self-check
uv run python src/edd/baseline.py bottle   # -> reports/baseline_bottle.json
uv run python src/edd/patchcore.py --self-check
uv run python src/edd/patchcore.py bottle --crop      # headline config
uv run python src/edd/patchcore.py screw --sampling random   # ablation
uv run python src/edd/patchcore.py screw --frac 0.10         # ablation
uv run python src/edd/summarize.py         # -> reports/results.md
```

Images are not committed (~150 MB/category); `fetch_mvtec.py` rebuilds them from
the tracked index into the canonical MVTec layout.

## Roadmap

- [x] **1 — Data.** Fetch + layout reconstruction + mask-pairing validation, EDA.
- [x] **2 — Baseline.** Frozen ResNet18 embeddings + kNN distance, no training loop.
- [x] **3 — PatchCore.** Patch-level features, greedy k-center coreset, anomaly maps,
      sampling/bank-size/preprocessing ablations.
- [ ] **4 — Explainability.** Anomaly maps scored against ground-truth masks with
      pixel AUROC — measured, not eyeballed.
- [ ] **5 — Deployment.** Streamlit + Docker + Hugging Face Spaces.
- [ ] **6 — Docs.** Architecture diagram, full 15-category run, honest write-up.

## Stack

PyTorch (MPS), torchvision, scikit-learn, NumPy, Pillow, matplotlib. Managed with `uv`.

## Data & licence

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad),
CC BY-NC-SA 4.0 — research/non-commercial use.
