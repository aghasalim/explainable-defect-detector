---
title: Explainable Visual Defect Detector
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Explainable Visual Defect Detector

Industrial visual inspection by **anomaly detection**: the model is built from
normal examples only — no defect was ever labelled for training — and every
prediction comes with a heatmap showing which regions look unlike normal.

Pick a sample or upload your own image of a bottle, screw, or pill.

- **Method:** PatchCore (Roth et al., CVPR 2022), implemented from scratch.
- **Detection:** 0.9874 mean image AUROC across all 15 MVTec AD categories.
- **The threshold** is calibrated on held-out *normal* images at a 1% false-alarm
  target — never on test data. The sidebar slider exposes the trade-off.

Full code, ablations, and honest failure analysis:
**https://github.com/aghasalim/explainable-defect-detector**

Data: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CC BY-NC-SA 4.0).
