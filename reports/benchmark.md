# Full MVTec AD benchmark

PatchCore, 1% coreset, paper preprocessing, frozen WideResNet50-2. `paper` columns are Roth et al., CVPR 2022.

## Detection (image level)

| category | image AUROC | paper | gap | AP | acc @F1 | majority acc |
|---|---|---|---|---|---|---|
| bottle | **1.0000** | 1.000 | +0.0000 | 1.0000 | 1.0000 | 0.7590 |
| cable | **0.9983** | 0.993 | +0.0053 | 0.9990 | 0.9867 | 0.6133 |
| capsule | **0.9773** | 0.980 | -0.0027 | 0.9949 | 0.9545 | 0.8258 |
| carpet | **0.9904** | 0.987 | +0.0034 | 0.9972 | 0.9658 | 0.7607 |
| grid | **0.9699** | 0.981 | -0.0111 | 0.9907 | 0.9487 | 0.7308 |
| hazelnut | **1.0000** | 1.000 | +0.0000 | 1.0000 | 1.0000 | 0.6364 |
| leather | **1.0000** | 1.000 | +0.0000 | 1.0000 | 1.0000 | 0.7419 |
| metal_nut | **0.9990** | 0.998 | +0.0010 | 0.9998 | 0.9913 | 0.8087 |
| pill | **0.9569** | 0.966 | -0.0091 | 0.9921 | 0.9341 | 0.8443 |
| screw | **0.9412** | 0.981 | -0.0398 | 0.9804 | 0.9187 | 0.7438 |
| tile | **0.9917** | 0.987 | +0.0047 | 0.9972 | 0.9829 | 0.7179 |
| toothbrush | **1.0000** | 1.000 | +0.0000 | 1.0000 | 1.0000 | 0.7143 |
| transistor | **1.0000** | 1.000 | +0.0000 | 1.0000 | 1.0000 | 0.6000 |
| wood | **0.9895** | 0.992 | -0.0025 | 0.9971 | 0.9747 | 0.7595 |
| zipper | **0.9968** | 0.985 | +0.0118 | 0.9991 | 0.9868 | 0.7881 |
| **mean** | **0.9874** | 0.990 | -0.0026 | | | |

## Localisation (pixel level)

Pixel AUROC and AUPRO use every pixel of the whole test split, normal images included, as in the MVTec AD protocol; AUPRO is integrated up to FPR 0.3. Peak-in-mask and top-1% precision use the defective images only. Every column is paired with a random-map control on the same images. Without it, a high pixel AUROC is unfalsifiable.

| category | pixel AUROC | ctrl | AUPRO | ctrl | peak-in-mask | ctrl | top-1% prec | ctrl | defect px |
|---|---|---|---|---|---|---|---|---|---|
| bottle | 0.9830 | 0.499 | **0.9295** | 0.149 | **0.9841** | 0.095 | 0.9398 | 0.096 | 0.0991 |
| cable | 0.9841 | 0.500 | **0.9255** | 0.148 | **0.9239** | 0.109 | 0.8321 | 0.063 | 0.0611 |
| capsule | 0.9863 | 0.501 | **0.9172** | 0.159 | **0.6881** | 0.018 | 0.4258 | 0.014 | 0.0145 |
| carpet | 0.9880 | 0.499 | **0.9333** | 0.150 | **0.8090** | 0.011 | 0.6892 | 0.026 | 0.0269 |
| grid | 0.9689 | 0.499 | **0.8830** | 0.151 | **0.6316** | 0.018 | 0.4235 | 0.011 | 0.0114 |
| hazelnut | 0.9841 | 0.499 | **0.9355** | 0.148 | **0.8571** | 0.057 | 0.7267 | 0.043 | 0.0437 |
| leather | 0.9907 | 0.500 | **0.9582** | 0.148 | **0.8913** | 0.011 | 0.5161 | 0.011 | 0.0111 |
| metal_nut | 0.9853 | 0.500 | **0.9194** | 0.152 | **0.9462** | 0.194 | 0.8757 | 0.189 | 0.1884 |
| pill | 0.9741 | 0.501 | **0.9267** | 0.154 | **0.6950** | 0.035 | 0.5690 | 0.053 | 0.0519 |
| screw | 0.9716 | 0.499 | **0.8911** | 0.150 | **0.4958** | 0.000 | 0.2653 | 0.004 | 0.0043 |
| tile | 0.9554 | 0.500 | **0.7937** | 0.153 | **0.9048** | 0.143 | 0.8350 | 0.115 | 0.1153 |
| toothbrush | 0.9842 | 0.501 | **0.8187** | 0.147 | **0.5667** | 0.000 | 0.4955 | 0.026 | 0.0272 |
| transistor | 0.9710 | 0.500 | **0.9458** | 0.149 | **0.9500** | 0.150 | 0.7173 | 0.158 | 0.1552 |
| wood | 0.9393 | 0.499 | **0.8307** | 0.151 | **0.9000** | 0.100 | 0.7218 | 0.057 | 0.0600 |
| zipper | 0.9829 | 0.501 | **0.9298** | 0.150 | **0.9664** | 0.042 | 0.8679 | 0.034 | 0.0328 |
| **mean** | 0.9766 | | **0.9025** | | **0.8140** | | 0.6601 | | |

**Localisation is not uniform.** Peak-in-mask ranges from 50% (`screw`, defects cover 0.43% of the image) to 98% (`bottle`). Pixel AUROC hides this: `screw` still scores 0.9716 there, because the metric is dominated by easy background. A single headline number for 'explainability' would be misleading.
