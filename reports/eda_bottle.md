# EDA - MVTec AD `bottle`

292 images.

## Class balance

| split | class | n |
|---|---|---|
| test | broken_large | 20 |
| test | broken_small | 22 |
| test | contamination | 21 |
| test | good | 20 |
| train | good | 209 |

**Defective images in `train`: 0.** Supervised binary classification is therefore impossible without moving defects out of `test`, which contaminates the only clean evaluation set the benchmark has. This is the single most important fact in this dataset.

**Test set is 20 good / 63 defective.** A model predicting the majority class scores 75.9% accuracy while catching nothing. Report AUROC and precision/recall; accuracy is not a defensible headline number here.

## Defect size (fraction of pixels marked defective)

63 masks found.

| stat | value |
|---|---|
| min | 0.5757% |
| p25 | 2.5741% |
| median | 5.7178% |
| p75 | 12.8291% |
| max | 27.4186% |
| mean | 7.6166% |

| defect type | n | median area |
|---|---|---|
| broken_large | 20 | 11.8670% |
| broken_small | 22 | 2.4370% |
| contamination | 21 | 7.2479% |

The median defect covers **5.72%** of the image. Predicting 'no defect' for every pixel already yields ~92.38% pixel accuracy, so pixel accuracy is useless - use pixel AUROC / PRO. It also bounds how far we can downsample: at 224x224 the smallest defect here occupies about 288.9 pixels.

## Resolution & channels

| property | values |
|---|---|
| size | [(900, 900)] |
| mode | ['RGB'] |

## Exact duplicates

None. No train/test contamination via identical files.


## Exposure confound

Mean grey level - good 136.5+/-1.4, defective 137.6+/-2.2 (difference 1.2).

If this gap were large, a model could separate the classes on global brightness alone and the heatmaps would be meaningless - worth re-checking on self-collected data, where lighting is not controlled.
