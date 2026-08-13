# Results

## Headline: PatchCore, 1% coreset, paper preprocessing

| category | image AUROC | paper | gap | pixel AUROC | paper | gap | sec |
|---|---|---|---|---|---|---|---|
| bottle | **1.0000** | 1.000 | +0.0000 | **0.9825** | 0.986 | -0.0035 | 22 |
| pill | **0.9569** | 0.966 | -0.0091 | **0.9703** | 0.976 | -0.0057 | 35 |
| screw | **0.9412** | 0.981 | -0.0398 | **0.9572** | 0.994 | -0.0368 | 45 |

## All runs

| category | method | preproc | coreset | bank | image AUROC | pixel AUROC | acc @F1 | majority acc | sec |
|---|---|---|---|---|---|---|---|---|---|
| bottle | pc-224crop | crop | 1% | 1,638 | **1.0000** | 0.9825 | 1.0000 | 0.7590 | 22 |
| bottle | pc | resize | 1% | 1,638 | **1.0000** | 0.9851 | 1.0000 | 0.7590 | 26 |
| bottle | baseline | resize | - | - | **0.9937** | nan | 0.9880 | 0.7590 | 0 |
| pill | pc-224crop | crop | 1% | 2,093 | **0.9569** | 0.9703 | 0.9341 | 0.8443 | 35 |
| pill | pc | resize | 1% | 2,093 | **0.9504** | 0.9836 | 0.9341 | 0.8443 | 39 |
| pill | baseline | resize | - | - | **0.7534** | nan | 0.8683 | 0.8443 | 0 |
| screw | pc-224crop | crop | 1% | 2,508 | **0.9412** | 0.9572 | 0.9187 | 0.7438 | 45 |
| screw | pc-0.1 | resize | 10% | 25,088 | **0.9289** | 0.9832 | 0.8750 | 0.7438 | 343 |
| screw | pc-0.05 | resize | 5% | 12,544 | **0.9182** | 0.9823 | 0.8750 | 0.7438 | 179 |
| screw | pc | resize | 1% | 2,508 | **0.8737** | 0.9642 | 0.8125 | 0.7438 | 44 |
| screw | baseline | resize | - | - | **0.7639** | nan | 0.7875 | 0.7438 | 0 |
| screw | pc-random | resize | 1% | 2,508 | **0.5518** | 0.9607 | 0.7438 | 0.7438 | 13 |

`acc @F1` vs `majority acc` is the sanity column: where they are close, the model is barely beating a constant prediction, whatever the AUROC says.

Pixel AUROC is **not comparable across preprocessing rows**: under `crop` it is computed over a different (zoomed, smaller) pixel set than under `resize`. Compare pixel numbers only within the same preprocessing column.
