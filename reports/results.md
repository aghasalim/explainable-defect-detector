# Results

| category | method | image AUROC | AP | best F1 | acc @F1 | majority acc | PatchCore (paper) | gap |
|---|---|---|---|---|---|---|---|---|
| bottle | baseline | **0.9937** | 0.9979 | 0.9921 | 0.9880 | 0.7590 | 1.000 | -0.0063 |
| pill | baseline | **0.7534** | 0.9404 | 0.9262 | 0.8683 | 0.8443 | 0.966 | -0.2126 |
| screw | baseline | **0.7639** | 0.9018 | 0.8722 | 0.7875 | 0.7438 | 0.981 | -0.2171 |

`acc @F1` vs `majority acc` is the sanity column: where they are close, the model is barely beating a constant prediction, whatever the AUROC says.

Categories within 5 points of the majority-class baseline: `pill`, `screw`.
