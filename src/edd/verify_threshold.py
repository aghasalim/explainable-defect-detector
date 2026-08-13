"""Does the calibrated operating point actually behave in production?

export.py sets the threshold from held-out NORMAL images at a 1% false-alarm
target, never touching test data. That is the correct way to choose it - but
it is a prediction, not a result. This script checks the prediction against the
real test split: if the realised false-alarm rate is far above 1%, the
calibration set was not representative and the demo would cry wolf.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset import MVTecCategory
from export import MODELS, load
from patchcore import PatchFeatures, device, extract, score

ROOT = Path(__file__).resolve().parents[2]

rows = []
dev = device()
model = PatchFeatures().to(dev)
for f in sorted(MODELS.glob("*.pt")):
    art = load(f.stem, dev)
    ds = MVTecCategory(art["category"], "test", art["size"], art["crop"])
    feats, labels, _, _, _ = extract(model, ds, dev)
    s = score(art["bank"], feats, dev).max(dim=1).values.numpy()
    flag = s >= art["threshold"]
    fpr = float(flag[labels == 0].mean())
    rec = float(flag[labels == 1].mean())
    rows.append({"category": art["category"], "threshold": art["threshold"],
                 "target_fpr": art["target_fpr"], "realised_fpr_on_test": fpr,
                 "recall_on_test": rec, "n_normal": int((labels == 0).sum()),
                 "n_anomalous": int((labels == 1).sum())})
    print(f"{art['category']:9} thr {art['threshold']:.3f} | target FPR {art['target_fpr']:.0%} "
          f"| realised FPR {fpr:.1%} ({int(flag[labels==0].sum())}/{(labels==0).sum()}) "
          f"| recall {rec:.1%}")

(ROOT / "reports" / "threshold_check.json").write_text(json.dumps(rows, indent=1))
