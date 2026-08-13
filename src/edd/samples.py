"""Pick the demo's sample images using the SHIPPED artefact, not by eye.

The first sample set was chosen arbitrarily and two of the three defect samples
scored below their own deployed threshold - so the demo would have shown a
defective part labelled OK. That is not a bug: `screw` recall at the calibrated
operating point is 53.8%. But a visitor cannot tell an honest miss from a
broken demo.

So each category ships three samples, chosen by scoring the real test split
with the exported model:

  good              a normal part, comfortably below threshold
  <defect>          a defect the detector actually catches
  <defect>_MISSED   a defect it does NOT catch, labelled as such

Keeping the miss is the point. A demo that only shows wins is marketing; this
one shows the failure mode the metrics already report, and names it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


from dataset import MVTecCategory
from export import MODELS, load
from patchcore import PatchFeatures, device, extract, score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "samples"


def pick(category: str) -> dict:
    dev = device()
    model = PatchFeatures().to(dev)
    art = load(category, dev)
    thr = art["threshold"]

    ds = MVTecCategory(category, "test", art["size"], art["crop"])
    feats, labels, _, _, _ = extract(model, ds, dev)
    s = score(art["bank"], feats, dev).max(dim=1).values.numpy()

    rows = [
        {"path": ds.items[i][0], "label": int(labels[i]),
         "defect": ds.items[i][2], "score": float(s[i])}
        for i in range(len(labels))
    ]
    normals = sorted([r for r in rows if r["label"] == 0], key=lambda r: r["score"])
    defects = sorted([r for r in rows if r["label"] == 1], key=lambda r: -r["score"])

    chosen = []
    # a typical normal, not the very lowest - that would flatter the demo
    chosen.append(("good", normals[len(normals) // 2]))
    # the most confidently caught defect
    if defects and defects[0]["score"] >= thr:
        chosen.append((defects[0]["defect"], defects[0]))
    # the worst miss, if the model misses anything at this threshold
    missed = [d for d in defects if d["score"] < thr]
    if missed:
        chosen.append((f"{missed[-1]['defect']}_MISSED", missed[-1]))

    for name, r in chosen:
        shutil.copy(r["path"], OUT / f"{category}__{name}.png")

    meta = {
        "category": category, "threshold": thr,
        "n_defects": len(defects), "n_missed": len(missed),
        "recall_at_threshold": 1 - len(missed) / max(len(defects), 1),
        "samples": [
            {"name": n, "file": f"{category}__{n}.png", "score": round(r["score"], 3),
             "flagged": bool(r["score"] >= thr), "defect": r["defect"]}
            for n, r in chosen
        ],
    }
    for x in meta["samples"]:
        print(f"  {x['name']:26} score {x['score']:6.3f}  thr {thr:.3f}  "
              f"-> {'DEFECT' if x['flagged'] else 'OK'}")
    print(f"  recall at shipped threshold: {meta['recall_at_threshold']:.1%} "
          f"({len(missed)}/{len(defects)} missed)")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="*", default=None)
    a = p.parse_args()
    cats = a.categories or sorted(f.stem for f in MODELS.glob("*.pt"))

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()

    out = []
    for c in cats:
        print(f"=== {c} ===")
        out.append(pick(c))
    (ROOT / "assets" / "samples.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
