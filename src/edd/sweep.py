"""Milestone 6 - run the headline config across all 15 MVTec categories.

Three categories is a demo; fifteen is a result. Reporting the full benchmark
is also the only way the weak categories stay visible - it removes the option
of quietly showing only the ones that worked.

One pass per category computes both detection and localisation metrics, so the
maps are scored against the masks they were actually produced from.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import baseline
from explain import localisation_metrics
from patchcore import fit_score

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def one(category: str, frac: float, crop: bool) -> dict:
    t0 = time.time()
    r = fit_score(category, frac, 224, "coreset", crop)
    maps = r["maps"].numpy()[:, 0]
    masks = r["masks"].numpy()[:, 0]
    labels = r["labels"]
    a = labels == 1

    det = baseline.evaluate(labels, r["img_scores"])
    loc = localisation_metrics(maps[a], masks[a], np.random.default_rng(0))

    out = {
        "category": category, "coreset_frac": frac, "center_crop": crop,
        "image_auroc": det["image_auroc"], "average_precision": det["average_precision"],
        "accuracy_at_best_f1": det["accuracy_at_best_f1"],
        "majority_class_accuracy": det["majority_class_accuracy"],
        "n_normal": det["n_normal"], "n_anomalous": det["n_anomalous"],
        **{k: v for k, v in loc["model"].items()},
        **{f"control_{k}": v for k, v in loc["random_control"].items()},
        "defect_pixel_fraction": loc["defect_pixel_fraction"],
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / "reports" / f"full_{category}.json").write_text(json.dumps(out, indent=1))
    print(f"{category:11} img {out['image_auroc']:.4f} | px {out['pixel_auroc']:.4f} "
          f"| aupro {out['aupro']:.4f} | peak {out['peak_in_mask']:.4f} "
          f"| {out['seconds']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="*", default=CATEGORIES)
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--no-crop", action="store_true")
    a = p.parse_args()
    for c in a.categories:
        one(c, a.frac, not a.no_crop)
