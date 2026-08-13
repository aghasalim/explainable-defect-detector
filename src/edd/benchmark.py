"""Milestone 6 - the full 15-category MVTec AD run.

Three categories is a demo; the whole benchmark is a result. Reporting all 15
also removes the option of quietly leading with the easy ones.

Each category is fitted ONCE and scored for both detection and localisation,
rather than running patchcore.py and explain.py separately over the same work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baseline import evaluate
from explain import localisation_metrics
from patchcore import fit_score
from summarize import PAPER

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = sorted(PAPER)


def one(category: str, frac: float, crop: bool) -> dict:
    r = fit_score(category, frac, 224, "coreset", crop)
    maps = r["maps"].numpy()[:, 0]
    masks = r["masks"].numpy()[:, 0]
    labels = r["labels"]

    det = evaluate(labels, r["img_scores"])
    a = labels == 1
    loc = localisation_metrics(maps[a], masks[a], np.random.default_rng(0))

    pi, pp = PAPER[category]
    out = {
        "category": category, "coreset_frac": frac, "center_crop": crop,
        "n_test": int(len(labels)), "n_anomalous": int(a.sum()),
        "image_auroc": det["image_auroc"], "average_precision": det["average_precision"],
        "accuracy_at_best_f1": det["accuracy_at_best_f1"],
        "majority_class_accuracy": det["majority_class_accuracy"],
        "paper_image_auroc": pi, "paper_pixel_auroc": pp,
        "seconds": r["seconds"],
        **{k: v for k, v in loc["model"].items()},
        **{f"control_{k}": v for k, v in loc["random_control"].items()},
        "defect_pixel_fraction": loc["defect_pixel_fraction"],
    }
    (ROOT / "reports" / f"bench_{category}.json").write_text(json.dumps(out, indent=1))
    return out


def table(rows: list[dict]) -> str:
    o = ["# Full MVTec AD benchmark\n",
         "PatchCore, 1% coreset, paper preprocessing, frozen WideResNet50-2. "
         "`paper` columns are Roth et al., CVPR 2022.\n",
         "## Detection (image level)\n",
         "| category | image AUROC | paper | gap | AP | acc @F1 | majority acc |",
         "|" + "---|" * 7]
    for r in rows:
        o.append(
            f"| {r['category']} | **{r['image_auroc']:.4f}** | {r['paper_image_auroc']:.3f} "
            f"| {r['image_auroc'] - r['paper_image_auroc']:+.4f} "
            f"| {r['average_precision']:.4f} | {r['accuracy_at_best_f1']:.4f} "
            f"| {r['majority_class_accuracy']:.4f} |")
    mi = np.mean([r["image_auroc"] for r in rows])
    mp = np.mean([r["paper_image_auroc"] for r in rows])
    o.append(f"| **mean** | **{mi:.4f}** | {mp:.3f} | {mi - mp:+.4f} | | | |")

    o += ["\n## Localisation (pixel level, anomalous images only)\n",
          "Every column is paired with a random-map control on the same images. Without it, "
          "a high pixel AUROC is unfalsifiable.\n",
          "| category | pixel AUROC | ctrl | AUPRO | ctrl | peak-in-mask | ctrl "
          "| top-1% prec | ctrl | defect px |",
          "|" + "---|" * 10]
    for r in rows:
        o.append(
            f"| {r['category']} | {r['pixel_auroc']:.4f} | {r['control_pixel_auroc']:.3f} "
            f"| **{r['aupro']:.4f}** | {r['control_aupro']:.3f} "
            f"| **{r['peak_in_mask']:.4f}** | {r['control_peak_in_mask']:.3f} "
            f"| {r['top1pct_precision']:.4f} | {r['control_top1pct_precision']:.3f} "
            f"| {r['defect_pixel_fraction']:.4f} |")
    for k in ("pixel_auroc", "aupro", "peak_in_mask", "top1pct_precision"):
        pass
    o.append(
        f"| **mean** | {np.mean([r['pixel_auroc'] for r in rows]):.4f} | | "
        f"**{np.mean([r['aupro'] for r in rows]):.4f}** | | "
        f"**{np.mean([r['peak_in_mask'] for r in rows]):.4f}** | | "
        f"{np.mean([r['top1pct_precision'] for r in rows]):.4f} | | |")

    worst = min(rows, key=lambda r: r["peak_in_mask"])
    best = max(rows, key=lambda r: r["peak_in_mask"])
    o.append(
        f"\n**Localisation is not uniform.** Peak-in-mask ranges from "
        f"{worst['peak_in_mask']:.0%} (`{worst['category']}`, defects cover "
        f"{worst['defect_pixel_fraction']:.2%} of the image) to {best['peak_in_mask']:.0%} "
        f"(`{best['category']}`). Pixel AUROC hides this: `{worst['category']}` still scores "
        f"{worst['pixel_auroc']:.4f} there, because the metric is dominated by easy "
        f"background. A single headline number for 'explainability' would be misleading.\n")
    return "\n".join(o)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--no-crop", action="store_true")
    a = p.parse_args()

    rows, t0 = [], time.time()
    for i, c in enumerate(CATEGORIES, 1):
        r = one(c, a.frac, not a.no_crop)
        rows.append(r)
        print(f"[{i:2}/15] {c:11} image {r['image_auroc']:.4f} "
              f"(paper {r['paper_image_auroc']:.3f})  aupro {r['aupro']:.4f}  "
              f"peak {r['peak_in_mask']:.3f}  {r['seconds']:.0f}s", flush=True)

    md = ROOT / "reports" / "benchmark.md"
    md.write_text(table(rows))
    print(f"\nmean image AUROC {np.mean([r['image_auroc'] for r in rows]):.4f} "
          f"(paper {np.mean([r['paper_image_auroc'] for r in rows]):.4f})")
    print(f"total {time.time() - t0:.0f}s -> {md}")


if __name__ == "__main__":
    main()
