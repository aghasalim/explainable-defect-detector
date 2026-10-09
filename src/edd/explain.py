"""Verify the explanations instead of admiring them.

A heatmap that lights up in the wrong place is worse than no heatmap, because
it looks like evidence. MVTec ships pixel-level ground truth, so "does the
explanation point at the defect" is a measurable question, not a vibe check.

Four measurements, each answering something the others cannot:

  pixel AUROC   ranking quality over every pixel. Dominated by easy background,
                so it stays high even when the map is bad - reported, but never
                alone.
  AUPRO         per-REGION overlap up to 30% FPR, the standard MVTec
                localisation metric, computed as the MVTec AD code does. Weights every defect region equally, so a
                small defect counts as much as a large one; pixel AUROC lets
                one big region carry the score.
  peak-in-mask  does the single hottest pixel land inside the defect? This is
                what a human actually reads off a heatmap.
  top-1% prec.  of the 1% hottest pixels, how many are really defect?

Every one of them is reported against a RANDOM-MAP CONTROL on the same images.
Without the control, "0.96 pixel AUROC" is unfalsifiable - the control shows
what the metric returns for a map that knows nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from patchcore import fit_score
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from style import PALETTE

# Fixed meanings, so the same idea keeps the same colour across every figure.
NEUTRAL = PALETTE[5]   # good, or nothing wrong
DEFECT = PALETTE[1]    # defective
MASK = PALETTE[2]      # the ground truth outline

ROOT = Path(__file__).resolve().parents[2]


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, divided by it.

    This follows the MVTec AD evaluation code (Bergmann et al., IJCV 2021) and
    gives the same numbers as anomalib's AUPRO:

      - regions are 8-connected components of each ground-truth mask, and each
        region carries the same weight whatever its size
      - FPR counts every defect-free pixel of every image passed in, so normal
        test images belong in `maps` and `masks` as well
      - the curve is exact: one point per distinct score, no threshold grid
      - it is integrated with the trapezoid rule up to fpr_limit, with the last
        point linearly interpolated at fpr_limit, then divided by fpr_limit

    A detector that finds one large defect and misses ten small ones should not
    look good, and this metric makes sure it does not.
    """
    maps = np.asarray(maps, dtype=np.float64)
    fp = np.zeros(maps.shape)
    pro = np.zeros(maps.shape)
    n_regions = 0
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m > 0, structure=np.ones((3, 3), dtype=int))
        fp[i][lab == 0] = 1.0
        for r in range(1, n + 1):
            region = lab == r
            pro[i][region] = 1.0 / region.sum()
        n_regions += n
    if n_regions == 0:
        return float("nan")
    fp /= fp.sum()
    pro /= n_regions

    order = np.argsort(maps.ravel(), kind="stable")[::-1]
    s = maps.ravel()[order]
    fprs = np.cumsum(fp.ravel()[order])
    pros = np.cumsum(pro.ravel()[order])
    # tied scores cross the threshold together, so keep the last of each run
    keep = np.append(np.diff(s) != 0, True)
    fprs = np.concatenate([[0.0], np.clip(fprs[keep], 0, 1)])
    pros = np.concatenate([[0.0], np.clip(pros[keep], 0, 1)])
    return float(_trapezoid_to(fprs, pros, fpr_limit) / fpr_limit)


def _trapezoid_to(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    """Trapezoid area under y(x) on [0, x_max]; x is sorted ascending."""
    j = np.searchsorted(x, x_max, side="right")
    if j == len(x):
        return float(np.trapezoid(y, x))
    # linear interpolation for the point at x_max
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    y_end = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
    return float(np.trapezoid(np.append(y[:j], y_end), np.append(x[:j], x_max)))


def localisation_metrics(maps: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
                         defective: np.ndarray | None = None) -> dict:
    """All four measurements, plus the random-map control for each.

    Pass the whole test split and `defective = labels == 1`. Pixel AUROC and
    AUPRO are computed over every pixel of every image, normal images included,
    as in the MVTec AD protocol. Peak-in-mask and top-1% precision only mean
    something on a defective image, so they use those alone. A defective image
    whose defect the centre crop removed still counts there, as a miss.
    Without `defective`, every image passed in is taken to be defective.
    """
    has_defect = np.ones(len(maps), bool) if defective is None else np.asarray(defective, bool)

    def suite(m: np.ndarray) -> dict:
        peak = []
        top1 = []
        for mm, gt in zip(m[has_defect], masks[has_defect], strict=True):
            yx = np.unravel_index(np.argmax(mm), mm.shape)
            peak.append(bool(gt[yx] > 0))
            k = max(1, int(mm.size * 0.01))
            idx = np.argpartition(mm.ravel(), -k)[-k:]
            top1.append(float((gt.ravel()[idx] > 0).mean()))
        return {
            "pixel_auroc": float(roc_auc_score(masks.ravel() > 0, m.ravel())),
            "aupro": aupro(m, masks),
            "peak_in_mask": float(np.mean(peak)),
            "top1pct_precision": float(np.mean(top1)),
        }

    real = suite(maps)
    control = suite(rng.random(maps.shape).astype(np.float32))
    return {
        "model": real,
        "random_control": control,
        "defect_pixel_fraction": float((masks[has_defect] > 0).mean()),
    }


def figure(paths, maps, masks, labels, scores, out: Path, n: int = 6) -> None:
    """image | ground truth | anomaly map | overlay, for the most confident hits.

    Deliberately includes the WORST-scoring anomalous image as the last row:
    showing only wins is how a portfolio project stops being evidence.
    """
    anom = [i for i, l in enumerate(labels) if l == 1]
    order = sorted(anom, key=lambda i: -scores[i])
    pick = order[: n - 1] + [order[-1]]

    fig, ax = plt.subplots(len(pick), 4, figsize=(11, 2.7 * len(pick)))
    for r, i in enumerate(pick):
        img = Image.open(paths[i]).convert("RGB").resize(maps.shape[1:][::-1])
        worst = r == len(pick) - 1
        ax[r, 0].imshow(img)
        ax[r, 0].set_ylabel(
            ("WORST\n" if worst else "") + Path(paths[i]).parent.name, fontsize=8,
            color=DEFECT if worst else "black")
        ax[r, 1].imshow(masks[i], cmap="gray")
        ax[r, 2].imshow(maps[i], cmap="inferno")
        ax[r, 3].imshow(img)
        ax[r, 3].imshow(maps[i], cmap="inferno", alpha=0.5)
        ax[r, 3].contour(masks[i] > 0, levels=[0.5], colors=[MASK], linewidths=1.2)
        for c in range(4):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
    for c, t in enumerate(["input", "ground truth", "anomaly map", "overlay (GT = green)"]):
        ax[0, c].set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run(category: str, crop: bool = True, frac: float = 0.01) -> dict:
    r = fit_score(category, frac, 224, "coreset", crop)
    maps = r["maps"].numpy()[:, 0]
    masks = r["masks"].numpy()[:, 0]
    labels, paths, scores = r["labels"], r["paths"], r["img_scores"]

    met = localisation_metrics(maps, masks, np.random.default_rng(0), labels == 1)
    met["category"] = category
    met["n_anomalous"] = int((labels == 1).sum())

    figure(paths, maps, masks, labels, scores, ROOT / "reports" / f"explain_{category}.png")
    (ROOT / "reports" / f"explain_{category}.json").write_text(json.dumps(met, indent=1))

    print(f"=== {category} ({met['n_anomalous']} anomalous images) ===")
    print(f"{'metric':20} {'model':>9} {'random':>9}")
    for k in ("pixel_auroc", "aupro", "peak_in_mask", "top1pct_precision"):
        print(f"{k:20} {met['model'][k]:9.4f} {met['random_control'][k]:9.4f}")
    print(f"{'defect px fraction':20} {met['defect_pixel_fraction']:9.4f}")
    return met


def demo() -> None:
    """Self-check: the metrics must fail loudly on a map that is wrong."""
    gt = np.zeros((4, 32, 32), dtype=np.float32)
    gt[:, 8:16, 8:16] = 1
    perfect = gt.copy()
    inverted = 1 - gt

    rng = np.random.default_rng(0)
    good = localisation_metrics(perfect, gt, rng)["model"]
    bad = localisation_metrics(inverted, gt, rng)["model"]

    assert good["peak_in_mask"] == 1.0 and bad["peak_in_mask"] == 0.0
    assert good["pixel_auroc"] == 1.0 and bad["pixel_auroc"] == 0.0
    assert good["top1pct_precision"] == 1.0 and bad["top1pct_precision"] == 0.0
    assert good["aupro"] > 0.9 > bad["aupro"], (good["aupro"], bad["aupro"])
    print("self-check ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="bottle")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    demo() if a.self_check else run(a.category, not a.no_crop)
