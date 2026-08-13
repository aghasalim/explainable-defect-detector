"""Milestone 5a - export a deployable artefact for one category.

Writes models/<category>.pt: the coreset memory bank, the decision threshold,
and the preprocessing config. There are no trained weights to ship, because
nothing was trained - the backbone is frozen ImageNet and the "model" is a set
of remembered normal patches.

THRESHOLD CALIBRATION - the part that is easy to fake
AUROC needs no threshold. Shipping does. The tempting move is to reuse the
threshold that maximised F1 on the test set, but that threshold was chosen with
defect labels the deployed system will never have, so the demo would be
calibrated on its own exam.

Instead the threshold comes from NORMAL data only, in two steps:

  1. k-fold cross-calibration gives every training image an out-of-bank score,
     so the tail is estimated from 209-391 scores instead of a single 10%
     holdout of 21-39. (See calibration_scores.)
  2. the threshold is a one-sided nonparametric TOLERANCE BOUND rather than an
     empirical quantile, because "99% of normal parts score below this, with
     95% confidence" is the deployable claim; an empirical quantile is a point
     estimate that lands too low about half the time. (See tolerance_rank.)

Only normal data is used - exactly what a factory has on day one. Whether that
prediction survives contact with real defects is a separate question, which
verify_threshold.py answers against the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from dataset import MVTecCategory
from patchcore import PatchFeatures, coreset, device, extract, score, to_maps

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def calibration_scores(feats: torch.Tensor, frac: float, dev: torch.device,
                       k_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-bank score for EVERY training image, via k-fold cross-calibration.

    The previous version held out 10% of the training images once and took the
    99th percentile of those 21-39 scores. That fails for a reason worth
    spelling out: the empirical 99th percentile of n samples is essentially the
    sample maximum, and the maximum of n draws estimates about the n/(n+1)
    quantile - roughly the 96th percentile at n=28, not the 99th. So the tail
    was underestimated and the threshold came out too low. On `carpet`, 12 of
    28 test normals scored above the entire calibration set's maximum, and the
    demo flagged 43% of good parts.

    Rotating the held-out fold gives one out-of-bank score per training image
    instead of a tenth of them - 209 to 391 scores rather than 21 to 39 - which
    is enough for the 99th percentile to mean something.

    Each fold's bank is built from (k-1)/k of the images, so it is slightly
    sparser than the bank actually shipped, which nudges calibration scores up
    and the threshold with them. That bias is toward fewer false alarms, which
    is the safe direction to be wrong in.
    """
    n = feats.shape[0]
    k = max(2, min(k_folds, n))
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)

    out = np.empty(n, dtype=np.float32)
    for i, fold in enumerate(folds):
        rest = np.concatenate([f for j, f in enumerate(folds) if j != i])
        flat = feats[rest].reshape(-1, feats.shape[-1])
        bank = flat[coreset(flat, frac, dev)]
        out[fold] = score(bank, feats[fold], dev).max(dim=1).values.numpy()
    return out


def tolerance_rank(n: int, p: float = 0.99, gamma: float = 0.95) -> int | None:
    """Rank of the order statistic that covers the p-quantile with confidence gamma.

    The empirical p-quantile is a point estimate: it lands above the true
    quantile about half the time, so aiming at a 1% false-alarm rate that way
    misses high about half the time. A one-sided nonparametric tolerance bound
    asks the question a deployment actually cares about - "at least p of normal
    parts score below this, and I am gamma confident of it".

    For the m-th smallest of n samples, F(X_(m)) ~ Beta(m, n-m+1), so the
    guarantee holds when P(Beta(m, n-m+1) > p) >= gamma.

    Returns None when even the sample maximum cannot deliver the guarantee,
    which for p=0.99, gamma=0.95 means n < 299 (since 1 - 0.99^n >= 0.95).
    Most MVTec categories have 200-280 training images, so they fall short and
    the caller has to fall back to the maximum and say so.
    """
    for m in range(1, n + 1):
        if stats.beta.sf(p, m, n - m + 1) >= gamma:
            return m
    return None


def choose_threshold(cal: np.ndarray, target_fpr: float = 0.01,
                     method: str = "tolerance", confidence: float = 0.95) -> tuple[float, dict]:
    """Turn calibration scores into a decision threshold."""
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    n = len(cal)
    p = 1.0 - target_fpr
    if method == "quantile":
        return float(np.quantile(cal, p)), {"threshold_method": "empirical_quantile"}

    m = tolerance_rank(n, p, confidence)
    thr = float(cal[m - 1]) if m else float(cal[-1])
    return thr, {
        "threshold_method": "tolerance_bound",
        "tolerance_rank": m,
        "tolerance_confidence": confidence,
        # n needed for the guarantee to be attainable at all
        "n_required_for_guarantee": int(np.ceil(np.log(1 - confidence) / np.log(p))),
        "guarantee_met": m is not None,
    }


def build(category: str, frac: float = 0.01, size: int = 224, crop: bool = True,
          k_folds: int = 5, target_fpr: float = 0.01, seed: int = 0,
          method: str = "tolerance") -> dict:
    torch.manual_seed(seed)
    dev = device()
    model = PatchFeatures().to(dev)

    train = MVTecCategory(category, "train", size, crop)
    feats, _, _, _, grid = extract(model, train, dev)
    n = feats.shape[0]

    cal = calibration_scores(feats, frac, dev, k_folds, seed)
    threshold, thr_meta = choose_threshold(cal, target_fpr, method)

    # The shipped bank uses ALL the normal training images - more normal data
    # can only make the bank a better description of "normal".
    flat = feats.reshape(-1, feats.shape[-1])
    bank = flat[coreset(flat, frac, dev)]
    bank_idx = range(n)
    n_cal = n

    MODELS.mkdir(exist_ok=True)
    # float16 halves the artefact (151 MB -> 76 MB across 15 categories) and is
    # measurably lossless here: max score change 1.8e-4 against thresholds near
    # 2.0, and zero verdict flips over 410 test images. load() casts back up.
    art = {
        "category": category, "bank": bank.cpu().half(), "threshold": threshold,
        "target_fpr": target_fpr, "size": size, "crop": crop, "grid": tuple(grid),
        "bank_size": int(bank.shape[0]), "bank_dim": int(bank.shape[1]),
        "coreset_frac": frac, "n_train_total": int(n), "n_bank_images": len(bank_idx),
        "k_folds": k_folds, **thr_meta,
        "n_calib": int(n_cal), "calib_scores": cal.astype(np.float32),
    }
    torch.save(art, MODELS / f"{category}.pt")

    meta = {k: v for k, v in art.items() if k not in ("bank", "calib_scores")}
    meta |= {
        "calib_score_min": float(cal.min()), "calib_score_max": float(cal.max()),
        "artefact_mb": round((MODELS / f"{category}.pt").stat().st_size / 1e6, 2),
    }
    (MODELS / f"{category}.json").write_text(json.dumps(meta, indent=1, default=str))
    for k, v in meta.items():
        print(f"{k:20} : {v:.4f}" if isinstance(v, float) else f"{k:20} : {v}")
    return meta


def load(category: str, dev: torch.device | None = None) -> dict:
    """Load an exported artefact, with the bank moved onto the target device."""
    dev = dev or torch.device("cpu")
    art = torch.load(MODELS / f"{category}.pt", weights_only=False, map_location="cpu")
    art["bank"] = art["bank"].float().to(dev)
    return art


@torch.no_grad()
def predict(art: dict, x: torch.Tensor, model: PatchFeatures, dev: torch.device):
    """Score one preprocessed image. -> (image score, flagged, HxW anomaly map)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    feats, grid = model(x.to(dev))
    ps = score(art["bank"], feats.cpu(), dev)
    amap = to_maps(ps, grid, art["size"])[0, 0].numpy()
    s = float(ps.max())
    return s, bool(s >= art["threshold"]), amap


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # several categories at once, or --all for the whole benchmark; the README
    # documented the multi-category form before the script accepted it
    p.add_argument("categories", nargs="*", default=["bottle"])
    p.add_argument("--all", action="store_true", help="every downloaded category")
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--method", choices=["tolerance", "quantile"], default="tolerance")
    a = p.parse_args()

    cats = a.categories or ["bottle"]
    if a.all:
        cats = sorted(d.name for d in (ROOT / "data" / "mvtec").iterdir() if d.is_dir())

    for i, c in enumerate(cats, 1):
        print(f"=== [{i}/{len(cats)}] {c} ===")
        build(c, frac=a.frac, target_fpr=a.target_fpr, k_folds=a.k_folds, method=a.method)
