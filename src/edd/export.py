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

Instead, hold out 10% of the NORMAL training images, keep them out of the
memory bank, score them, and place the threshold at the (1 - target_fpr)
quantile of those scores. Only normal data is used - exactly what a factory has
on day one. Whether that prediction survives contact with real defects is a
separate question, which verify_threshold.py answers against the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dataset import MVTecCategory
from patchcore import PatchFeatures, coreset, device, extract, score, to_maps

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def build(category: str, frac: float = 0.01, size: int = 224, crop: bool = True,
          calib_frac: float = 0.10, target_fpr: float = 0.01, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    dev = device()
    model = PatchFeatures().to(dev)

    train = MVTecCategory(category, "train", size, crop)
    feats, _, _, _, grid = extract(model, train, dev)

    # Hold out calibration images BEFORE building the bank. If they were in it,
    # every calibration patch would match itself at distance ~0 and the
    # threshold would collapse to zero - the classic self-match leak.
    n = feats.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = max(1, int(round(n * calib_frac)))
    cal_idx, bank_idx = perm[:n_cal], perm[n_cal:]

    flat = feats[bank_idx].reshape(-1, feats.shape[-1])
    bank = flat[coreset(flat, frac, dev)]

    cal = score(bank, feats[cal_idx], dev).max(dim=1).values.numpy()
    threshold = float(np.quantile(cal, 1.0 - target_fpr))

    MODELS.mkdir(exist_ok=True)
    # float16 halves the artefact (151 MB -> 76 MB across 15 categories) and is
    # measurably lossless here: max score change 1.8e-4 against thresholds near
    # 2.0, and zero verdict flips over 410 test images. load() casts back up.
    art = {
        "category": category, "bank": bank.cpu().half(), "threshold": threshold,
        "target_fpr": target_fpr, "size": size, "crop": crop, "grid": tuple(grid),
        "bank_size": int(bank.shape[0]), "bank_dim": int(bank.shape[1]),
        "coreset_frac": frac, "n_train_total": int(n), "n_bank_images": len(bank_idx),
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
    a = p.parse_args()

    cats = a.categories or ["bottle"]
    if a.all:
        cats = sorted(d.name for d in (ROOT / "data" / "mvtec").iterdir() if d.is_dir())

    for i, c in enumerate(cats, 1):
        print(f"=== [{i}/{len(cats)}] {c} ===")
        build(c, frac=a.frac, target_fpr=a.target_fpr)
