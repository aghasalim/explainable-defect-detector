"""Milestone 2 - end-to-end anomaly-detection baseline.

Deliberately the simplest thing that is still a real anomaly detector:

    1. frozen ImageNet ResNet18
    2. embed every NORMAL training image to one 512-d vector (global avg pool)
    3. score a test image by its distance to the nearest normal embedding
    4. threshold that score

There is no training loop and no gradient step. That is the point: it uses
only normal data, exactly like the deployed setting where defects are rare,
and it establishes the number every later model has to beat.

Its known weakness is that global average pooling collapses the whole image
to one vector, so a small defect is averaged away and cannot be localised.
That weakness is what motivates the patch-level model in Milestone 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from dataset import MVTecCategory

ROOT = Path(__file__).resolve().parents[2]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def embedder() -> torch.nn.Module:
    """ResNet18 truncated after global average pooling -> (N, 512)."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval()


@torch.no_grad()
def embed(model: torch.nn.Module, ds: MVTecCategory, dev: torch.device, bs: int = 32):
    feats, labels, paths = [], [], []
    for img, label, _mask, path in DataLoader(ds, batch_size=bs, num_workers=0):
        feats.append(model(img.to(dev)).cpu())
        labels.append(label)
        paths += list(path)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), paths


def knn_scores(train: np.ndarray, test: np.ndarray, k: int = 1) -> np.ndarray:
    """Mean distance to the k nearest normal embeddings. Higher = more anomalous."""
    d = np.linalg.norm(test[:, None, :] - train[None, :, :], axis=2)
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC + AP, and precision/recall at the best-F1 threshold.

    Accuracy is reported only next to the majority-class baseline, so it can
    never be quoted on its own as if it meant something.
    """
    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    t = thr[min(i, len(thr) - 1)]
    pred = scores >= t
    return {
        "image_auroc": float(auroc),
        "average_precision": float(ap),
        "best_f1": float(f1[i]),
        "precision_at_best_f1": float(prec[i]),
        "recall_at_best_f1": float(rec[i]),
        "threshold": float(t),
        "accuracy_at_best_f1": float((pred == labels).mean()),
        "majority_class_accuracy": float(max(labels.mean(), 1 - labels.mean())),
        "n_normal": int((labels == 0).sum()),
        "n_anomalous": int((labels == 1).sum()),
    }


def run(category: str, k: int = 1, size: int = 224) -> dict:
    torch.manual_seed(0)
    dev = device()
    model = embedder().to(dev)

    train = MVTecCategory(category, "train", size)
    test = MVTecCategory(category, "test", size)
    assert set(np.unique([l for _, l, _ in train.items])) == {0}, "train must be normal-only"

    tr, _, _ = embed(model, train, dev)
    te, labels, paths = embed(model, test, dev)
    scores = knn_scores(tr, te, k)

    m = evaluate(labels, scores)
    m |= {"category": category, "k": k, "backbone": "resnet18", "n_train_normal": len(tr)}

    out = ROOT / "reports" / f"baseline_{category}.json"
    out.write_text(json.dumps(
        {"metrics": m, "scores": [
            {"path": str(Path(p).relative_to(ROOT)), "label": int(l), "score": float(s)}
            for p, l, s in zip(paths, labels, scores)]}, indent=1))

    w = max(len(x) for x in m)
    for key, v in m.items():
        print(f"{key:{w}} : {v:.4f}" if isinstance(v, float) else f"{key:{w}} : {v}")
    print(f"\nwrote {out}")
    return m


def demo() -> None:
    """Self-check: the metric code must be right before any number is trusted."""
    lab = np.array([0, 0, 1, 1])
    assert evaluate(lab, np.array([0.1, 0.2, 0.8, 0.9]))["image_auroc"] == 1.0
    assert evaluate(lab, np.array([0.9, 0.8, 0.2, 0.1]))["image_auroc"] == 0.0
    # kNN distance: an exact copy of a training point must score 0
    tr = np.array([[0.0, 0.0], [3.0, 4.0]])
    s = knn_scores(tr, np.array([[0.0, 0.0], [0.0, 5.0]]), k=1)
    # [0,5] is sqrt(10) from [3,4] and 5.0 from [0,0] -> nearest is sqrt(10)
    assert s[0] == 0.0 and abs(s[1] - 10**0.5) < 1e-6, s
    print("self-check ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="bottle")
    p.add_argument("-k", type=int, default=1)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    demo() if a.self_check else run(a.category, a.k)
