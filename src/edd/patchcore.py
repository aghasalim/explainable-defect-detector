"""Milestone 3 - PatchCore (Roth et al., CVPR 2022), implemented directly.

The baseline collapsed each image to one vector, so small defects were averaged
away and nothing could be localised. PatchCore keeps the spatial grid:

  1. frozen WideResNet50-2, take layer2 + layer3 feature maps
  2. 3x3 average pool each map  -> every patch sees its neighbourhood, which
     buys robustness to small shifts without giving up spatial resolution
  3. upsample layer3 to layer2's grid and concatenate -> 1536-d per patch.
     layer2 alone is too texture-local, layer4 is too ImageNet-class-specific;
     mid-level features transfer to industrial images the backbone never saw
  4. memory bank = every patch of every NORMAL training image
  5. greedy k-center coreset -> keep ~1%, so inference stays fast and the bank
     fits in memory, while preserving coverage of the feature space
  6. score each test patch by distance to its nearest normal patch
     -> a 28x28 anomaly map, upsampled and blurred to image resolution
     -> image score = max over patches

Still no training loop and no gradient step: the backbone stays frozen. The
only "fitting" is memorising normal patches.

Deviation from the paper, stated plainly: the image score here is the plain max
over patch distances. The paper additionally reweights that max by how isolated
the matched bank point is. The reweighting is a refinement worth roughly a
fraction of a point; it is omitted for clarity and noted in the README.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from baseline import evaluate
from dataset import MVTecCategory
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

ROOT = Path(__file__).resolve().parents[2]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PatchFeatures(torch.nn.Module):
    """Frozen WideResNet50-2 -> locally-aggregated layer2+layer3 patch embeddings."""

    def __init__(self) -> None:
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, P, D) with P = (H/8)^2 patches."""
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        # neighbourhood aggregation, keeping the grid size
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)               # (B, 1536, h, w)
        b, d, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b, h * w, d), (h, w)


@torch.no_grad()
def extract(model: PatchFeatures, ds: MVTecCategory, dev: torch.device, bs: int = 8):
    feats, labels, masks, paths, grid = [], [], [], [], None
    for img, label, mask, path in DataLoader(ds, batch_size=bs, num_workers=0):
        f, grid = model(img.to(dev))
        feats.append(f.cpu())
        labels.append(label)
        masks.append(mask)
        paths += list(path)
    return torch.cat(feats), torch.cat(labels).numpy(), torch.cat(masks), paths, grid


def coreset(x: torch.Tensor, frac: float, dev: torch.device, seed: int = 0) -> torch.Tensor:
    """Greedy k-center subsampling. Returns indices of the kept rows.

    Selects the point furthest from everything already chosen, repeatedly, so
    the retained set covers the feature space rather than clustering in its
    dense regions - random sampling would over-represent whatever is most
    common in the training images.

    Distances are computed in a Johnson-Lindenstrauss random projection (the
    paper's trick): a 128-d projection preserves pairwise distances well enough
    to pick the same kind of spread-out points, ~12x cheaper than full 1536-d.
    """
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    k = max(1, int(n * frac))
    proj = torch.randn(x.shape[1], 128, generator=g) / 128**0.5
    y = (x @ proj).to(dev)

    idx = [int(torch.randint(n, (1,), generator=g))]
    dist = torch.cdist(y, y[idx[0]: idx[0] + 1]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(dist))
        idx.append(i)
        dist = torch.minimum(dist, torch.cdist(y, y[i: i + 1]).squeeze(1))
    return torch.tensor(idx)


@torch.no_grad()
def score(bank: torch.Tensor, test: torch.Tensor, dev: torch.device, chunk: int = 16):
    """Per-patch distance to the nearest normal patch. -> (N, P)"""
    bank = bank.to(dev)
    out = []
    for i in range(0, test.shape[0], chunk):
        t = test[i: i + chunk].to(dev)                       # (c, P, D)
        d = torch.cdist(t, bank.unsqueeze(0).expand(t.shape[0], -1, -1))
        out.append(d.min(dim=2).values.cpu())
    return torch.cat(out)


def to_maps(patch_scores: torch.Tensor, grid, size: int, sigma: float = 4.0) -> torch.Tensor:
    """(N,P) patch scores -> (N,1,size,size) smooth anomaly maps."""
    n = patch_scores.shape[0]
    m = patch_scores.reshape(n, 1, *grid)
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    # gaussian blur, as in the paper - raw upsampled patches are blocky
    r = int(4 * sigma) | 1
    c = torch.arange(r) - r // 2
    k = torch.exp(-(c**2) / (2 * sigma**2))
    k = (k / k.sum()).to(m.dtype)
    # reflect-pad, not zero-pad: zero padding pulls the blurred map down near
    # the border, systematically under-scoring defects at the image edge
    m = F.pad(m, (r // 2,) * 2 + (0, 0), mode="reflect")
    m = F.conv2d(m, k.view(1, 1, 1, -1))
    m = F.pad(m, (0, 0) + (r // 2,) * 2, mode="reflect")
    return F.conv2d(m, k.view(1, 1, -1, 1))


def fit_score(category: str, frac: float = 0.01, size: int = 224, sampling: str = "coreset",
              crop: bool = False) -> dict:
    """Build the memory bank from normal data and score the test split.

    Returns everything downstream code needs (maps, masks, bank, scores) so
    the explainability pass never has to reload a stale .npy and risk pairing
    maps with the wrong masks.
    """
    torch.manual_seed(0)
    dev = device()
    model = PatchFeatures().to(dev)
    t0 = time.time()

    train = MVTecCategory(category, "train", size, crop)
    test = MVTecCategory(category, "test", size, crop)
    tr, _, _, _, grid = extract(model, train, dev)
    te, labels, masks, paths, _ = extract(model, test, dev)

    flat = tr.reshape(-1, tr.shape[-1])
    if sampling == "coreset":
        keep = coreset(flat, frac, dev)
    else:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(flat.shape[0], generator=g)[: max(1, int(flat.shape[0] * frac))]
    bank = flat[keep]

    ps = score(bank, te, dev)
    return {
        "maps": to_maps(ps, grid, size), "masks": masks, "labels": labels, "paths": paths,
        "img_scores": ps.max(dim=1).values.numpy(), "bank": bank, "grid": grid,
        "n_patches": int(flat.shape[0]), "patch_dim": int(flat.shape[1]),
        "seconds": round(time.time() - t0, 1),
    }


def run(category: str, frac: float = 0.01, size: int = 224, sampling: str = "coreset",
        crop: bool = False) -> dict:
    r = fit_score(category, frac, size, sampling, crop)
    maps, masks, labels = r["maps"], r["masks"], r["labels"]
    paths, img_scores, bank, grid = r["paths"], r["img_scores"], r["bank"], r["grid"]

    m = evaluate(labels, img_scores)
    m["pixel_auroc"] = float(
        roc_auc_score(masks.numpy().ravel().astype(int), maps.numpy().ravel())
    )
    m |= {
        "category": category, "backbone": "wide_resnet50_2", "layers": "layer2+layer3",
        "patch_dim": r["patch_dim"], "grid": list(grid), "sampling": sampling,
        "coreset_frac": frac, "bank_size": int(bank.shape[0]),
        "input_size": size, "center_crop": crop,
        "patches_before_sampling": r["n_patches"], "seconds": r["seconds"],
    }

    tag = "patchcore" if sampling == "coreset" else f"patchcore-{sampling}"
    if crop or size != 224:
        tag += f"-{size}{'crop' if crop else ''}"
    if frac != 0.01:                       # keep sweep points from overwriting each other
        tag += f"-{frac:g}"
    (ROOT / "reports" / f"{tag}_{category}.json").write_text(json.dumps(
        {"metrics": m, "scores": [
            {"path": str(Path(p).relative_to(ROOT)), "label": int(l), "score": float(s)}
            for p, l, s in zip(paths, labels, img_scores, strict=True)]}, indent=1))
    np.save(ROOT / "reports" / f"{tag}_{category}_maps.npy", maps.numpy().astype(np.float16))

    w = max(len(x) for x in m)
    for key, v in m.items():
        print(f"{key:{w}} : {v:.4f}" if isinstance(v, float) else f"{key:{w}} : {v}")
    return m


def demo() -> None:
    """Self-check on the two pieces that are easy to get silently wrong."""
    # coreset must spread out: given two tight clusters, it must pick from both
    a = torch.randn(200, 8) * 0.01
    b = torch.randn(200, 8) * 0.01 + 50.0
    x = torch.cat([a, b])
    idx = coreset(x, 0.05, torch.device("cpu"))
    picked_a = (idx < 200).sum().item()
    assert 0 < picked_a < len(idx), f"coreset collapsed into one cluster: {picked_a}/{len(idx)}"

    # a patch identical to a bank patch must score 0; a far one must not
    bank = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    s = score(bank, torch.tensor([[[0.0, 0.0], [9.0, 9.0]]]), torch.device("cpu"))
    assert s[0, 0].item() == 0.0 and s[0, 1].item() > 10, s

    # blurring must preserve the location of a single hot patch
    p = torch.zeros(1, 49)
    p[0, 24] = 1.0
    mp = to_maps(p, (7, 7), 224)[0, 0]
    yx = divmod(int(torch.argmax(mp)), 224)
    assert 90 < yx[0] < 134 and 90 < yx[1] < 134, yx
    print("self-check ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="bottle")
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--sampling", choices=["coreset", "random"], default="coreset")
    p.add_argument("--crop", action="store_true", help="Resize(256)+CenterCrop(224), as in the paper")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    demo() if a.self_check else run(a.category, a.frac, a.size, a.sampling, a.crop)
