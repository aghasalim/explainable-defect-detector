"""Milestone 4b - the supervised comparison, run honestly, plus Grad-CAM.

The brief asks for Grad-CAM. Grad-CAM needs a classifier to backpropagate
through, and PatchCore has none - so the comparison is built properly rather
than bolted on:

  * MVTec's `train/` holds no defects, so a supervised classifier can only be
    trained by taking defects out of `test/`. That is done ONCE, stratified by
    defect type, with a fixed seed, and the split is written to disk.
  * The classifier trains on half the test defects (plus the designated normal
    train split). It is evaluated on the other half, which it never sees.
  * PatchCore is then re-scored on EXACTLY the same held-out half, so the two
    methods are compared on identical images. Comparing a supervised model's
    held-out score against PatchCore's full-test score would flatter one of
    them for free.

The interesting question is not which gets the higher AUROC on a few dozen
images - that is noisy. It is whether a model that was shown defects explains
itself better than one that never saw a single defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms

import baseline
from dataset import IMAGENET_MEAN, IMAGENET_STD, MVTecCategory, build_transform
from explain import localisation_metrics
from patchcore import fit_score

ROOT = Path(__file__).resolve().parents[2]
metrics_of = baseline.evaluate


def make_split(category: str, seed: int = 0) -> dict:
    """Stratified 50/50 split of the TEST set, by defect type. Written to disk.

    Stratifying matters: an unstratified split can hand every `contamination`
    example to training and none to evaluation, so the held-out score would
    measure a defect type the model never had to generalise to.
    """
    ds = MVTecCategory(category, "test", 224)
    by_type: dict[str, list[str]] = {}
    for p, _l, d in ds.items:
        by_type.setdefault(d, []).append(str(p.relative_to(ROOT)))

    rng = np.random.default_rng(seed)
    fit, held = [], []
    for d, paths in sorted(by_type.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        half = len(paths) // 2
        fit += paths[:half]
        held += paths[half:]
    split = {"category": category, "seed": seed, "fit": sorted(fit), "held_out": sorted(held)}
    (ROOT / "reports" / f"split_{category}.json").write_text(json.dumps(split, indent=1))
    return split


class Images(Dataset):
    def __init__(self, paths: list[str], train: bool) -> None:
        self.paths = paths
        self.tf = (
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]) if train else build_transform(224, crop=True)
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = ROOT / self.paths[i]
        label = int(p.parent.name != "good")
        return self.tf(Image.open(p).convert("RGB")), label, str(p)


def train_classifier(category: str, split: dict, epochs: int = 12, dev=None) -> torch.nn.Module:
    dev = dev or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    normals = [str(p.relative_to(ROOT)) for p, _, _ in MVTecCategory(category, "train", 224).items]
    paths = split["fit"] + normals
    labels = np.array([int(Path(p).parent.name != "good") for p in paths])

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(512, 2)
    m = m.to(dev).train()

    # heavy imbalance: normals outnumber defects several-fold once train/good
    # is included, so weight the loss instead of throwing usable data away
    w = torch.tensor([len(labels) / (2 * (labels == c).sum()) for c in (0, 1)],
                     dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    dl = DataLoader(Images(paths, True), batch_size=16, shuffle=True)

    for ep in range(epochs):
        tot = n = 0
        for x, y, _ in dl:
            opt.zero_grad()
            loss = F.cross_entropy(m(x.to(dev)), y.to(dev), weight=w)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / n:.4f}", flush=True)
    return m.eval()


def grad_cam(model: torch.nn.Module, x: torch.Tensor, dev) -> np.ndarray:
    """Grad-CAM on layer4 for the 'defective' logit. -> (B,224,224) in [0,1].

    Gradients are required here, so this deliberately runs outside no_grad -
    a common silent failure is wrapping the whole eval in torch.no_grad() and
    getting an all-zero CAM.
    """
    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    h1 = model.layer4.register_forward_hook(lambda _m, _i, o: acts.append(o))
    h2 = model.layer4.register_full_backward_hook(lambda _m, _gi, go: grads.append(go[0]))
    try:
        x = x.to(dev).requires_grad_(True)
        out = model(x)
        model.zero_grad()
        out[:, 1].sum().backward()
        a, g = acts[0], grads[0]
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[:, 0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    mx = cam.reshape(len(cam), -1).max(1).reshape(-1, 1, 1)
    return cam / np.maximum(mx, 1e-8)


def run(category: str, epochs: int = 12) -> dict:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = make_split(category)
    held = split["held_out"]
    print(f"{category}: fit on {len(split['fit'])} test images, hold out {len(held)}")

    model = train_classifier(category, split, epochs, dev)

    # --- classifier on held-out ---------------------------------------------
    scores, labels, cams, masks, paths = [], [], [], [], []
    for x, y, p in DataLoader(Images(held, False), batch_size=8):
        with torch.no_grad():
            scores.append(F.softmax(model(x.to(dev)), dim=1)[:, 1].cpu().numpy())
        cams.append(grad_cam(model, x, dev))
        labels.append(y.numpy())
        paths += list(p)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    cams = np.concatenate(cams)

    # ground-truth masks under the SAME transform the classifier saw
    ref = MVTecCategory(category, "test", 224, crop=True)
    idx = {str(p): i for i, (p, _, _) in enumerate(ref.items)}
    for p in paths:
        masks.append(ref[idx[p]][2].numpy()[0])
    masks = np.stack(masks)

    cls = metrics_of(labels, scores)
    a = labels == 1
    cls_loc = localisation_metrics(cams[a], masks[a], np.random.default_rng(0))["model"]

    # --- PatchCore on the SAME held-out images ------------------------------
    pc = fit_score(category, 0.01, 224, "coreset", True)
    pc_idx = [pc["paths"].index(p) for p in paths]
    pc_scores = pc["img_scores"][pc_idx]
    pc_maps = pc["maps"].numpy()[:, 0][pc_idx]
    pcm = metrics_of(labels, pc_scores)
    pc_loc = localisation_metrics(pc_maps[a], masks[a], np.random.default_rng(0))["model"]

    res = {
        "category": category, "n_held_out": len(held), "n_anomalous_held_out": int(a.sum()),
        "n_fit": len(split["fit"]), "epochs": epochs,
        "classifier": {"image_auroc": cls["image_auroc"],
                       "average_precision": cls["average_precision"],
                       **{f"gradcam_{k}": v for k, v in cls_loc.items()}},
        "patchcore": {"image_auroc": pcm["image_auroc"],
                      "average_precision": pcm["average_precision"],
                      **{f"map_{k}": v for k, v in pc_loc.items()}},
    }
    (ROOT / "reports" / f"compare_{category}.json").write_text(json.dumps(res, indent=1))

    print(f"\n=== {category}: held-out {len(held)} images ({a.sum()} anomalous) ===")
    print(f"{'':22} {'classifier':>12} {'PatchCore':>12}")
    print(f"{'  (saw defects?)':22} {'yes':>12} {'no':>12}")
    print(f"{'image AUROC':22} {cls['image_auroc']:12.4f} {pcm['image_auroc']:12.4f}")
    for k in ("pixel_auroc", "aupro", "peak_in_mask", "top1pct_precision"):
        print(f"{'loc ' + k:22} {cls_loc[k]:12.4f} {pc_loc[k]:12.4f}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="bottle")
    p.add_argument("--epochs", type=int, default=12)
    a = p.parse_args()
    run(a.category, a.epochs)
