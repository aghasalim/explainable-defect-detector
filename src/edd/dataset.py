"""Dataset plumbing + the preprocessing decision it forces.

Torchvision's stock ImageNet transform is Resize(256) -> CenterCrop(224),
which silently discards the outer ~12% of every image. On a defect dataset
that is not a cosmetic choice: a defect that lives near the edge is deleted
from the input while its label still says "defective", so the model is asked
to predict an anomaly from an image that no longer contains one.

Run this module directly to measure how much defect area the crop would
destroy for a category, and pick the transform on evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size: int = 224, crop: bool = False) -> transforms.Compose:
    """crop=True reproduces the stock ImageNet pipeline; default is a plain resize."""
    steps: list = [transforms.Resize(int(size * 256 / 224) if crop else (size, size))]
    if crop:
        steps.append(transforms.CenterCrop(size))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class MVTecCategory(Dataset):
    """One MVTec category/split. Returns (image, label, mask, path).

    label: 0 normal, 1 anomalous.
    mask:  float tensor (1,size,size); all-zero for normal images, which is
           what the ground truth genuinely says, not a placeholder.
    """

    def __init__(self, category: str, split: str, size: int = 224, crop: bool = False,
                 root: Path | None = None) -> None:
        self.dir = (root or ROOT / "data" / "mvtec") / category
        self.size, self.crop = size, crop
        self.tf = build_transform(size, crop)
        self.items: list[tuple[Path, int, str]] = []
        for defect_dir in sorted((self.dir / split).iterdir()):
            if not defect_dir.is_dir():
                continue
            for p in sorted(defect_dir.glob("*.png")):
                self.items.append((p, int(defect_dir.name != "good"), defect_dir.name))

    def __len__(self) -> int:
        return len(self.items)

    def _mask_tf(self, im: Image.Image) -> torch.Tensor:
        # nearest-neighbour so the mask stays binary after resizing
        r = transforms.Resize(
            int(self.size * 256 / 224) if self.crop else (self.size, self.size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        im = r(im)
        if self.crop:
            im = transforms.CenterCrop(self.size)(im)
        return (torch.from_numpy(np.asarray(im, dtype=np.float32)) > 0).float()[None]

    def __getitem__(self, i: int):
        path, label, defect = self.items[i]
        img = self.tf(Image.open(path).convert("RGB"))
        mp = self.dir / "ground_truth" / defect / f"{path.stem}_mask.png"
        mask = (
            self._mask_tf(Image.open(mp).convert("L"))
            if mp.exists()
            else torch.zeros(1, self.size, self.size)
        )
        return img, label, mask, str(path)


def measure_crop_loss(category: str, size: int = 224) -> None:
    """How much ground-truth defect area does CenterCrop delete?

    Measured in ORIGINAL image coordinates. Comparing defect pixel counts
    before/after the torchvision pipeline is invalid, because Resize(256) +
    CenterCrop(224) magnifies the object: the crop can raise the pixel count
    while still cutting spatial extent. The scale-free question is what
    fraction of the mask falls inside the crop box.
    """
    ds = MVTecCategory(category, "test", size)
    frac = size / int(size * 256 / 224)  # 224/256 = 0.875 of each side, centred
    kept, lost_all, touched = [], [], 0
    for path, label, defect in ds.items:
        if not label:
            continue
        mp = ds.dir / "ground_truth" / defect / f"{path.stem}_mask.png"
        m = np.asarray(Image.open(mp).convert("L")) > 0
        h, w = m.shape
        ch, cw = int(h * frac), int(w * frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        full = m.sum()
        inside = m[y0 : y0 + ch, x0 : x0 + cw].sum()
        r = inside / max(full, 1)
        kept.append(r)
        if r < 1.0:
            touched += 1
        if inside == 0 and full > 0:
            lost_all.append(f"{defect}/{path.name}")

    k = np.array(kept)
    print(f"category            : {category}  ({len(k)} anomalous test images)")
    print(f"crop box            : central {frac:.1%} of each side")
    print(f"defect area retained: {k.mean():.2%} mean, {k.min():.2%} worst case")
    print(f"images clipped at all: {touched}/{len(k)}")
    print(f"defects erased      : {len(lost_all)} image(s) {lost_all[:5]}")
    print(
        "\nverdict: "
        + (
            "CenterCrop is safe here, but resize-only costs nothing and generalises "
            "to categories where it is not."
            if not lost_all and k.min() > 0.9
            else "CenterCrop destroys labelled defect area -> use resize-only."
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", default="bottle")
    measure_crop_loss(p.parse_args().category)
