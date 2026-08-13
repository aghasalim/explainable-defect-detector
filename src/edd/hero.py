"""Build the README montage: one worked example per category.

Shows input / anomaly map / overlay-with-ground-truth side by side, so a reader
can judge the explanations without running anything. Categories are chosen to
span the range honestly - strong localisation and weak localisation both appear.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from patchcore import fit_score

ROOT = Path(__file__).resolve().parents[2]


def main(categories: list[str]) -> None:
    fig, ax = plt.subplots(len(categories), 3, figsize=(8.2, 2.75 * len(categories)))
    for r, cat in enumerate(categories):
        res = fit_score(cat, 0.01, 224, "coreset", True)
        maps = res["maps"].numpy()[:, 0]
        masks = res["masks"].numpy()[:, 0]
        lab, paths, sc = res["labels"], res["paths"], res["img_scores"]

        # median-scoring anomalous image: not the best case, not the worst
        anom = [i for i in range(len(lab)) if lab[i] == 1]
        i = sorted(anom, key=lambda j: sc[j])[len(anom) // 2]

        img = Image.open(paths[i]).convert("RGB").resize((224, 224))
        peak = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
        hit = masks[i][peak] > 0

        ax[r, 0].imshow(img)
        ax[r, 0].set_ylabel(f"{cat}\n{Path(paths[i]).parent.name}", fontsize=8)
        ax[r, 1].imshow(maps[i], cmap="inferno")
        ax[r, 2].imshow(img)
        ax[r, 2].imshow(maps[i], cmap="inferno", alpha=0.5)
        ax[r, 2].contour(masks[i] > 0, levels=[0.5], colors="lime", linewidths=1.2)
        ax[r, 2].plot(peak[1], peak[0], "o", ms=7, mfc="none", mew=1.8,
                      mec="lime" if hit else "red")
        for c in range(3):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
        print(f"{cat:11} median-scoring defect, peak {'inside' if hit else 'OUTSIDE'} mask",
              flush=True)

    for c, t in enumerate(["input", "anomaly map", "overlay — GT green, peak ○"]):
        ax[0, c].set_title(t, fontsize=10)
    fig.suptitle("PatchCore anomaly maps — median-difficulty defect per category", fontsize=11)
    fig.tight_layout()
    out = ROOT / "reports" / "hero.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="*",
                   default=["bottle", "hazelnut", "leather", "pill", "screw"])
    main(p.parse_args().categories)
