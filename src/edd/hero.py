"""Build the README montage: one worked example per category.

Shows input, anomaly map and overlay for five categories, so a reader can judge
the explanations without running anything. The categories span the range
honestly: strong localisation and weak localisation both appear, and the image
picked for each is the median-scoring defect, not the best one.

    python src/edd/hero.py [category ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from patchcore import fit_score
from PIL import Image
from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[2]

# Same reading as everywhere else in the repo: green is the measured thing
# agreeing with the ground truth, red is it disagreeing.
HIT, MISS = PALETTE[2], PALETTE[1]
ROWS = ["input", "anomaly map", "overlay"]


def _tile(ax) -> None:
    """An image panel: no ticks, no grid, a light box so white images have edges."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)


def main(categories: list[str]) -> Path:
    figure, ax = plt.subplots(3, len(categories), figsize=(2.55 * len(categories), 8.3),
                              squeeze=False)
    for column, category in enumerate(categories):
        result = fit_score(category, 0.01, 224, "coreset", True)
        maps = result["maps"].numpy()[:, 0]
        masks = result["masks"].numpy()[:, 0]
        labels, paths, scores = result["labels"], result["paths"], result["img_scores"]

        # median-scoring anomalous image: not the best case, not the worst
        anomalous = [i for i in range(len(labels)) if labels[i] == 1]
        i = sorted(anomalous, key=lambda j: scores[j])[len(anomalous) // 2]

        image = Image.open(paths[i]).convert("RGB").resize((224, 224))
        peak = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
        hit = masks[i][peak] > 0

        ax[0, column].imshow(image)
        ax[1, column].imshow(maps[i], cmap="inferno")
        ax[2, column].imshow(image)
        ax[2, column].imshow(maps[i], cmap="inferno", alpha=0.5)
        ax[2, column].contour(masks[i] > 0, levels=[0.5], colors="white", linewidths=1.6)
        # clip_on: a peak on the border still gets a whole circle drawn
        ax[2, column].plot(peak[1], peak[0], "o", ms=11, mfc="none", mew=2.4,
                           mec=HIT if hit else MISS, clip_on=False)
        for row in range(3):
            _tile(ax[row, column])
        titled(ax[0, column], category, Path(paths[i]).parent.name.replace("_", " "))
        print(f"{category:11} median-scoring defect, "
              f"peak {'inside' if hit else 'OUTSIDE'} the mask", flush=True)

    for row, label in enumerate(ROWS):
        ax[row, 0].set_ylabel(label)

    figure.text(0.008, 0.988, "Where the model says the defect is",
                fontsize=12.5, fontweight="semibold", ha="left", va="top")
    figure.text(0.008, 0.962,
                "the median-difficulty defect of each category, white outline is the "
                "labelled defect, the circle is the hottest pixel and is red when it "
                "falls outside",
                fontsize=9.3, color="#5a5a5a", ha="left", va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.945))
    out = ROOT / "reports" / "hero.png"
    figure.savefig(out)
    plt.close(figure)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="*",
                   default=["bottle", "hazelnut", "leather", "pill", "screw"])
    main(p.parse_args().categories)
