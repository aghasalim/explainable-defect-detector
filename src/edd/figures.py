"""Draw the README figures from reports/*.json.

Reads the saved benchmark output only -- no MVTec download, no model, no GPU.
Every number here already appears in reports/benchmark.md.

    python -m edd.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def benches() -> list[dict]:
    """Every per-category benchmark record, ordered by category."""
    return sorted(
        (json.loads(p.read_text()) for p in REPORTS.glob("bench_*.json")),
        key=lambda d: d["category"],
    )


def reproduction(out: Path) -> Path:
    """Measured AUROC against the number the PatchCore paper reports.

    A reimplementation that only reports its own score is unfalsifiable. Plotting
    it against the published value per category shows where this one tracks the
    paper and where it does not.
    """
    rows = benches()
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, mine, theirs, title in (
        (left, "image_auroc", "paper_image_auroc", "image-level AUROC"),
        (right, "pixel_auroc", "paper_pixel_auroc", "pixel-level AUROC"),
    ):
        ax.barh(positions - 0.2, [r[mine] for r in rows], 0.4,
                label="this repo", color="#2166ac", edgecolor="0.3", lw=0.4)
        ax.barh(positions + 0.2, [r.get(theirs) or np.nan for r in rows], 0.4,
                label="PatchCore paper", color="#bdbdbd", edgecolor="0.3", lw=0.4)
        ax.set_xlim(0.5, 1.02)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_yticks(positions)
    left.set_yticklabels(categories, fontsize=8)
    left.invert_yaxis()
    left.legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Reproduction against the published numbers, per MVTec category.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def localisation_control(out: Path) -> Path:
    """Localisation quality against a control that ignores the image.

    A heatmap can look convincing and still be no better than chance at pointing
    at the defect. The control columns are what a score with no spatial
    information achieves on the same masks, which is the bar these have to clear.
    """
    rows = benches()
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, real, control, title in (
        (left, "peak_in_mask", "control_peak_in_mask", "peak inside the defect mask"),
        (right, "top1pct_precision", "control_top1pct_precision",
         "precision of the top 1% of pixels"),
    ):
        ax.barh(positions - 0.2, [r[real] for r in rows], 0.4,
                label="measured", color="#1a9850", edgecolor="0.3", lw=0.4)
        ax.barh(positions + 0.2, [r[control] for r in rows], 0.4,
                label="control (no spatial information)", color="#b2182b",
                edgecolor="0.3", lw=0.4)
        ax.set_xlim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_yticks(positions)
    left.set_yticklabels(categories, fontsize=8)
    left.invert_yaxis()
    left.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def threshold_check(out: Path) -> Path:
    """Does the calibrated threshold deliver the false-positive rate it promised?"""
    rows = json.loads((REPORTS / "threshold_check.json").read_text())
    rows = sorted(rows, key=lambda r: r["recall_on_test"])
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))
    target = rows[0]["target_fpr"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    left.barh(positions, [r["realised_fpr_on_test"] * 100 for r in rows],
              color="#2166ac", edgecolor="0.3", lw=0.4)
    left.axvline(target * 100, color="#b2182b", ls="--", lw=1.4)
    left.text(target * 100, len(rows) - 0.4, f"  target {target:.0%}",
              fontsize=8, color="#b2182b")
    left.set_yticks(positions)
    left.set_yticklabels(categories, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("realised false-positive rate on test (%)")
    left.set_title("the threshold is conservative almost everywhere", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.barh(positions, [r["recall_on_test"] * 100 for r in rows],
               color="#f4a582", edgecolor="0.3", lw=0.4)
    right.set_xlabel("recall on test (%)")
    right.set_title("and this is what that costs in recall", fontsize=10)
    right.set_xlim(0, 105)
    right.spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(rows):
        right.text(row["recall_on_test"] * 100 + 1, index,
                   f"n={row['n_anomalous']}", va="center", fontsize=7, color="0.45")

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def calibration_rules(out: Path) -> Path:
    """A 99th-percentile threshold against a distribution-free tolerance bound.

    The tolerance bound is the honest one: it holds without assuming the
    calibration scores are normal. It is also strictly more conservative, and the
    recall column is the price.
    """
    rows = json.loads((REPORTS / "calibration_compare.json").read_text())
    rows = sorted(rows, key=lambda r: r["cat"])
    categories = [r["cat"] for r in rows]
    positions = np.arange(len(rows))

    figure, ax = plt.subplots(figsize=(11, 5.4))
    ax.barh(positions - 0.2, [r["q99_recall"] * 100 for r in rows], 0.4,
            label="99th-percentile threshold", color="#9ecae1",
            edgecolor="0.3", lw=0.4)
    ax.barh(positions + 0.2, [r["tol_recall"] * 100 for r in rows], 0.4,
            label="tolerance bound (distribution-free)", color="#2166ac",
            edgecolor="0.3", lw=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("recall on test (%)")
    ax.set_xlim(0, 105)
    ax.set_title(
        "Both hold the false-positive target. The distribution-free bound gives "
        "up recall for it.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def guarantee(out: Path) -> Path:
    """Whether each category has enough calibration images for the stated guarantee.

    The tolerance bound needs 299 normal calibration images for a 95%-confidence
    1% bound. MVTec's training splits are smaller than that for most categories,
    so the guarantee is claimed nowhere and the shortfall is shown instead.
    """
    models = sorted(
        (json.loads(p.read_text()) for p in (ROOT / "models").glob("*.json")),
        key=lambda d: d["n_calib"],
    )
    categories = [m["category"] for m in models]
    positions = np.arange(len(models))
    required = models[0]["n_required_for_guarantee"]

    figure, ax = plt.subplots(figsize=(10.5, 5.2))
    colours = ["#1a9850" if m.get("guarantee_met") else "#b2182b" for m in models]
    ax.barh(positions, [m["n_calib"] for m in models], color=colours,
            edgecolor="0.3", lw=0.4)
    ax.axvline(required, color="0.25", ls="--", lw=1.5)
    ax.text(required, len(models) - 0.4,
            f"  {required} needed for the 95% / 1% bound", fontsize=9, color="0.3")
    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normal calibration images available")
    met = sum(1 for m in models if m.get("guarantee_met"))
    ax.set_title(
        f"The distribution-free guarantee is met in {met} of {len(models)} "
        "categories.\nIt is reported as unmet rather than quietly assumed.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        reproduction(FIGURES / "reproduction.png"),
        localisation_control(FIGURES / "localisation-control.png"),
        threshold_check(FIGURES / "threshold-check.png"),
        calibration_rules(FIGURES / "calibration-rules.png"),
        guarantee(FIGURES / "guarantee.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
