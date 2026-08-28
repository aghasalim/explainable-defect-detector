"""Draw the README figures from reports/*.json.

Reads the saved benchmark output only, so there is no MVTec download, no model
and no GPU here. Every number a figure shows already appears in
reports/benchmark.md or reports/results.md, and this module only reads it.

    python src/edd/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from PIL import Image
from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Green for the measured heatmap and red for the no-information control,
# because that is the pairing the README argues about. Grey is always the
# thing being compared against: the paper's number, or the rule I did not
# ship. Everything else takes a colour from the shared palette.
MEASURED, CONTROL = "#1a9850", "#b2182b"
REFERENCE = "#8c8c8c"
MINE = PALETTE[0]

# Guarantee met or not met keeps the same green and red, since the README
# talks about it as a pass or fail rather than as a quantity.
MET, UNMET = MEASURED, CONTROL


def benches() -> list[dict]:
    """Every per-category benchmark record, ordered by category."""
    return sorted(
        (json.loads(p.read_text()) for p in REPORTS.glob("bench_*.json")),
        key=lambda d: d["category"],
    )


def _dumbbell(ax, positions, low, high) -> None:
    """The connecting line for a paired dot plot.

    Bars for two near-identical AUROC values are 95% shared ink and the eye has
    to measure the tips. A line between two dots puts the difference itself on
    the page.
    """
    ax.hlines(positions, low, high, color="#cccccc", lw=1.8, zorder=1)


def _rows(ax, positions, labels) -> None:
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(positions) - 0.6, -0.6)
    ax.grid(axis="y", visible=False)


def reproduction(out: Path) -> Path:
    """Measured AUROC against the number the PatchCore paper reports.

    A reimplementation that only reports its own score is unfalsifiable.
    Plotting it against the published value per category shows where this one
    tracks the paper and where it does not.
    """
    rows = sorted(benches(), key=lambda r: r["image_auroc"] - r["paper_image_auroc"])
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))
    worst = rows[0]
    image_gaps = [r["image_auroc"] - r["paper_image_auroc"] for r in rows[1:]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = (
        (left, "image_auroc", "paper_image_auroc", 0.93,
         f"Only {worst['category']} misses the published AUROC",
         f"every other category is within {max(abs(g) for g in image_gaps):.3f} of the paper, "
         f"{worst['category']} is {abs(worst['image_auroc'] - worst['paper_image_auroc']):.3f} short",
         "image-level AUROC (1.0 = perfect ranking)"),
        (right, "pixel_auroc", "paper_pixel_auroc", 0.90,
         "Pixel AUROC is short in all 15",
         "the score reweighting step of the paper is not implemented here",
         "pixel-level AUROC (1.0 = perfect ranking)"),
    )
    for ax, mine, theirs, floor, title, subtitle, xlabel in panels:
        measured = np.array([r[mine] for r in rows])
        published = np.array([r[theirs] for r in rows])
        _dumbbell(ax, positions, published, measured)
        ax.plot(published, positions, "o", color=REFERENCE, ms=6.5, zorder=2)
        ax.plot(measured, positions, "o", color=MINE, ms=6.5, zorder=3)
        ax.set_xlim(floor, 1.006)
        ax.set_xlabel(xlabel)
        titled(ax, title, subtitle)
    _rows(left, positions, categories)

    handles = [
        Line2D([], [], marker="o", ls="", color=MINE, label="this repo"),
        Line2D([], [], marker="o", ls="", color=REFERENCE,
               label="PatchCore paper (Roth et al., CVPR 2022)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(out)
    plt.close(figure)
    return out


def localisation_control(out: Path) -> Path:
    """Localisation quality against a control that ignores the image.

    A heatmap can look convincing and still be no better than chance at
    pointing at the defect. The control is what a score with no spatial
    information achieves on the same masks, which is the bar these have to
    clear.
    """
    rows = sorted(benches(), key=lambda r: r["peak_in_mask"])
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))
    worst = rows[0]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = (
        (left, "peak_in_mask", "control_peak_in_mask",
         "The peak beats chance in all 15",
         f"worst is {worst['category']} at {worst['peak_in_mask']:.0%}, against a control "
         f"of {worst['control_peak_in_mask']:.0%} on the same masks",
         "defect images where the peak falls inside the mask (%)"),
        (right, "top1pct_precision", "control_top1pct_precision",
         "So does the hottest 1% of pixels",
         "the control is a random heatmap, so it lands at the defect pixel fraction",
         "share of the top 1% of pixels that are defect pixels (%)"),
    )
    for ax, real, control, title, subtitle, xlabel in panels:
        measured = np.array([r[real] for r in rows]) * 100
        chance = np.array([r[control] for r in rows]) * 100
        _dumbbell(ax, positions, chance, measured)
        ax.plot(chance, positions, "o", color=CONTROL, ms=6.5, zorder=3)
        ax.plot(measured, positions, "o", color=MEASURED, ms=6.5, zorder=2)
        ax.set_xlim(-2, 102)
        ax.set_xlabel(xlabel)
        titled(ax, title, subtitle)
    _rows(left, positions, categories)

    handles = [
        Line2D([], [], marker="o", ls="", color=MEASURED, label="measured"),
        Line2D([], [], marker="o", ls="", color=CONTROL,
               label="control with no spatial information"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(out)
    plt.close(figure)
    return out


def threshold_check(out: Path) -> Path:
    """Does the calibrated threshold deliver the false-positive rate it promised?

    Thirteen of the fifteen realised rates are exactly zero, so bars would draw
    thirteen invisible rectangles. Dots put every category on the page.
    """
    rows = json.loads((REPORTS / "threshold_check.json").read_text())
    rows = sorted(rows, key=lambda r: r["recall_on_test"])
    categories = [r["category"] for r in rows]
    positions = np.arange(len(rows))
    target = rows[0]["target_fpr"] * 100
    held = sum(1 for r in rows if r["realised_fpr_on_test"] * 100 <= target)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)

    realised = np.array([r["realised_fpr_on_test"] * 100 for r in rows])
    left.hlines(positions, 0, realised, color="#cccccc", lw=1.8, zorder=1)
    left.plot(realised, positions, "o", color=MINE, ms=6.5, zorder=3)
    over = realised > target
    left.plot(realised[over], positions[over], "o", color=CONTROL, ms=6.5, zorder=4)
    left.axvline(target, color=CONTROL, ls="--", lw=1.3, zorder=2)
    left.text(target + 0.6, len(rows) - 1.2, f"target {target:.0f}%",
              color=CONTROL, va="center")
    left.set_xlim(-0.8, 27)
    left.set_xlabel("false alarms on the test split (% of normal images)")
    titled(left, f"{held} of {len(rows)} categories hold the 1% target",
           "calibrated on training images only, then measured on the held-out test split")

    recall = np.array([r["recall_on_test"] * 100 for r in rows])
    right.hlines(positions, 0, recall, color="#cccccc", lw=1.8, zorder=1)
    right.plot(recall, positions, "o", color=MINE, ms=6.5, zorder=3)
    right.set_xlim(0, 122)
    right.set_xlabel("defects caught at that threshold (% of defective images)")
    titled(right, f"Recall is the price: {recall.min():.0f}% to {recall.max():.0f}%",
           "the label is how many defective test images the percentage is out of")
    for index, row in enumerate(rows):
        right.text(row["recall_on_test"] * 100 + 2.5, index, f"n={row['n_anomalous']}",
                   va="center", fontsize=8.5, color="#777777")
    _rows(left, positions, categories)

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def calibration_rules(out: Path) -> Path:
    """A 99th-percentile threshold against a distribution-free tolerance bound.

    The tolerance bound is the honest one: it holds without assuming the
    calibration scores are normal. It is also strictly more conservative, and
    recall is the price.
    """
    rows = json.loads((REPORTS / "calibration_compare.json").read_text())
    rows = sorted(rows, key=lambda r: r["tol_recall"])
    categories = [r["cat"] for r in rows]
    positions = np.arange(len(rows))
    percentile = np.array([r["q99_recall"] for r in rows]) * 100
    tolerance = np.array([r["tol_recall"] for r in rows]) * 100
    drop = percentile.mean() - tolerance.mean()
    on_target = sum(1 for r in rows if r["tol_fpr"] <= 0.01)

    figure, ax = plt.subplots(figsize=(11, 5.6))
    _dumbbell(ax, positions, tolerance, percentile)
    ax.plot(percentile, positions, "o", color=REFERENCE, ms=6.5, zorder=2)
    ax.plot(tolerance, positions, "o", color=MINE, ms=6.5, zorder=3)
    _rows(ax, positions, categories)
    ax.set_xlim(0, 104)
    ax.set_xlabel("defects caught on the test split (% of defective images)")
    titled(ax, f"The distribution-free bound costs {drop:.0f} points of mean recall",
           f"and it is what puts {on_target} of {len(rows)} categories inside the 1% "
           "false-alarm target; both thresholds come from the same 5-fold calibration scores")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=MINE, label="tolerance bound, shipped"),
        Line2D([], [], marker="o", ls="", color=REFERENCE, label="99th-percentile threshold"),
    ], loc="lower left")
    figure.tight_layout()
    figure.savefig(out)
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
    met = sum(1 for m in models if m.get("guarantee_met"))

    figure, ax = plt.subplots(figsize=(11, 5.4))
    colours = [MET if m.get("guarantee_met") else UNMET for m in models]
    ax.barh(positions, [m["n_calib"] for m in models], 0.62, color=colours, zorder=2)
    ax.axvline(required, color="#333333", ls="--", lw=1.4, zorder=3)
    ax.annotate(f"{required} images needed for the 95% / 1% bound  ", xy=(required, 0.97),
                xycoords=("data", "axes fraction"), color="#333333", va="top", ha="right")
    _rows(ax, positions, categories)
    ax.set_xlim(0, 410)
    ax.set_xlabel("normal calibration images available (count)")
    titled(ax, f"Only {met} of {len(models)} categories have enough images for the guarantee",
           "green means the bound is supported by the sample size, red means it is "
           "computed and used but recorded as unmet")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _shrink(path: Path) -> Path:
    """Rewrite every frame onto one shared palette, which roughly halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)
    return path


def threshold_sweep(out: Path, category: str = "screw", frames: int = 80,
                    hold: int = 14, fps: int = 13) -> Path:
    """Slide the decision threshold across one category's committed test scores.

    The model is fixed. Only the threshold moves, and every frame is arithmetic
    on the 160 scores already in reports/patchcore-224crop_screw.json. Applying
    the shipped threshold to those same scores reproduces the false-alarm rate
    and recall recorded in reports/threshold_check.json, which is why both can
    be drawn on one axis.
    """
    scores = json.loads((REPORTS / f"patchcore-224crop_{category}.json").read_text())["scores"]
    checks = json.loads((REPORTS / "threshold_check.json").read_text())
    shipped = next(r for r in checks if r["category"] == category)["threshold"]
    normal = np.array([s["score"] for s in scores if s["label"] == 0])
    defect = np.array([s["score"] for s in scores if s["label"] == 1])

    lo, hi = min(normal.min(), defect.min()), max(normal.max(), defect.max())
    pad = 0.03 * (hi - lo)
    cuts = np.linspace(hi + pad, lo - pad, frames)
    false_alarm = np.array([(normal >= c).mean() for c in cuts]) * 100
    caught = np.array([(defect >= c).mean() for c in cuts]) * 100

    # Deterministic jitter, so the two rows of dots do not overplot. It moves
    # points off a line they were never on; the horizontal axis is the data.
    rng = np.random.default_rng(0)
    rows = {1.0: (normal, MINE), 0.0: (defect, CONTROL)}
    jitter = {y: rng.uniform(-0.17, 0.17, len(v)) for y, (v, _) in rows.items()}

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    left.set_xlim(lo - 2 * pad, hi + 2 * pad)
    left.set_ylim(-0.55, 1.75)
    left.set_yticks([1.0, 0.0])
    left.set_yticklabels([f"normal\nn={len(normal)}", f"defective\nn={len(defect)}"])
    left.grid(axis="y", visible=False)
    left.set_xlabel("image anomaly score (max patch distance, unitless)")
    titled(left, "One model, and the threshold decides its worth",
           f"{category}, the {len(scores)} committed test scores, nothing re-run")

    art = {}
    for y, (values, colour) in rows.items():
        art[f"pass{y}"] = left.plot(values, jitter[y] + y, "o", color=colour, ms=4.4,
                                    alpha=0.22, ls="", zorder=2)[0]
        art[f"flag{y}"] = left.plot([], [], "o", color=colour, ms=4.4, ls="", zorder=3)[0]
    art["cut"] = left.axvline(cuts[0], color="#333333", lw=1.4, zorder=4)
    left.axvline(shipped, color="#777777", ls="--", lw=1.2, zorder=1)
    left.text(shipped, 0.55, "shipped threshold", color="#777777", fontsize=9,
              ha="center", va="center")
    # opaque box, else the swept rule and the shipped rule both strike through
    # the numbers as they pass under the readout
    art["readout"] = left.text(0.055, 0.965, "", transform=left.transAxes, fontsize=9.5,
                               color="#444444", ha="left", va="top", zorder=6,
                               bbox={"boxstyle": "square,pad=0.45", "facecolor": "white",
                                     "edgecolor": "#dedede", "linewidth": 0.7})

    right.set_xlim(-3, 103)
    right.set_ylim(-3, 103)
    right.set_xlabel("false alarms (% of normal images flagged)")
    right.set_ylabel("defects caught (% of defective images flagged)")
    ship_x = (normal >= shipped).mean() * 100
    ship_y = (defect >= shipped).mean() * 100
    # Both panels have to name the threshold they describe. The title tracks the
    # swept one, the star and its label stay on the shipped one, so the same
    # percentage can never mean two things in one frame.
    def swept_title(i: int) -> str:
        return (f"Swept threshold: {false_alarm[i]:.0f}% false alarms, "
                f"{caught[i]:.0f}% caught")

    titled(right, swept_title(0),
           "the same sweep drawn as an ROC, the star is the threshold this repo ships")
    # ax.title is the centre title and titles are left-aligned here, so take the
    # artist set_title actually returns. The pad is the one style.titled uses to
    # clear the subtitle, and set_title would reset it to the rcParam otherwise.
    art["title"] = right.set_title(swept_title(0), pad=26)
    right.plot([0, 100], [0, 100], ls=":", lw=1.1, color="#bbbbbb", zorder=1)
    right.plot(false_alarm, caught, color=REFERENCE, lw=1.8, zorder=2)
    right.plot([ship_x], [ship_y], "*", ms=13, color=MEASURED, zorder=5)
    right.annotate(f"  shipped threshold: {ship_y:.0f}% caught, {ship_x:.0f}% false alarms",
                   xy=(ship_x, ship_y), xytext=(6, -2), textcoords="offset points",
                   color=MEASURED, fontsize=9, va="top")
    art["trail"] = right.plot([], [], color=MINE, lw=2.4, zorder=3)[0]
    art["head"] = right.plot([], [], "o", color=MINE, ms=7, zorder=4)[0]

    def draw(index: int):
        i = min(index, frames - 1)
        cut = cuts[i]
        for y, (values, _) in rows.items():
            keep = values >= cut
            art[f"flag{y}"].set_data(values[keep], jitter[y][keep] + y)
        art["cut"].set_xdata([cut, cut])
        art["readout"].set_text(
            f"swept threshold {cut:.2f}\n"
            f"flags {int((normal >= cut).sum())} of {len(normal)} normal\n"
            f"catches {int((defect >= cut).sum())} of {len(defect)} defective")
        art["title"].set_text(swept_title(i))
        art["trail"].set_data(false_alarm[:i + 1], caught[:i + 1])
        art["head"].set_data(false_alarm[i:i + 1], caught[i:i + 1])
        return list(art.values())

    figure.tight_layout()
    animation = FuncAnimation(figure, draw, frames=frames + hold,
                              interval=1000 // fps, blit=False)
    # savefig.bbox is "tight" everywhere else in the repo, but an animated title
    # changes width, a tight box then changes the frame size, and PillowWriter
    # reads every frame at the size of the first one. Pin the box to the figure.
    with mpl.rc_context({"savefig.bbox": None}):
        animation.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return _shrink(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        reproduction(FIGURES / "reproduction.png"),
        localisation_control(FIGURES / "localisation-control.png"),
        threshold_check(FIGURES / "threshold-check.png"),
        calibration_rules(FIGURES / "calibration-rules.png"),
        guarantee(FIGURES / "guarantee.png"),
        threshold_sweep(FIGURES / "threshold-sweep.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
