"""Milestone 1 EDA for one MVTec AD category.

Answers the questions that actually change later modelling decisions, rather
than printing pretty pictures:

  1. class balance per split      -> which metrics are honest to report
  2. is there ANY defect in train -> whether supervised classification is even
                                     possible without stealing from the test set
  3. defect area as % of pixels   -> whether pixel accuracy is meaningless, and
                                     how much downsizing we can afford before
                                     small defects vanish
  4. resolution / channel checks  -> preprocessing assumptions
  5. exact-duplicate detection    -> silent train/test contamination
  6. brightness spread            -> is "anomaly" confounded with exposure

Writes reports/eda_<category>.md and reports/eda_<category>.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from style import PALETTE

# Fixed meanings, so the same idea keeps the same colour across every figure.
NEUTRAL = PALETTE[5]   # good, or nothing wrong
DEFECT = PALETTE[1]    # defective
MASK = PALETTE[2]      # the ground truth outline

ROOT = Path(__file__).resolve().parents[2]


def scan(cat_dir: Path) -> list[dict]:
    """One record per image, with mask stats attached where a mask exists."""
    rows = []
    for split in ("train", "test"):
        for defect_dir in sorted((cat_dir / split).iterdir()):
            for img_path in sorted(defect_dir.glob("*.png")):
                im = Image.open(img_path)
                arr = np.asarray(im.convert("L"), dtype=np.float32)
                rec = {
                    "path": img_path,
                    "split": split,
                    "defect": defect_dir.name,
                    "label": 0 if defect_dir.name == "good" else 1,
                    "w": im.width,
                    "h": im.height,
                    "mode": im.mode,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "sha": hashlib.sha1(img_path.read_bytes()).hexdigest(),
                    "defect_frac": None,
                }
                # canonical MVTec mask name: 000.png -> 000_mask.png
                mask = cat_dir / "ground_truth" / defect_dir.name / f"{img_path.stem}_mask.png"
                if mask.exists():
                    m = np.asarray(Image.open(mask).convert("L")) > 0
                    rec["defect_frac"] = float(m.mean())
                rows.append(rec)
    return rows


def report(cat: str, rows: list[dict]) -> str:
    out: list[str] = [f"# EDA - MVTec AD `{cat}`\n", f"{len(rows)} images.\n"]

    # --- 1/2. class balance -------------------------------------------------
    out.append("## Class balance\n")
    out.append("| split | class | n |\n|---|---|---|")
    counts: Counter = Counter((r["split"], r["defect"]) for r in rows)
    for (split, defect), n in sorted(counts.items()):
        out.append(f"| {split} | {defect} | {n} |")
    tr_bad = sum(1 for r in rows if r["split"] == "train" and r["label"] == 1)
    te_ok = sum(1 for r in rows if r["split"] == "test" and r["label"] == 0)
    te_bad = sum(1 for r in rows if r["split"] == "test" and r["label"] == 1)
    out.append(
        f"\n**Defective images in `train`: {tr_bad}.** Supervised binary classification is "
        f"therefore impossible without moving defects out of `test`, which contaminates the "
        f"only clean evaluation set the benchmark has. This is the single most important "
        f"fact in this dataset.\n"
    )
    maj = max(te_ok, te_bad) / (te_ok + te_bad)
    out.append(
        f"**Test set is {te_ok} good / {te_bad} defective.** A model predicting the majority "
        f"class scores {maj:.1%} accuracy while catching nothing. Report AUROC and "
        f"precision/recall; accuracy is not a defensible headline number here.\n"
    )

    # --- 3. defect size -----------------------------------------------------
    fr = np.array([r["defect_frac"] for r in rows if r["defect_frac"] is not None])
    out.append("## Defect size (fraction of pixels marked defective)\n")
    out.append(f"{len(fr)} masks found.\n")
    out.append("| stat | value |\n|---|---|")
    for k, v in [
        ("min", fr.min()), ("p25", np.percentile(fr, 25)), ("median", np.median(fr)),
        ("p75", np.percentile(fr, 75)), ("max", fr.max()), ("mean", fr.mean()),
    ]:
        out.append(f"| {k} | {v:.4%} |")
    per_type = defaultdict(list)
    for r in rows:
        if r["defect_frac"] is not None:
            per_type[r["defect"]].append(r["defect_frac"])
    out.append("\n| defect type | n | median area |\n|---|---|---|")
    for d, v in sorted(per_type.items()):
        out.append(f"| {d} | {len(v)} | {np.median(v):.4%} |")
    out.append(
        f"\nThe median defect covers **{np.median(fr):.2%}** of the image. Predicting "
        f"'no defect' for every pixel already yields ~{1 - fr.mean():.2%} pixel accuracy, so "
        f"pixel accuracy is useless - use pixel AUROC / PRO. It also bounds how far we can "
        f"downsample: at 224x224 the smallest defect here occupies about "
        f"{fr.min() * 224 * 224:.1f} pixels.\n"
    )

    # --- 4. resolution / mode ----------------------------------------------
    out.append("## Resolution & channels\n")
    out.append("| property | values |\n|---|---|")
    out.append(f"| size | {sorted({(r['w'], r['h']) for r in rows})} |")
    out.append(f"| mode | {sorted({r['mode'] for r in rows})} |")

    # --- 5. duplicates ------------------------------------------------------
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha"]].append(r)
    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    out.append("\n## Exact duplicates\n")
    if not dups:
        out.append("None. No train/test contamination via identical files.\n")
    else:
        out.append(f"**{len(dups)} duplicated image(s):**\n")
        for v in dups.values():
            out.append(f"- {', '.join(f'{r['split']}/{r['defect']}/{r['path'].name}' for r in v)}")

    # --- 6. brightness confound --------------------------------------------
    g = [r["mean"] for r in rows if r["label"] == 0]
    b = [r["mean"] for r in rows if r["label"] == 1]
    out.append("\n## Exposure confound\n")
    out.append(
        f"Mean grey level - good {np.mean(g):.1f}+/-{np.std(g):.1f}, "
        f"defective {np.mean(b):.1f}+/-{np.std(b):.1f} "
        f"(difference {abs(np.mean(g) - np.mean(b)):.1f}).\n"
    )
    out.append(
        "If this gap were large, a model could separate the classes on global brightness "
        "alone and the heatmaps would be meaningless - worth re-checking on self-collected "
        "data, where lighting is not controlled.\n"
    )
    return "\n".join(out)


def figure(cat: str, rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    c = Counter(f"{r['split']}/{r['defect']}" for r in rows)
    k = sorted(c)
    ax[0, 0].barh(k, [c[i] for i in k], color=[NEUTRAL if "good" in i else DEFECT for i in k])
    ax[0, 0].set_title("class balance")

    fr = np.array([r["defect_frac"] for r in rows if r["defect_frac"] is not None])
    ax[0, 1].hist(fr * 100, bins=30, color=DEFECT)
    ax[0, 1].set_title("defect area (% of pixels)")
    ax[0, 1].set_xlabel("%")

    for lbl, col, nm in ((0, NEUTRAL, "good"), (1, DEFECT, "defective")):
        ax[0, 2].hist([r["mean"] for r in rows if r["label"] == lbl], bins=25,
                      alpha=0.6, color=col, label=nm)
    ax[0, 2].legend()
    ax[0, 2].set_title("mean grey level")

    # one example per defect type, with mask outline
    types = sorted({r["defect"] for r in rows if r["label"] == 1})
    for i, d in enumerate(types[:3]):
        r = next(x for x in rows if x["defect"] == d)
        ax[1, i].imshow(Image.open(r["path"]))
        m = r["path"].parents[2] / "ground_truth" / d / f"{r['path'].stem}_mask.png"
        if m.exists():
            ax[1, i].contour(np.asarray(Image.open(m).convert("L")) > 0,
                             levels=[0.5], colors=[MASK], linewidths=1.5)
        ax[1, i].set_title(f"{d}  ({r['defect_frac']:.2%})")
        ax[1, i].axis("off")
    for j in range(len(types[:3]), 3):
        ax[1, j].axis("off")

    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("category", default="bottle", nargs="?")
    a = p.parse_args()

    cat_dir = ROOT / "data" / "mvtec" / a.category
    rows = scan(cat_dir)

    md = ROOT / "reports" / f"eda_{a.category}.md"
    png = ROOT / "reports" / f"eda_{a.category}.png"
    md.write_text(report(a.category, rows))
    figure(a.category, rows, png)

    # machine-readable, so later milestones can assert against these numbers
    (ROOT / "reports" / f"eda_{a.category}.json").write_text(
        json.dumps(
            [{**r, "path": str(r["path"].relative_to(ROOT))} for r in rows], indent=1
        )
    )
    print(md.read_text())
    print(f"\nwrote {md}\nwrote {png}")


if __name__ == "__main__":
    main()
