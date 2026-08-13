"""Collect every reports/<method>_<category>.json into comparison tables.

Published PatchCore numbers (Roth et al., CVPR 2022) are shown as a reference
column. The point is not to claim parity - it is to make any gap visible, so a
weak result gets reported as weak instead of quietly framed as a success.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Roth et al. 2022, PatchCore-1%: (image AUROC, pixel AUROC)
PAPER = {
    "bottle": (1.000, 0.986), "cable": (0.993, 0.984), "capsule": (0.980, 0.988),
    "carpet": (0.987, 0.990), "grid": (0.981, 0.987), "hazelnut": (1.000, 0.987),
    "leather": (1.000, 0.993), "metal_nut": (0.998, 0.984), "pill": (0.966, 0.976),
    "screw": (0.981, 0.994), "tile": (0.987, 0.959), "toothbrush": (1.000, 0.987),
    "transistor": (1.000, 0.964), "wood": (0.992, 0.951), "zipper": (0.985, 0.989),
}

# headline config: paper preprocessing, 1% coreset
HEADLINE = "patchcore-224crop"


def load() -> list[tuple[str, str, dict]]:
    rows = []
    for f in sorted((ROOT / "reports").glob("*.json")):
        method, _, cat = f.stem.rpartition("_")
        # eda_*.json lives here too and is a plain list, not a metrics report
        if cat not in PAPER or not method.startswith(("baseline", "patchcore")):
            continue
        rows.append((cat, method, json.loads(f.read_text())["metrics"]))
    return rows


def main() -> None:
    rows = load()
    out = ["# Results\n"]

    # ---- headline -----------------------------------------------------------
    out += [
        "## Headline: PatchCore, 1% coreset, paper preprocessing\n",
        "| category | image AUROC | paper | gap | pixel AUROC | paper | gap | sec |",
        "|" + "---|" * 8,
    ]
    for cat, method, m in sorted(rows):
        if method != HEADLINE:
            continue
        pi, pp = PAPER[cat]
        out.append(
            f"| {cat} | **{m['image_auroc']:.4f}** | {pi:.3f} | {m['image_auroc'] - pi:+.4f} "
            f"| **{m['pixel_auroc']:.4f}** | {pp:.3f} | {m['pixel_auroc'] - pp:+.4f} "
            f"| {m['seconds']:.0f} |"
        )

    # ---- everything else ----------------------------------------------------
    out += [
        "\n## All runs\n",
        "| category | method | preproc | coreset | bank | image AUROC | pixel AUROC "
        "| acc @F1 | majority acc | sec |",
        "|" + "---|" * 10,
    ]
    for cat, method, m in sorted(rows, key=lambda r: (r[0], -r[2]["image_auroc"])):
        pre = "crop" if m.get("center_crop") else "resize"
        cs = f"{m['coreset_frac']:.0%}" if "coreset_frac" in m else "-"
        bank = f"{m['bank_size']:,}" if "bank_size" in m else "-"
        out.append(
            f"| {cat} | {method.replace('patchcore', 'pc')} | {pre} | {cs} | {bank} "
            f"| **{m['image_auroc']:.4f}** | {m.get('pixel_auroc', float('nan')):.4f} "
            f"| {m['accuracy_at_best_f1']:.4f} | {m['majority_class_accuracy']:.4f} "
            f"| {m.get('seconds', 0):.0f} |"
        )

    out.append(
        "\n`acc @F1` vs `majority acc` is the sanity column: where they are close, the model "
        "is barely beating a constant prediction, whatever the AUROC says.\n"
        "\nPixel AUROC is **not comparable across preprocessing rows**: under `crop` it is "
        "computed over a different (zoomed, smaller) pixel set than under `resize`. Compare "
        "pixel numbers only within the same preprocessing column.\n"
    )

    p = ROOT / "reports" / "results.md"
    p.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
