"""Collect every reports/<method>_<category>.json into one comparison table.

Published PatchCore image-AUROC (Roth et al., CVPR 2022, Table 1) is included
as a reference column. The point is not to claim parity - it is to make the
gap visible, so a weak result is reported as weak instead of quietly framed
as a success.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Roth et al. 2022, PatchCore-1% image-level AUROC
PATCHCORE_PAPER = {
    "bottle": 1.000, "cable": 0.993, "capsule": 0.980, "carpet": 0.987, "grid": 0.981,
    "hazelnut": 1.000, "leather": 1.000, "metal_nut": 0.998, "pill": 0.966, "screw": 0.981,
    "tile": 0.987, "toothbrush": 1.000, "transistor": 1.000, "wood": 0.992, "zipper": 0.985,
}


def main() -> None:
    rows = []
    for f in sorted((ROOT / "reports").glob("*_*.json")):
        method, _, cat = f.stem.partition("_")
        if method not in {"baseline", "patchcore"}:
            continue
        m = json.loads(f.read_text())["metrics"]
        rows.append((cat, method, m))

    hdr = (f"| category | method | image AUROC | AP | best F1 | acc @F1 | majority acc | "
           f"PatchCore (paper) | gap |")
    out = ["# Results\n", hdr, "|" + "---|" * 9]
    for cat, method, m in sorted(rows):
        ref = PATCHCORE_PAPER.get(cat)
        gap = m["image_auroc"] - ref if ref else None
        out.append(
            f"| {cat} | {method} | **{m['image_auroc']:.4f}** | {m['average_precision']:.4f} "
            f"| {m['best_f1']:.4f} | {m['accuracy_at_best_f1']:.4f} "
            f"| {m['majority_class_accuracy']:.4f} "
            f"| {ref:.3f} | {gap:+.4f} |" if ref else
            f"| {cat} | {method} | **{m['image_auroc']:.4f}** | {m['average_precision']:.4f} "
            f"| {m['best_f1']:.4f} | {m['accuracy_at_best_f1']:.4f} "
            f"| {m['majority_class_accuracy']:.4f} | - | - |"
        )

    beats = [
        (c, m) for c, _, m in rows
        if m["accuracy_at_best_f1"] - m["majority_class_accuracy"] < 0.05
    ]
    out.append(
        "\n`acc @F1` vs `majority acc` is the sanity column: where they are close, the model "
        "is barely beating a constant prediction, whatever the AUROC says.\n"
    )
    if beats:
        out.append("Categories within 5 points of the majority-class baseline: "
                   + ", ".join(f"`{c}`" for c, _ in beats) + ".\n")

    p = ROOT / "reports" / "results.md"
    p.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
