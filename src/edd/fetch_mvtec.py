"""Fetch one MVTec AD category and rebuild the canonical on-disk layout.

The upstream mvtec.com download link is dead, so we pull from the Voxel51
FiftyOne mirror on HuggingFace, which flattens every image into arbitrary
shards (data/data_7/031.png). data/_mvtec_index.json maps each file back to
its (category, split, defect, mask), so we can restore the layout every
MVTec paper and tool expects:

    <cat>/train/good/*.png
    <cat>/test/good/*.png
    <cat>/test/<defect>/*.png
    <cat>/ground_truth/<defect>/*_mask.png

Keeping the canonical layout matters: it is what makes our numbers
comparable to published PatchCore/PaDiM results instead of only to ourselves.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

REPO = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main"
ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "_mvtec_index.json"


def load_index() -> list[dict]:
    return json.loads(INDEX.read_text())["samples"]


def category_stats(samples: list[dict]) -> dict[str, Counter]:
    """Per-category counts. Runs off the index alone - no download needed."""
    stats: dict[str, Counter] = {}
    for s in samples:
        cat = s["category"]["label"]
        defect = s["defect"]["label"]
        split = s["split"]
        c = stats.setdefault(cat, Counter())
        c[f"{split}/{defect}"] += 1
        c["_total"] += 1
        if defect != "good":
            c["_defect_types"] = len(
                {k.split("/")[1] for k in c if k.startswith("test/") and not k.endswith("/good")}
            )
    return stats


def _canon(name: str) -> str:
    """Undo the mirror's flattening suffix: '000-94.png' -> '000.png'.

    Every file in the mirror is stored in a shared pool, so the uploader
    appended '-<n>' to keep names unique across categories. The suffix differs
    between an image and its own mask ('000-94.png' vs '000_mask-67.png'),
    which silently breaks image<->mask pairing if left in place.
    """
    return re.sub(r"-\d+(?=\.\w+$)", "", name)


def _get(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())


def fetch(category: str, out: Path, workers: int = 12) -> Path:
    samples = [s for s in load_index() if s["category"]["label"] == category]
    if not samples:
        sys.exit(f"unknown category: {category}")

    jobs: list[tuple[str, Path]] = []
    for s in samples:
        stem = _canon(Path(s["filepath"]).name)
        split, defect = s["split"], s["defect"]["label"]
        jobs.append((f"{REPO}/{s['filepath']}", out / category / split / defect / stem))
        # ground truth masks exist only for anomalous test images
        if mask := s.get("defect_mask"):
            jobs.append(
                (
                    f"{REPO}/{mask['mask_path']}",
                    out / category / "ground_truth" / defect / _canon(Path(mask["mask_path"]).name),
                )
            )

    print(f"{category}: {len(samples)} images + masks -> {len(jobs)} files")
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(lambda j: _get(*j), jobs), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    return out / category


def validate(cat_dir: Path) -> None:
    """Every anomalous test image must have exactly one matching mask.

    This is the invariant the flattening suffix broke; assert it at fetch time
    so a rename bug can never reach the EDA or the metrics.
    """
    missing = []
    for d in sorted((cat_dir / "test").iterdir()):
        if d.name == "good":
            continue
        for img in d.glob("*.png"):
            if not (cat_dir / "ground_truth" / d.name / f"{img.stem}_mask.png").exists():
                missing.append(img)
    n = sum(1 for d in (cat_dir / "test").iterdir() if d.name != "good" for _ in d.glob("*.png"))
    if missing:
        sys.exit(f"FAIL: {len(missing)}/{n} anomalous images have no mask, e.g. {missing[:3]}")
    print(f"validated: {n}/{n} anomalous test images paired with a mask")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("category", nargs="?", help="e.g. bottle; omit to list categories")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "mvtec")
    a = p.parse_args()

    if not a.category:
        stats = category_stats(load_index())
        print(f"{'category':14} {'total':>6} {'train':>6} {'test_ok':>8} {'test_bad':>9} {'types':>6}")
        for cat, c in sorted(stats.items()):
            train = sum(v for k, v in c.items() if k.startswith("train/"))
            ok = c.get("test/good", 0)
            bad = sum(v for k, v in c.items() if k.startswith("test/") and k != "test/good")
            types = len({k.split("/", 1)[1] for k in c if k.startswith("test/") and k != "test/good"})
            print(f"{cat:14} {c['_total']:6} {train:6} {ok:8} {bad:9} {types:6}")
        return

    dest = fetch(a.category, a.out)
    validate(dest)
    print(f"done -> {dest}")


if __name__ == "__main__":
    main()
