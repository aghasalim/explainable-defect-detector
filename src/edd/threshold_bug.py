"""Reproduce the 99th-percentile threshold bug next to its fix.

The first export.py (before commit 0dfdb13) held out 10% of the training
normals, built the bank from the other 90%, and put the threshold at the 99th
percentile of the held-out scores. With 21 to 39 held-out images the 99th
percentile is close to the sample maximum, and the maximum of n draws sits at
about the n/(n+1) quantile, the 96th at n = 28. So the threshold was too low
and the false-alarm rate came out far above the 1% target.

Two parts:

  synthetic()   no data. Draws calibration scores from a known distribution, so
                the true false-alarm rate of any threshold is exact. Shows the
                bug and the fix in isolation from MVTec.
  real()        MVTec. Runs the old rule, the 5-fold rule with a 99th
                percentile, and the shipped 5-fold rule with a tolerance bound,
                and measures each one's false-alarm rate on the test normals.

Run:  python src/edd/threshold_bug.py --synthetic
      python src/edd/threshold_bug.py [categories...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def synthetic(n_holdout: int = 28, n_kfold: int = 280, trials: int = 20000,
              seed: int = 0) -> dict:
    """True false-alarm rate of each rule when normal scores are N(0, 1).

    n_holdout = 28 and n_kfold = 280 are carpet's numbers: 10% of its 280
    training images for the old rule, all 280 for the k-fold rule.
    """
    from export import choose_threshold

    rng = np.random.default_rng(seed)
    out = {}
    for name, n, method in (("old: 10% holdout, 99th percentile", n_holdout, "quantile"),
                            ("5-fold, 99th percentile", n_kfold, "quantile"),
                            ("5-fold, tolerance bound (shipped)", n_kfold, "tolerance")):
        far = np.array([stats.norm.sf(choose_threshold(rng.standard_normal(n), 0.01, method)[0])
                        for _ in range(trials)])
        out[name] = {"n_calibration": n, "mean_far": float(far.mean()),
                     "median_far": float(np.median(far)),
                     "share_of_runs_above_1pct": float((far > 0.01).mean())}
    return out


def real(categories: list[str], frac: float = 0.01, seed: int = 0) -> list[dict]:
    import torch
    from dataset import MVTecCategory
    from export import calibration_scores, choose_threshold
    from patchcore import PatchFeatures, coreset, device, extract, score

    dev = device()
    model = PatchFeatures().to(dev)
    rows = []
    for c in categories:
        torch.manual_seed(seed)
        tr, _, _, _, _ = extract(model, MVTecCategory(c, "train", 224, True), dev)
        te, labels, _, _, _ = extract(model, MVTecCategory(c, "test", 224, True), dev)
        normal = labels == 0
        n = tr.shape[0]

        # the old export.py, line for line: seed-0 permutation, 10% held out,
        # bank from the other 90%, 99th percentile of the held-out scores
        perm = np.random.default_rng(seed).permutation(n)
        n_cal = max(1, int(round(n * 0.10)))
        cal_idx, bank_idx = perm[:n_cal], perm[n_cal:]
        flat = tr[bank_idx].reshape(-1, tr.shape[-1])
        old_bank = flat[coreset(flat, frac, dev, seed)]
        old_cal = score(old_bank, tr[cal_idx], dev).max(dim=1).values.numpy()
        old_thr = float(np.quantile(old_cal, 0.99))
        old = score(old_bank, te, dev).max(dim=1).values.numpy() >= old_thr

        # the fix: every training image scored out of fold, full bank shipped
        cal = calibration_scores(tr, frac, dev, 5, seed)
        q_thr, _ = choose_threshold(cal, 0.01, "quantile")
        t_thr, _ = choose_threshold(cal, 0.01, "tolerance")
        flat = tr.reshape(-1, tr.shape[-1])
        full = score(flat[coreset(flat, frac, dev, seed)], te, dev).max(dim=1).values.numpy()
        q, t = full >= q_thr, full >= t_thr
        old_fa, q_fa, t_fa = (int(x[normal].sum()) for x in (old, q, t))
        old_rec, q_rec, t_rec = (float(x[~normal].mean()) for x in (old, q, t))

        nn = int(normal.sum())
        row = {"category": c, "n_train": n, "n_test_normal": nn,
               "old_n_calibration": n_cal, "old_threshold": old_thr,
               "old_false_alarms": old_fa, "old_recall": old_rec,
               "kfold_q99_threshold": q_thr, "kfold_q99_false_alarms": q_fa,
               "kfold_q99_recall": q_rec,
               "kfold_tolerance_threshold": t_thr, "kfold_tolerance_false_alarms": t_fa,
               "kfold_tolerance_recall": t_rec}
        rows.append(row)
        print(f"{c:11} old {old_fa:2}/{nn} ({old_fa / nn:5.1%}) | 5-fold q99 {q_fa:2}/{nn} "
              f"({q_fa / nn:5.1%}) | 5-fold tolerance {t_fa:2}/{nn} ({t_fa / nn:5.1%})",
              flush=True)
    return rows


def summary(rows: list[dict]) -> str:
    o = ["| rule | within 1% target | mean FAR | mean recall |", "|---|---|---|---|"]
    for name, k in (("10% holdout, 99th percentile (old)", "old"),
                    ("5-fold, 99th percentile", "kfold_q99"),
                    ("5-fold, tolerance bound (shipped)", "kfold_tolerance")):
        fars = [r[f"{k}_false_alarms"] / r["n_test_normal"] for r in rows]
        o.append(f"| {name} | {sum(f <= 0.01 for f in fars)} / {len(rows)} | "
                 f"{np.mean(fars):.1%} | {np.mean([r[f'{k}_recall'] for r in rows]):.1%} |")
    return "\n".join(o)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="*", default=CATEGORIES)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    if a.synthetic:
        for k, v in synthetic().items():
            print(f"{k:36} n={v['n_calibration']:3} mean FAR {v['mean_far']:.2%} "
                  f"median {v['median_far']:.2%} runs above 1%: {v['share_of_runs_above_1pct']:.1%}")
    else:
        rows = real(a.categories)
        (ROOT / "reports" / "threshold_bug.json").write_text(json.dumps(rows, indent=1))
        print(summary(rows))
