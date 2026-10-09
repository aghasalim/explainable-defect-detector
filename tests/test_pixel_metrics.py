"""Pixel AUROC and AUPRO on synthetic masks where the answer is known.

AUPRO follows the MVTec AD evaluation code: 8-connected regions weighted
equally, FPR over every defect-free pixel including those of normal images,
an exact curve, trapezoid up to FPR 0.3 with interpolation, divided by 0.3.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "edd"))

from explain import aupro, localisation_metrics  # noqa: E402


def slow_aupro(maps, masks, limit=0.3):
    """The definition, written out threshold by threshold."""
    regions, neg = [], masks == 0
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m > 0, structure=np.ones((3, 3)))
        regions += [(i, lab == r) for r in range(1, n + 1)]
    xs, ys = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        pred = maps >= t
        xs.append((pred & neg).sum() / neg.sum())
        ys.append(np.mean([pred[i][r].mean() for i, r in regions]))
    xs, ys = np.array(xs), np.array(ys)
    y_lim = np.interp(limit, xs, ys)
    keep = xs <= limit
    return np.trapezoid(np.append(ys[keep], y_lim), np.append(xs[keep], limit)) / limit


def two_regions():
    """One 20x20 region and one 3x3 region in a 64x64 image."""
    gt = np.zeros((1, 64, 64))
    gt[0, 0:20, 0:20] = 1
    gt[0, 50:53, 50:53] = 1
    return gt


def test_perfect_map_scores_one():
    gt = two_regions()
    assert aupro(gt.copy(), gt) == pytest.approx(1.0)


def test_inverted_map_scores_zero():
    gt = two_regions()
    assert aupro(1 - gt, gt) == pytest.approx(0.0)


def test_each_region_counts_the_same():
    """Find the 400-pixel region, miss the 9-pixel one: PRO is 0.5 at every FPR."""
    gt = two_regions()
    pred = np.full_like(gt, 0.5)
    pred[0, 0:20, 0:20] = 1.0
    pred[0, 50:53, 50:53] = 0.0
    assert aupro(pred, gt) == pytest.approx(0.5)
    # pixel AUROC sees 400 of 409 defect pixels ranked first and calls it near perfect
    loc = localisation_metrics(pred, gt, np.random.default_rng(0))["model"]
    assert loc["pixel_auroc"] > 0.97


def test_diagonal_neighbours_are_one_region():
    """8-connectivity: a pixel touching a 2x2 block at a corner joins it.

    Find only the single pixel. As one 5-pixel region that is PRO 0.2. With
    4-connectivity it would be two regions, one fully found, giving 0.5.
    """
    gt = np.zeros((1, 8, 8))
    gt[0, 2, 2] = 1
    gt[0, 3:5, 3:5] = 1
    pred = np.zeros_like(gt)
    pred[0, 2, 2] = 1.0
    pred[0, 3:5, 3:5] = -1.0      # below every defect-free pixel
    assert aupro(pred, gt) == pytest.approx(0.2)


def test_normal_images_count_towards_fpr():
    """False positives on a defect-free image must lower AUPRO."""
    gt = np.zeros((2, 32, 32))
    gt[0, 10:20, 10:20] = 1
    pred = gt.copy() * 0.5
    clean = aupro(pred, gt)
    noisy = pred.copy()
    noisy[1, :16, :] = 0.9        # half of the normal image outscores the defect
    assert clean == pytest.approx(1.0)
    assert aupro(noisy, gt) < clean
    # 512 of 1948 defect-free pixels sit above the defect, so PRO is 0 up to
    # FPR 512/1948 = 0.263 and 1 after it
    assert aupro(noisy, gt) == pytest.approx((0.3 - 512 / 1948) / 0.3)
    assert aupro(noisy, gt) == pytest.approx(slow_aupro(noisy, gt))


def test_defect_found_only_past_the_fpr_limit_scores_zero():
    """Half the defect-free pixels outscore the defect, so PRO is 0 until FPR 0.5."""
    gt = np.zeros((1, 10, 10))
    gt[0, :, :5] = 1
    pred = np.zeros_like(gt)
    pred[0, :, :5] = 1.0
    pred[0, :, 5:] = np.linspace(0, 2, 50).reshape(10, 5)
    # 25 of the 50 defect-free pixels score above 1.0
    assert aupro(pred, gt) == pytest.approx(0.0)
    assert aupro(pred, gt, fpr_limit=1.0) == pytest.approx(slow_aupro(pred, gt, 1.0))


@pytest.mark.parametrize("seed", range(5))
def test_matches_the_definition_on_random_maps(seed):
    rng = np.random.default_rng(seed)
    gt = np.zeros((4, 24, 24))
    for i in range(3):            # the fourth image stays normal
        for _ in range(rng.integers(1, 4)):
            y, x = rng.integers(0, 20, 2)
            gt[i, y:y + rng.integers(1, 5), x:x + rng.integers(1, 5)] = 1
    # rounded so there are ties to handle
    pred = np.round(rng.random(gt.shape) + gt * rng.random(), 2)
    assert aupro(pred, gt) == pytest.approx(slow_aupro(pred, gt), abs=1e-9)


def test_random_map_aupro_is_near_its_expectation():
    """A map with no information has PRO = FPR, so AUPRO up to 0.3 is 0.15."""
    rng = np.random.default_rng(0)
    gt = np.zeros((8, 64, 64))
    gt[:4, 20:40, 20:40] = 1
    assert aupro(rng.random(gt.shape), gt) == pytest.approx(0.15, abs=0.01)
