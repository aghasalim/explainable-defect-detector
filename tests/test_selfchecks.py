"""The project's runnable checks.

Deliberately narrow: these run with no dataset, no model weights and no
network, so CI stays fast and green for the right reason. They cover the logic
that is easy to get silently wrong - metric definitions, coreset coverage,
map orientation - not the numbers, which depend on data that is not committed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "edd"))

import baseline           # noqa: E402
import explain            # noqa: E402
import patchcore          # noqa: E402


def test_baseline_selfcheck():
    baseline.demo()


def test_patchcore_selfcheck():
    patchcore.demo()


def test_explain_selfcheck():
    explain.demo()


def test_metrics_reject_a_constant_predictor():
    """A model that outputs one number must not look good on any metric."""
    labels = np.array([0] * 20 + [1] * 60)          # the real MVTec imbalance
    m = baseline.evaluate(labels, np.full(80, 0.5))
    assert m["image_auroc"] == pytest.approx(0.5)
    # ...while its accuracy is high, which is exactly why accuracy is not reported alone
    assert m["accuracy_at_best_f1"] == pytest.approx(m["majority_class_accuracy"])


def test_anomaly_map_preserves_location():
    """A hot patch in a corner must stay in that corner after upsample+blur."""
    for corner, (yr, xr) in {
        0: ((0, 60), (0, 60)),
        6: ((0, 60), (164, 224)),
        42: ((164, 224), (0, 60)),
        48: ((164, 224), (164, 224)),
    }.items():
        p = torch.zeros(1, 49)
        p[0, corner] = 1.0
        m = patchcore.to_maps(p, (7, 7), 224)[0, 0]
        y, x = divmod(int(torch.argmax(m)), 224)
        assert yr[0] <= y < yr[1] and xr[0] <= x < xr[1], (corner, y, x)


def test_blur_does_not_bias_borders():
    """Reflect padding: a flat map must stay flat, including at the edges.

    With zero padding the border drops below the interior, which silently
    under-scores defects at the image edge.
    """
    flat = torch.ones(1, 49)
    m = patchcore.to_maps(flat, (7, 7), 224)[0, 0]
    assert torch.allclose(m, m.mean(), atol=1e-4), (m.min().item(), m.max().item())


def test_aupro_weights_regions_equally():
    """One big region found + one small region missed must not score near 1.

    Pixel AUROC would be dominated by the big region; AUPRO must not be.
    """
    gt = np.zeros((1, 64, 64), dtype=np.float32)
    gt[0, 0:20, 0:20] = 1        # large region
    gt[0, 50:53, 50:53] = 1      # small region, far away
    pred = np.zeros_like(gt)
    pred[0, 0:20, 0:20] = 1      # only the large one is found
    assert explain.aupro(pred, gt) < 0.6
