"""Checks on the threshold code in export.py, on tiny synthetic features."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "edd"))

import export  # noqa: E402


def test_calibration_coreset_uses_the_seed(monkeypatch):
    seen = []
    real = export.coreset

    def spy(x, frac, dev, seed=0):
        seen.append(seed)
        return real(x, frac, dev, seed)

    monkeypatch.setattr(export, "coreset", spy)
    feats = torch.randn(10, 4, 8)
    export.calibration_scores(feats, 0.5, torch.device("cpu"), k_folds=5, seed=3)
    assert seen == [3] * 5


def test_tolerance_rank_needs_299_samples_for_99_at_95():
    # 1 - 0.99**n >= 0.95 first holds at n = 299, and then only the maximum works
    assert export.tolerance_rank(298) is None
    assert export.tolerance_rank(299) == 299
