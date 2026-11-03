"""Sanity checks for the statistics in analysis/report.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from report import paired_bootstrap_diff, wilson_ci  # noqa: E402


def test_wilson_known_value():
    # 8/10: Wilson 95% interval is approx [0.490, 0.943] (textbook value)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_wilson_extremes_stay_in_unit_interval():
    assert wilson_ci(0, 30)[0] == 0.0 and wilson_ci(0, 30)[1] > 0
    assert wilson_ci(30, 30)[1] == 1.0 and wilson_ci(30, 30)[0] < 1


def test_paired_bootstrap_identical_models():
    a = np.array([1, 0, 1, 1, 0] * 6, dtype=float)
    diff, lo, hi = paired_bootstrap_diff(a, a.copy())
    assert diff == lo == hi == 0.0


def test_paired_bootstrap_clear_winner():
    a = np.ones(60)
    b = np.r_[np.ones(20), np.zeros(40)]
    diff, lo, hi = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(2 / 3)
    assert lo > 0.4 and hi <= 1.0
