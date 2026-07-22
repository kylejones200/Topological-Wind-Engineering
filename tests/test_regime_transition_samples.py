"""Tests for regime transition sample-based windowing."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ripser")

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIME_CODE = (
    REPO_ROOT
    / "seeing-around-corners-predicting-wind-turbine-regime-changes-through-topological-precursors"
    / "code"
)
sys.path.insert(0, str(REGIME_CODE))

from regime_transition import extract_zigzag_window  # noqa: E402


def _make_df(n=200):
    return pd.DataFrame(
        {
            "wind_turbulent": np.linspace(5, 10, n),
            "rotor_speed": np.linspace(10, 20, n),
            "power": np.linspace(500, 1500, n),
            "regime": np.zeros(n, dtype=int),
        }
    )


def test_extract_window_uses_sample_counts_not_minutes():
    df = _make_df()
    subwindows = extract_zigzag_window(
        df,
        transition_idx=100,
        window_samples=20,
        subwindow_samples=5,
        slide_samples=1,
        lead_samples=10,
    )
    assert subwindows is not None
    assert len(subwindows) > 0
    # Each subwindow should span 5 hourly rows, not 5 minutes of sub-hourly data
    assert len(subwindows[0]) == 5
