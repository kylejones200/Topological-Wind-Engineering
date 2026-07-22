"""Tests for shared tda_utils helpers."""
import numpy as np
import pandas as pd

from tda_utils import (
    TurbineConfig,
    add_power_noise,
    compute_persistence_entropy,
    compute_power_curve_vectorized,
    extract_persistence_lifetimes,
)


def test_power_curve_respects_cut_in_and_rated():
    config = TurbineConfig()
    ws = np.array([0.0, 2.9, 3.0, 7.0, 12.0, 20.0, 26.0])
    power = compute_power_curve_vectorized(ws, config)
    assert power[0] == 0.0
    assert power[1] == 0.0
    assert power[2] == 0.0
    assert 0.0 < power[3] < config.rated_power
    assert power[4] == config.rated_power
    assert power[5] == config.rated_power
    assert power[6] == 0.0


def test_persistence_lifetimes_ignore_infinite_deaths():
    diagram = np.array([[0.0, 1.0], [0.5, np.inf]])
    lifetimes = extract_persistence_lifetimes(diagram, remove_infinite=True)
    assert lifetimes.tolist() == [1.0]


def test_persistence_entropy_zero_for_empty_diagram():
    assert compute_persistence_entropy(np.empty((0, 2))) == 0.0


def test_add_power_noise_stays_within_bounds():
    np.random.seed(0)
    base = np.full(100, 1.0)
    noisy = add_power_noise(base, noise_std=0.01, rated_power=2.0)
    assert noisy.min() >= 0.0
    assert noisy.max() <= 2.2


def test_datetime_features_shape():
    from tda_utils import extract_datetime_features

    ts = pd.date_range("2020-01-01", periods=24, freq="h")
    feats = extract_datetime_features(ts)
    assert len(feats["hour_fractional"]) == 24
    assert len(feats["dayofyear"]) == 24
