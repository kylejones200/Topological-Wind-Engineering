"""Sliding-window feature extraction for CARE sequences."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import skew

from tda_utils import extract_persistence_lifetimes


def normal_mask(status: pd.Series) -> pd.Series:
    """CARE normal-operation mask (status_type_id == 0)."""
    return status.astype(int) == 0


def ground_truth_anomaly_mask(
    index: pd.Index,
    normal_idx: pd.Series,
    event_label: str,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> pd.Series:
    """CARE-compatible pointwise ground truth."""
    gt = pd.Series(data=~normal_idx, index=normal_idx.index, dtype=bool)
    if event_label == "anomaly":
        if isinstance(gt.index, pd.DatetimeIndex):
            gt.loc[event_start:event_end] = True
        else:
            mask = (index >= event_start) & (index <= event_end)
            gt.loc[mask] = True
    return gt.reindex(index).fillna(False).astype(bool)


def sliding_windows(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (n_windows, window_size, n_features) and end indices."""
    n, dim = values.shape
    if n < window_size:
        return np.empty((0, window_size, dim)), np.array([], dtype=int)
    ends = np.arange(window_size - 1, n, stride, dtype=int)
    windows = np.stack([values[end - window_size + 1 : end + 1] for end in ends], axis=0)
    return windows, ends


def statistical_features(window: np.ndarray) -> np.ndarray:
    feats = []
    for col in range(window.shape[1]):
        x = window[:, col]
        feats.extend([x.mean(), x.std(), x.min(), x.max(), skew(x) if len(x) > 2 else 0.0])
    return np.array(feats, dtype=float)


def spectral_features(window: np.ndarray, n_bands: int = 3) -> np.ndarray:
    feats = []
    for col in range(window.shape[1]):
        x = window[:, col] - window[:, col].mean()
        power = np.abs(rfft(x)) ** 2
        if len(power) == 0:
            feats.extend([0.0] * n_bands)
            continue
        splits = np.array_split(power, n_bands)
        feats.extend([float(s.mean()) for s in splits])
    return np.array(feats, dtype=float)


def tda_features(window: np.ndarray) -> np.ndarray:
    try:
        from ripser import ripser
    except ImportError:
        return np.zeros(10, dtype=float)
    x = window.astype(float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    result = ripser(x, maxdim=1)["dgms"]
    feats = []
    for dim in (0, 1):
        lifetimes = extract_persistence_lifetimes(result[dim], remove_infinite=True)
        feats.extend(
            [
                float(len(lifetimes)),
                float(lifetimes.sum()) if len(lifetimes) else 0.0,
                float(lifetimes.max()) if len(lifetimes) else 0.0,
                float(lifetimes.mean()) if len(lifetimes) else 0.0,
                float(lifetimes.std()) if len(lifetimes) else 0.0,
            ]
        )
    return np.array(feats, dtype=float)


def power_residual_features(window: np.ndarray, expected_power: np.ndarray) -> np.ndarray:
    residual = window[:, 1] - expected_power
    ratio = np.divide(
        window[:, 1],
        expected_power,
        out=np.zeros_like(window[:, 1]),
        where=expected_power > 1e-3,
    )
    return np.array(
        [
            residual.mean(),
            residual.std(),
            np.percentile(np.abs(residual), 95),
            ratio.mean(),
            ratio.std(),
            float((ratio < 0.85).mean()),
        ],
        dtype=float,
    )


def build_trajectory_matrix(df: pd.DataFrame, sensors: Dict[str, str]) -> np.ndarray:
    cols = [sensors[k] for k in ("wind", "power", "rotor") if k in sensors and sensors[k] in df.columns]
    if len(cols) < 2:
        numeric = [c for c in df.columns if c not in {"asset_id", "status_type_id"} and pd.api.types.is_numeric_dtype(df[c])]
        cols = numeric[:3] if len(numeric) >= 2 else numeric
    return df[cols].to_numpy(dtype=float)


def extract_window_feature_matrix(
    values: np.ndarray,
    window_size: int,
    stride: int,
    feature_sets: List[str],
    expected_power: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    windows, ends = sliding_windows(values, window_size, stride)
    if len(windows) == 0:
        return np.empty((0, 0)), ends
    rows = []
    for i, window in enumerate(windows):
        parts = []
        if "statistical" in feature_sets or "combined" in feature_sets:
            parts.append(statistical_features(window))
        if "spectral" in feature_sets:
            parts.append(spectral_features(window))
        if "tda" in feature_sets or "combined" in feature_sets:
            parts.append(tda_features(window))
        if "power_curve" in feature_sets:
            exp = expected_power[ends[i] - window_size + 1 : ends[i] + 1] if expected_power is not None else window[:, 1]
            parts.append(power_residual_features(window, exp))
        rows.append(np.concatenate(parts))
    return np.vstack(rows), ends
