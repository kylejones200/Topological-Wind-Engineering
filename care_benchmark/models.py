"""Model families for the CARE benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from care_benchmark.windowing import (
    build_trajectory_matrix,
    extract_window_feature_matrix,
    ground_truth_anomaly_mask,
    normal_mask,
)
from tda_utils import TurbineConfig, compute_power_curve_vectorized


MODEL_FAMILIES = [
    "power_curve_residual",
    "statistical",
    "spectral",
    "isolation_forest",
    "tda_only",
    "tda_plus_conventional",
]


@dataclass
class FamilySpec:
    name: str
    feature_sets: List[str]
    use_isolation_forest: bool = False


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "power_curve_residual": FamilySpec("power_curve_residual", ["power_curve"]),
    "statistical": FamilySpec("statistical", ["statistical"]),
    "spectral": FamilySpec("spectral", ["spectral"]),
    "isolation_forest": FamilySpec("isolation_forest", ["statistical"], use_isolation_forest=True),
    "tda_only": FamilySpec("tda_only", ["tda"]),
    "tda_plus_conventional": FamilySpec("tda_plus_conventional", ["combined"]),
}


def expected_power_from_wind(wind: np.ndarray) -> np.ndarray:
    return compute_power_curve_vectorized(wind, TurbineConfig(rated_power=2.0))


def _fit_transformer(X: np.ndarray) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return imputer, scaler, X_scaled


def _transform(imputer: SimpleImputer, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(imputer.transform(X))


def fit_event_model(
    train_df: pd.DataFrame,
    sensors: Dict[str, str],
    spec: FamilySpec,
    window_size: int,
    stride: int,
    seed: int,
) -> Dict:
    values = build_trajectory_matrix(train_df, sensors)
    wind = values[:, 0]
    exp_power = expected_power_from_wind(wind)
    feature_sets = spec.feature_sets
    X, _ = extract_window_feature_matrix(values, window_size, stride, feature_sets, expected_power=exp_power)
    train_normal = normal_mask(train_df["status_type_id"]).to_numpy()
    window_normal = np.array([train_normal[max(0, i - window_size + 1) : i + 1].all() for i in range(len(train_df))])
    window_normal = window_normal[np.arange(window_size - 1, len(train_df), stride)[: len(X)]]
    X_normal = X[window_normal] if window_normal.any() else X
    if len(X_normal) == 0:
        X_normal = X

    imputer, scaler, X_scaled = _fit_transformer(X_normal)
    model = None
    threshold = np.quantile(np.linalg.norm(X_scaled, axis=1), 0.95)
    if spec.use_isolation_forest:
        model = IsolationForest(
            n_estimators=100,
            contamination=0.09,
            random_state=seed,
        )
        model.fit(X_scaled)
        scores = -model.score_samples(X_scaled)
        threshold = float(np.quantile(scores, 0.95))
    return {
        "spec": spec,
        "sensors": sensors,
        "window_size": window_size,
        "stride": stride,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "threshold": threshold,
    }


def predict_event(
    fitted: Dict,
    df: pd.DataFrame,
) -> pd.Series:
    values = build_trajectory_matrix(df, fitted["sensors"])
    wind = values[:, 0]
    exp_power = expected_power_from_wind(wind)
    X, ends = extract_window_feature_matrix(
        values,
        fitted["window_size"],
        fitted["stride"],
        fitted["spec"].feature_sets,
        expected_power=exp_power,
    )
    scores = pd.Series(0.0, index=df.index)
    if len(X) == 0:
        return scores.astype(bool)
    X_scaled = _transform(fitted["imputer"], fitted["scaler"], X)
    if fitted["model"] is not None:
        point_scores = -fitted["model"].score_samples(X_scaled)
    else:
        point_scores = np.linalg.norm(X_scaled, axis=1)
    threshold = fitted["threshold"]
    for end, score in zip(ends, point_scores):
        scores.iloc[end] = score
    # propagate window decision to trailing stride region
    predictions = pd.Series(False, index=df.index)
    for end, score in zip(ends, point_scores):
        start = max(0, end - fitted["stride"] + 1)
        predictions.iloc[start : end + 1] |= score >= threshold
    return predictions


def fit_pool_model(
    train_events,
    dataset,
    spec: FamilySpec,
    window_size: int,
    stride: int,
    seed: int,
) -> Dict:
    """Fit one model on pooled training events (leakage-safe: train split only)."""
    chunks = []
    sensors = None
    for event in train_events:
        sensors = dataset.select_key_sensors(event.wind_farm) or sensors
        if not sensors:
            sensors = {
                "wind": "sensor_wind",
                "power": "sensor_power",
                "rotor": "sensor_rotor",
            }
        values = build_trajectory_matrix(event.train, sensors)
        wind = values[:, 0]
        exp_power = expected_power_from_wind(wind)
        X, _ = extract_window_feature_matrix(values, window_size, stride, spec.feature_sets, expected_power=exp_power)
        if len(X):
            chunks.append(X)
    if not chunks:
        raise ValueError("No training features extracted")
    X_all = np.vstack(chunks)
    imputer, scaler, X_scaled = _fit_transformer(X_all)
    model = None
    threshold = float(np.quantile(np.linalg.norm(X_scaled, axis=1), 0.95))
    if spec.use_isolation_forest:
        model = IsolationForest(n_estimators=100, contamination=0.09, random_state=seed)
        model.fit(X_scaled)
        scores = -model.score_samples(X_scaled)
        threshold = float(np.quantile(scores, 0.95))
    return {
        "spec": spec,
        "sensors": sensors,
        "window_size": window_size,
        "stride": stride,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "threshold": threshold,
    }
