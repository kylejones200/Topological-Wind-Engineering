"""
Shared utilities for Topological-Wind-Engineering plotting and simulation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TufteColors:
    BLACK = "#1a1a1a"
    RED = "#c41e3a"
    GRAY = "#888888"
    BLUE = "#2166ac"


def setup_tufte_plot(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Apply minimalist Tufte-style axes labels and spines."""
    plt.rcParams["font.family"] = "serif"
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


@dataclass
class TurbineConfig:
    cut_in_wind_speed: float = 3.0
    rated_wind_speed: float = 12.0
    cut_out_wind_speed: float = 25.0
    rated_power: float = 2.0
    power_curve_exponent: float = 2.5


def compute_power_curve_vectorized(
    windspeed: np.ndarray, config: TurbineConfig = TurbineConfig()
) -> np.ndarray:
    """IEC-style rated power curve in MW."""
    ws = np.asarray(windspeed, dtype=float)
    power = np.zeros_like(ws)
    rated = config.rated_power
    cut_in = config.cut_in_wind_speed
    rated_ws = config.rated_wind_speed
    cut_out = config.cut_out_wind_speed
    exp = config.power_curve_exponent

    operating = (ws >= cut_in) & (ws <= cut_out)
    ramp = operating & (ws < rated_ws)
    rated_region = operating & (ws >= rated_ws)

    power[ramp] = rated * ((ws[ramp] - cut_in) / (rated_ws - cut_in)) ** exp
    power[rated_region] = rated
    return power


def add_power_noise(
    power: np.ndarray, noise_std: float = 0.05, rated_power: float = 2.0
) -> np.ndarray:
    """Add proportional Gaussian noise and clip to physical bounds."""
    p = np.asarray(power, dtype=float)
    noisy = p + np.random.normal(0, noise_std * rated_power, size=p.shape)
    return np.clip(noisy, 0.0, rated_power * 1.1)


def extract_datetime_features(timestamps: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """Return hour-of-day and day-of-year arrays for synthetic wind patterns."""
    ts = pd.DatetimeIndex(timestamps)
    hour_fractional = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    dayofyear = ts.dayofyear.to_numpy(dtype=float)
    return {"hour_fractional": hour_fractional.to_numpy(dtype=float), "dayofyear": dayofyear}


def create_seasonal_pattern(days: np.ndarray, amplitude: float = 2.0) -> np.ndarray:
    """Annual sinusoidal wind-speed perturbation."""
    return amplitude * np.sin(2 * np.pi * days / 365.25)


def create_diurnal_pattern(hours: np.ndarray, amplitude: float = 1.5) -> np.ndarray:
    """Diurnal sinusoidal wind-speed perturbation."""
    return amplitude * np.sin(2 * np.pi * hours / 24.0)


def extract_persistence_lifetimes(
    diagram: np.ndarray, remove_infinite: bool = True
) -> np.ndarray:
    """Return persistence lifetimes (death - birth) for a persistence diagram."""
    if diagram is None or len(diagram) == 0:
        return np.array([])
    dgm = np.asarray(diagram, dtype=float)
    if remove_infinite:
        finite = np.isfinite(dgm[:, 1])
        dgm = dgm[finite]
    if len(dgm) == 0:
        return np.array([])
    return dgm[:, 1] - dgm[:, 0]


def compute_persistence_entropy(diagram: np.ndarray) -> float:
    """Normalized Shannon entropy of finite persistence lifetimes."""
    lifetimes = extract_persistence_lifetimes(diagram, remove_infinite=True)
    if len(lifetimes) == 0:
        return 0.0
    total = float(np.sum(lifetimes))
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    if len(probs) <= 1:
        return 0.0
    return entropy / np.log(len(probs))


def print_classification_summary(results: Dict[str, Dict[str, Any]], best_model_name: str) -> None:
    """Log a short summary for the best classifier by accuracy."""
    best = results[best_model_name]
    print(f"\nBest model: {best_model_name}")
    print(f"  Accuracy: {best.get('accuracy', float('nan')):.3f}")
    if best.get("f1") is not None:
        print(f"  F1: {best['f1']:.3f}")
    if best.get("auc") is not None:
        print(f"  AUC: {best['auc']:.3f}")
