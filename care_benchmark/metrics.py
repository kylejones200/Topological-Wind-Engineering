"""CARE score and supplementary event metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, fbeta_score


def calculate_criticality(
    anomalies: pd.Series,
    normal_idx: pd.Series,
    init_criticality: int = 0,
    max_criticality: int = 1000,
) -> pd.Series:
    anomalies = anomalies.sort_index()
    normal_idx = normal_idx.reindex(anomalies.index).fillna(True).astype(bool)
    crit = np.empty(len(anomalies), dtype=np.int64)
    c = init_criticality
    for i, (is_anom, is_normal) in enumerate(zip(anomalies.to_numpy(), normal_idx.to_numpy())):
        if is_normal and is_anom:
            c += 1
        elif is_normal and not is_anom:
            c = max(0, c - 1)
        c = min(max_criticality, c)
        crit[i] = c
    return pd.Series(crit, index=anomalies.index)


def weighted_earliness_score(
    predicted: pd.Series,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> float:
    pred = predicted.loc[event_start:event_end]
    if pred.empty or not pred.any():
        return 0.0
    duration = (event_end - event_start).total_seconds()
    if duration <= 0:
        return float(pred.mean())
    weights = []
    hits = []
    for ts, flag in pred.items():
        if not flag:
            continue
        frac = (ts - event_start).total_seconds() / duration
        weight = 1.0 if frac <= 0.5 else max(0.0, 1.0 - 2.0 * (frac - 0.5))
        weights.append(weight)
        hits.append(1.0)
    if not weights:
        return 0.0
    return float(np.average(hits, weights=weights))


@dataclass
class CareScoreAccumulator:
    coverage_beta: float = 0.5
    reliability_beta: float = 0.5
    coverage_w: float = 1.0
    accuracy_w: float = 2.0
    earliness_w: float = 1.0
    reliability_w: float = 1.0
    criticality_threshold: int = 72
    records: List[Dict] = field(default_factory=list)

    def evaluate_event(
        self,
        event_id: int,
        event_label: str,
        predicted: pd.Series,
        ground_truth: pd.Series,
        normal_idx: pd.Series,
        event_start: pd.Timestamp,
        event_end: pd.Timestamp,
    ) -> Dict:
        normal_idx = normal_idx.reindex(predicted.index).fillna(True).astype(bool)
        scoring_mask = normal_idx
        y_true = ground_truth.reindex(predicted.index).fillna(False).astype(bool)
        y_pred = predicted.reindex(predicted.index).fillna(False).astype(bool)

        crit = calculate_criticality(y_pred, normal_idx)
        max_criticality = float(crit.max()) if len(crit) else 0.0
        anomaly_detected = max_criticality >= self.criticality_threshold

        if event_label == "anomaly":
            y_eval = y_true[scoring_mask]
            p_eval = y_pred[scoring_mask]
            f_beta = (
                float(fbeta_score(y_eval, p_eval, beta=self.coverage_beta, zero_division=0))
                if y_eval.any() or p_eval.any()
                else 0.0
            )
            ws = weighted_earliness_score(y_pred, event_start, event_end)
            accuracy = np.nan
        else:
            y_eval = (~y_true)[scoring_mask]
            p_eval = (~y_pred)[scoring_mask]
            f_beta = np.nan
            ws = np.nan
            accuracy = float(accuracy_score(y_eval, p_eval)) if len(y_eval) else np.nan

        record = {
            "event_id": event_id,
            "event_label": event_label,
            "f_beta_score": f_beta,
            "weighted_score": ws,
            "accuracy": accuracy,
            "max_criticality": max_criticality,
            "anomaly_detected": anomaly_detected,
            "pr_auc": (
                float(average_precision_score(y_true[scoring_mask], y_pred[scoring_mask].astype(int)))
                if event_label == "anomaly" and y_true[scoring_mask].any()
                else np.nan
            ),
        }
        self.records.append(record)
        return record

    def final_score(self) -> float:
        df = pd.DataFrame(self.records)
        if df.empty:
            return 0.0
        anomaly = df["event_label"] == "anomaly"
        normal = ~anomaly
        if anomaly.sum() == 0 or normal.sum() == 0:
            return 0.0
        if not df["anomaly_detected"].any():
            return 0.0
        avg_accuracy = float(df.loc[normal, "accuracy"].mean())
        if avg_accuracy < 0.5:
            return avg_accuracy
        avg_coverage = float(df.loc[anomaly, "f_beta_score"].mean())
        avg_earliness = float(df.loc[anomaly, "weighted_score"].mean())
        reliability = float(
            fbeta_score(
                df["event_label"] == "anomaly",
                df["anomaly_detected"],
                beta=self.reliability_beta,
                zero_division=0,
            )
        )
        num = (
            self.coverage_w * avg_coverage
            + self.earliness_w * avg_earliness
            + self.reliability_w * reliability
            + self.accuracy_w * avg_accuracy
        )
        den = self.coverage_w + self.earliness_w + self.reliability_w + self.accuracy_w
        return float(num / den)

    def summary(self) -> Dict[str, float]:
        df = pd.DataFrame(self.records)
        if df.empty:
            return {}
        anomaly = df["event_label"] == "anomaly"
        return {
            "care_score": self.final_score(),
            "coverage_f_beta": float(df.loc[anomaly, "f_beta_score"].mean()),
            "earliness_ws": float(df.loc[anomaly, "weighted_score"].mean()),
            "normal_accuracy": float(df.loc[~anomaly, "accuracy"].mean()),
            "event_recall": float(df.loc[anomaly, "anomaly_detected"].mean()) if anomaly.any() else np.nan,
            "false_alarm_rate": float(df.loc[~anomaly, "anomaly_detected"].mean()) if (~anomaly).any() else np.nan,
            "pr_auc": float(df.loc[anomaly, "pr_auc"].mean()),
        }


def bootstrap_ci(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))
