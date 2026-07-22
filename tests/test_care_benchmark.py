"""Tests for CARE benchmark components."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from care_benchmark.load_care import CareDataset, make_synthetic_care_dataset
from care_benchmark.metrics import CareScoreAccumulator, calculate_criticality
from care_benchmark.models import FAMILY_SPECS, fit_pool_model, predict_event
from care_benchmark.splits import leave_one_turbine_out_splits
from care_benchmark.windowing import extract_window_feature_matrix, ground_truth_anomaly_mask, normal_mask


@pytest.fixture
def synthetic_root(tmp_path):
    return make_synthetic_care_dataset(tmp_path / "care")


def test_loader_reads_events(synthetic_root):
    dataset = CareDataset(synthetic_root)
    events = list(dataset.iter_events(wind_farm="A"))
    assert len(events) == 6
    assert events[0].train.shape[0] > 0
    assert events[0].test.shape[0] > 0


def test_loto_splits_cover_all_assets(synthetic_root):
    events = list(CareDataset(synthetic_root).iter_events(wind_farm="A"))
    folds = list(leave_one_turbine_out_splits(events))
    assert len(folds) == 3


def test_criticality_increases_with_anomalies():
    idx = pd.date_range("2020-01-01", periods=10, freq="10min")
    anomalies = pd.Series([False, True, True, False, False, True, True, True, False, False], index=idx)
    normal_idx = pd.Series(True, index=idx)
    crit = calculate_criticality(anomalies, normal_idx)
    assert crit.max() >= 2


def test_window_features_shape():
    values = np.random.randn(200, 3)
    X, ends = extract_window_feature_matrix(values, window_size=36, stride=6, feature_sets=["statistical"])
    assert X.shape[0] > 0
    assert X.shape[1] == 15


def test_benchmark_fold_runs(synthetic_root):
    dataset = CareDataset(synthetic_root)
    events = list(dataset.iter_events(wind_farm="A"))
    held_out, train_events, test_events = next(iter(leave_one_turbine_out_splits(events)))
    spec = FAMILY_SPECS["statistical"]
    fitted = fit_pool_model(train_events, dataset, spec, window_size=36, stride=6, seed=0)
    accumulator = CareScoreAccumulator()
    for event in test_events:
        test_df = event.test.set_index("time_stamp")
        predicted = predict_event(fitted, test_df)
        nidx = normal_mask(test_df["status_type_id"])
        gt = ground_truth_anomaly_mask(
            test_df.index,
            nidx,
            event.event_label,
            event.event_start,
            event.event_end,
        )
        accumulator.evaluate_event(
            event.event_id,
            event.event_label,
            predicted,
            gt,
            nidx,
            event.event_start,
            event.event_end,
        )
    score = accumulator.final_score()
    assert 0.0 <= score <= 1.0
