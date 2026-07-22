"""Run the CARE leave-one-turbine-out benchmark."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from care_benchmark.load_care import CareDataset, make_synthetic_care_dataset
from care_benchmark.metrics import CareScoreAccumulator, bootstrap_ci
from care_benchmark.models import FAMILY_SPECS, MODEL_FAMILIES, fit_pool_model, predict_event
from care_benchmark.splits import leave_one_turbine_out_splits
from care_benchmark.windowing import ground_truth_anomaly_mask, normal_mask
from config.load import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_data_root(cfg: dict) -> Path:
    care = cfg.get("care", {})
    root = Path(care.get("data_root", "data/care/CARE_To_Compare"))
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root


def _load_events(cfg: dict):
    root = _resolve_data_root(cfg)
    if not root.is_dir():
        if cfg.get("care", {}).get("allow_synthetic_fallback", True):
            logger.warning("CARE data not found at %s — generating synthetic fallback", root)
            make_synthetic_care_dataset(root)
        else:
            raise FileNotFoundError(
                f"CARE dataset not found at {root}. "
                "Download CARE_To_Compare.zip from https://zenodo.org/records/15846963"
            )
    dataset = CareDataset(root)
    wind_farm = cfg.get("care", {}).get("wind_farm", "A")
    max_events = cfg.get("care", {}).get("max_events")
    events = list(dataset.iter_events(wind_farm=wind_farm))
    if max_events:
        events = events[: int(max_events)]
    return dataset, events


def run_benchmark(cfg: dict) -> pd.DataFrame:
    care_cfg = cfg.get("care", {})
    seed = int(cfg.get("global", {}).get("random_seed", 42))
    np.random.seed(seed)
    window_size = int(care_cfg.get("window_size", 144))
    stride = int(care_cfg.get("stride", 6))
    families = care_cfg.get("model_families", MODEL_FAMILIES)
    n_boot = int(care_cfg.get("bootstrap_samples", 200))

    dataset, events = _load_events(cfg)
    rows = []

    for held_out_asset, train_events, test_events in leave_one_turbine_out_splits(events):
        logger.info("LOTO fold: held-out asset %s (%d test events)", held_out_asset, len(test_events))
        for family_name in families:
            spec = FAMILY_SPECS[family_name]
            fitted = fit_pool_model(train_events, dataset, spec, window_size, stride, seed)
            accumulator = CareScoreAccumulator(
                criticality_threshold=int(care_cfg.get("criticality_threshold", 72))
            )
            event_scores = []
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
                    event_id=event.event_id,
                    event_label=event.event_label,
                    predicted=predicted,
                    ground_truth=gt,
                    normal_idx=nidx,
                    event_start=event.event_start,
                    event_end=event.event_end,
                )
                event_scores.append(accumulator.records[-1].get("f_beta_score", 0.0) or 0.0)
            summary = accumulator.summary()
            lo, hi = bootstrap_ci(np.array(event_scores, dtype=float), n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "held_out_asset": held_out_asset,
                    "model_family": family_name,
                    "n_test_events": len(test_events),
                    **summary,
                    "care_score_ci_low": lo,
                    "care_score_ci_high": hi,
                }
            )

    result = pd.DataFrame(rows)
    out_dir = _REPO_ROOT / care_cfg.get("results_dir", "care_benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "benchmark_table.csv"
    result.to_csv(result_path, index=False)
    summary_path = out_dir / "benchmark_summary.json"
    agg = (
        result.groupby("model_family")
        .agg(
            care_score=("care_score", "mean"),
            event_recall=("event_recall", "mean"),
            false_alarm_rate=("false_alarm_rate", "mean"),
            pr_auc=("pr_auc", "mean"),
            coverage_f_beta=("coverage_f_beta", "mean"),
            earliness_ws=("earliness_ws", "mean"),
            normal_accuracy=("normal_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("care_score", ascending=False)
    )
    summary_path.write_text(agg.to_json(orient="records", indent=2))
    logger.info("Wrote %s and %s", result_path, summary_path)
    print("\n=== CARE Benchmark (mean across LOTO folds) ===")
    print(agg.to_string(index=False))
    return result


def main(config_path=None):
    cfg = load_config(config_path)
    run_benchmark(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE leave-one-turbine-out benchmark")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)
