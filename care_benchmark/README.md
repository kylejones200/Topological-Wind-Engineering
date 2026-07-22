# CARE Benchmark (Phase 1)

Leave-one-turbine-out evaluation on the [CARE to Compare](https://zenodo.org/records/15846963) SCADA dataset.

## Setup

```bash
python care_benchmark/download.py   # optional, ~5.5 GB
```

Without the real dataset, the runner generates a minimal synthetic fallback for development.

## Run

```bash
# Full protocol (Farm A, six model families)
python care_benchmark/run_benchmark.py

# Quick dev run
python care_benchmark/run_benchmark.py --config config/care_quick.yaml
```

## Model families

| Family | Features |
|--------|----------|
| `power_curve_residual` | Power-curve residual statistics |
| `statistical` | Window mean/std/min/max/skew |
| `spectral` | FFT band energies |
| `isolation_forest` | Statistical features + Isolation Forest |
| `tda_only` | Sliding-window persistence (ripser) |
| `tda_plus_conventional` | Statistical + TDA |

## Evaluation

- **Split:** leave-one-turbine-out by `asset_id`
- **Leakage control:** imputer/scaler/PCA/IF fit on train-split rows from non-held-out turbines only
- **Metrics:** CARE score (Coverage, Accuracy, Reliability, Earliness), event recall, false-alarm rate, PR-AUC
- **Output:** `care_benchmark/results/benchmark_table.csv`

## Decision gate

Compare `tda_only` and `tda_plus_conventional` against conventional baselines on **incremental** CARE score and event recall. Continue the paper if TDA wins with bootstrap CIs; publish a null-result benchmark otherwise.
