# Manuscript Status

**Status: Frozen (July 2025)**

The six manuscript branches in this repository are **archived prototypes**. They are retained for code reuse and hypothesis development. They are **not** submission-ready and must not be distributed as completed research.

## Module status

| Module | Directory | Status | Notes |
|--------|-----------|--------|-------|
| Fault detection | `when-topology-detects-turbine-faults-...` | **Rebuild** | Primary candidate for CARE real-data paper |
| Regime transitions | `seeing-around-corners-...` | **Hold** | Requires true zigzag implementation + correct time base |
| Farm coordination | `the-dance-of-turbines-...` | **Hold** | Rename or replace with actual path homology |
| Yaw misalignment | `when-turbines-look-away-...` | **Hold** | Needs real SCADA comparison (e.g. OpenOA baseline) |
| Wake detection | `the-hidden-structure-of-wakes` | **Hold** | Needs real labels or physical simulator |
| Turbulence CNN | `the-texture-of-wind-...` | **Archive** | Synthetic classification only |

## What is frozen

- All `content/*.md` blog posts and IEEE drafts in module directories
- LinkedIn / social copy derived from prototype metrics
- Figures and animations generated from synthetic pipelines

Each frozen document carries a banner pointing to [SYNTHETIC_RESULTS.md](SYNTHETIC_RESULTS.md).

## Active work

1. **Phase 0 (hygiene):** disclaimers, `tda_utils`, LICENSE, tests — complete
2. **Phase 1 (kill test):** CARE benchmark with six model families and leave-one-turbine-out evaluation — **scaffold ready** (`care_benchmark/`)
3. **Phase 2 (decision):** continue if TDA shows incremental value; publish null result otherwise

## Target paper (if validated)

**Do Topological Features Improve Wind-Turbine Fault Detection? A Multi-Farm Event-Level Benchmark**

- Dataset: [CARE SCADA benchmark](https://zenodo.org/records/15846963)
- Comparison: power-curve residuals, statistical windows, spectral features, anomaly models, TDA-only, TDA+conventional
- Evaluation: event recall, false alarms per turbine-day, lead time, PR-AUC, CARE score, bootstrap CIs
- Holdout: turbine-level and farm-level separation; no test leakage into scaling or feature selection

## Do not submit

Until the CARE benchmark completes, do not submit any of the archived manuscripts or cite their performance tables as empirical findings.
