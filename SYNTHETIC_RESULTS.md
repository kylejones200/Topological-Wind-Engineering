# Synthetic Results Disclaimer

**All experiments in this repository, unless explicitly stated otherwise, use synthetic or simulated data.**

## What the data actually are

| Source | Used for | Production validity |
|--------|----------|---------------------|
| NREL Wind Toolkit (atmospheric) | Real wind speed / temperature time series | Atmospheric inputs only |
| Simulated turbine response | Power, rotor speed, pitch, faults, wakes, yaw | Not validated SCADA |
| Injected labels | Faults, wakes, turbulence classes, regime transitions | Derived from simulation rules |

Real atmospheric data from NREL does **not** make the downstream turbine labels or performance claims real. Faults, wakes, yaw errors, and regime transitions are imposed by simulation code. Classifiers that receive features derived from those injections are measuring recovery of the simulation, not field performance.

## Known methodological limitations

The archived prototype experiments have documented flaws. Do not cite their metrics as evidence of deployable performance.

### Fault detection (`when-topology-detects-turbine-faults-...`)

- **Severe class imbalance:** 102 evaluation windows, 98 labeled anomalous and 4 normal.
- **Trivial baseline:** An always-anomaly classifier matches reported accuracy (~95%) and F1 (~0.974).
- **Cross-validation leakage:** Scaler and PCA are fit on the full dataset before fold splits.
- **Label leakage:** Labels derive from the same synthetic fault process that alters power features.

### Regime transitions (`seeing-around-corners-...`)

- **Not zigzag persistence:** Code uses ordinary `ripser` persistence on sliding subwindows plus inter-diagram distances.
- **Time-unit bug (fixed in code):** Parameters historically named `*_minutes` were row indices on **hourly** data.

### Farm coordination (`the-dance-of-turbines-...`)

- **Not path homology:** Features are NetworkX graph statistics (cycles, density, reciprocity, path length).
- **No field validation:** Claims about commercial farm trials or 78% field accuracy are unsupported.

### Yaw misalignment (`when-turbines-look-away-...`)

- **Simulated misalignment** embedded in features available to the classifier.
- **Cross-turbine transfer claims** are not backed by held-out turbine experiments in code.

### Wake detection (`the-hidden-structure-of-wakes`)

- **Simulated wake hysteresis** imposed on power curves.
- Does not meet the simulation standard of recent wake-detection literature on SCADA / blade loads.

### Turbulence classification (`the-texture-of-wind-...`)

- Synthetic turbulence classes; archived in current form.

## Unsupported claims removed from active use

The following must not appear in future publications or public materials without new evidence:

- "First application" of persistent homology to wind-turbine monitoring (prior work exists, e.g. PHM Society 2024).
- Commercial / field-trial performance numbers (78%, 79%, etc.).
- Production-ready fault detection (100% recall, 95% precision).
- Lead-time guarantees (10–20 minutes) without correct temporal resolution and validation.
- Cross-turbine transfer without leave-one-turbine-out evaluation.

## Current research direction

The active academic question is:

> **Do topological features provide incremental predictive value over conventional SCADA features under proper turbine- and farm-level holdout?**

The planned evaluation uses the [CARE benchmark](https://zenodo.org/records/15846963) with leave-one-turbine-out splits, event-level metrics, and bootstrap confidence intervals. See [MANUSCRIPT_STATUS.md](MANUSCRIPT_STATUS.md).

## Citing results from this repo

- Cite **methods and code structure**, not archived metric tables.
- If you must reference archived numbers, label them explicitly as **synthetic, unaudited, and superseded**.
- Wait for the CARE benchmark before citing any performance claim.
