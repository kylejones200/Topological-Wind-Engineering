# Topological-Wind-Engineering

A software library for **Topological Data Analysis (TDA)** applied to wind energy monitoring, classification, and prediction. It provides reproducible implementations of persistent homology, Mapper, path homology, and zigzag persistence for use with standard SCADA data. The repository serves as supplemental material for associated journal submissions and is intended for researchers and engineers working at the intersection of applied topology and wind energy.

---

## Overview

We reframe wind turbine monitoring as **behavioral pattern recognition**. Instead of univariate thresholds, the methods here analyze the **shape** of multivariate trajectories in the \((u, \omega, P, \beta)\) state space (wind speed, rotor speed, power, pitch). Topological summaries—persistence diagrams, Mapper skeletons, path homology, and persistence images—are used to detect faults, misalignment, wake effects, and regime transitions from operational data.

### Core methodology

| Homology / tool | Role in this library |
|-----------------|----------------------|
| **H₀** | Mode transitions, clustering, connected components. |
| **H₁** | Cyclic behavior, oscillations, hysteresis loops in power–wind space. |
| **Persistence images** | Vectorization of persistence diagrams for deep learning (e.g. CNNs). |
| **Mapper** | Skeletonization of operational manifolds; used for yaw and regime structure. |
| **Path homology** | Directed structure on lead–lag networks; wake and grid coordination. |
| **Zigzag persistence** | Time-varying complexes; precursors to startups, shutdowns, ramps. |

---

## Library modules

| Module | Description | Location in repo |
|--------|-------------|------------------|
| **Turbulence CNN** | Classifies local turbulence intensity (high vs low) using persistence images and convolutional neural networks. | `the-texture-of-wind-.../code/turbulence_classification.py` |
| **Yaw Mapper** | Detects nacelle yaw misalignment by skeletonizing operational manifolds with the Mapper algorithm. | `when-turbines-look-away-.../code/yaw_mapper.py` |
| **Fault monitor** | Physics-informed anomaly detection for sub-optimal performance (TDA + PCA baselines; 100% recall configurations in associated content). | `when-topology-detects-turbine-faults-.../code/` (e.g. `turbine_tda_enhanced.py`, `turbine_tda_anomaly.py`) |
| **Farm coordination** | Analyzes wake propagation and grid feedback using persistent path homology on lead–lag networks. | `the-dance-of-turbines-.../code/farm_coordination.py`, `40_farm_coordination_path_homology_blog_code.py` |
| **Regime prognostics** | Predicts imminent startups, shutdowns, and ramp events using zigzag persistence and topological precursors. | `seeing-around-corners-.../code/regime_transition.py` |
| **Wake topology** | Quantifies wake-induced hysteresis in the power–windspeed phase space and classifies wake vs free-stream operation. | `the-hidden-structure-of-wakes/code/turbine_wake_detection.py` |

Each module is self-contained and includes or references a simulation environment; several use **NREL Wind Toolkit** data (see [Data](#data)).

---

## Installation

This project uses **[uv](https://docs.astral.sh/uv/)** for package management (recommended). You can also install with pip and `requirements.txt`.

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy, scikit-learn, **ripser**, **persim**, Matplotlib, **torch** (for turbulence CNN). See `pyproject.toml` or `requirements.txt`.

### Install with uv (recommended)

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
git clone https://github.com/kylejones200/Topological-Wind-Engineering.git
cd Topological-Wind-Engineering
uv sync
```

This creates a virtual environment, resolves dependencies from `pyproject.toml`, and installs into it. To generate/update the lockfile: `uv lock`. To install with optional NREL/OpenOA support: `uv sync --extra nrel`.

### Install with pip

```bash
pip install -r requirements.txt
```

### Minimal install (no torch, no NREL)

Edit `pyproject.toml` to drop `torch` and `persim` if you only need core TDA; or with pip:

```bash
pip install numpy pandas scipy scikit-learn ripser matplotlib
```

---

## Configuration

All runnable scripts are driven by a **config file** (no magic numbers in code). Defaults live in `config/default.yaml`. Override with `--config path/to.yaml` or the `CONFIG_PATH` environment variable. **Run from repo root** so the config loader finds `config/default.yaml`. Config sections: `global`, `nrel`, `simulation`, and module-specific keys (`farm_coordination`, `wake_detection`, `yaw_mapper`, `regime_tda`). Set `NREL_API_KEY` in the environment to override the API key.

---

## Usage

- **Run a single module:** from the repo root, run the script; it loads `config/default.yaml` (or use `--config`). Example:  
  `python "when-turbines-look-away-using-mapper-to-detect-yaw-misalignment-from-operational-patterns/code/yaw_mapper.py"`
- **Reproduce results:** each project directory contains `code/` and often `content/` or `status/` with write-ups and logs. Run the main script in that project’s `code/` directory; paths are set up relative to that folder.
- **Use as a library:** import functions from the module scripts after adding the relevant `code/` directory to `PYTHONPATH`, or copy the needed module into your own project.

---

## Data

Several modules use or simulate data compatible with the **NREL Wind Toolkit** (WTK). You can obtain API access at [NREL Developer Network](https://developer.nrel.gov/). Scripts that call the WTK use parameters (e.g. `lat`, `lon`, `years`) at the top of the file or in `main()`; adjust these for your site. Some scripts include fallback synthetic or sampled data when the API is unavailable.

---

## Repository layout

```
Topological-Wind-Engineering/
├── README.md
├── pyproject.toml    # Project metadata and dependencies (uv/pip)
├── uv.lock           # Lockfile for reproducible installs (uv lock)
├── requirements.txt  # Fallback for pip install
├── config/
│   ├── default.yaml  # Default run configuration (no magic numbers in code)
│   └── load.py       # Config loader used by all scripts
├── the-texture-of-wind-machine-learning-on-topological-images-to-classify-turbulence/
│   └── code/          # Turbulence classification (persistence images + CNN)
├── when-turbines-look-away-using-mapper-to-detect-yaw-misalignment-from-operational-patterns/
│   └── code/          # Yaw Mapper
├── when-topology-detects-turbine-faults-using-shape-to-monitor-wind-farm-performance/
│   └── code/          # Fault / anomaly detection (TDA, PCA, synthetic & NREL)
├── the-dance-of-turbines-detecting-coordinated-responses-in-wind-farms-using-path-homology/
│   └── code/          # Farm coordination (path homology)
├── seeing-around-corners-predicting-wind-turbine-regime-changes-through-topological-precursors/
│   └── code/          # Regime prognostics (zigzag persistence)
└── the-hidden-structure-of-wakes/
    └── code/          # Wake topology (power–windspeed hysteresis)
```

---

## Reproducibility

- Scripts use fixed seeds (e.g. `np.random.seed(42)`) where applicable.
- Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` (use `uv sync` for a reproducible environment). A `requirements.txt` is provided for pip users.
- For exact reproduction of figures or tables in the associated papers/posts, use the same Python version and run from the project subdirectory indicated in the corresponding content.

---

## Citation

If you use this library in academic work, please cite the repository and the relevant methodology papers (persistent homology, Mapper, path homology, zigzag persistence) as appropriate. Citation format for the software:

```bibtex
@software{topological_wind_engineering,
  author = {Jones, Kyle},
  title = {Topological-Wind-Engineering: TDA for wind energy monitoring and prediction},
  url = {https://github.com/kylejones200/Topological-Wind-Engineering},
  year = {2025}
}
```

---

## License

See the [LICENSE](LICENSE) file in this repository.

---

## Contributing

Contributions that improve reproducibility, extend the methodology, or add tests are welcome. For substantial changes, open an issue first to align with the research goals of the project.
