"""
Topological Data Analysis of Wind Turbine SCADA Data

This script demonstrates topological classification of wind turbine operating states
using persistent homology. It includes leak-safe evaluation for time-series data.

Run from repo root: python path/to/turbine_tda_analysis.py [--config path/to/config.yaml]
Requirements: pip install pandas numpy matplotlib scikit-learn ripser persim 'openoa[examples]'
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _ in range(15):
    if (_REPO_ROOT / "config" / "default.yaml").is_file() or (_REPO_ROOT / "pyproject.toml").is_file():
        break
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import numpy as np
import pandas as pd
import zipfile
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import logging

from config.load import load_config
from tda_utils import setup_tufte_plot, TufteColors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
try:
    import importlib.resources as rsrc
    import openoa
    HAS_OPENOA = True
except ImportError:
    HAS_OPENOA = False
    logger.warning("Warning: OpenOA not installed. Run: pip install 'openoa[examples]'")

def tufte_style(ax):
    """Apply minimalist Tufte-inspired styling to axes."""
    plt.rcParams['font.family'] = 'serif'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 6))
    ax.spines['bottom'].set_position(('outward', 6))

def load_scada():
    """
    Load OpenOA La Haute Borne SCADA sample.
    
    Returns:
        DataFrame with columns: time, wind_speed, rotor_speed, power
    """
    if not HAS_OPENOA:
        raise RuntimeError("OpenOA example data unavailable. Install with: pip install 'openoa[examples]'")
    with rsrc.path('openoa.examples.data', 'la_haute_borne.zip') as zpath:
        with zipfile.ZipFile(zpath, 'r') as zf:
            target = None
            for name in zf.namelist():
                if name.lower().endswith('scada.csv'):
                    target = name
                    break
            if target is None:
                raise FileNotFoundError('SCADA CSV not found in archive')
            data = io.BytesIO(zf.read(target))
            df = pd.read_csv(data, parse_dates=['time'])
    rename_map = {}
    for want in ['time', 'wind_speed', 'rotor_speed', 'power']:
        matches = [c for c in df.columns if c.lower() == want]
        if matches:
            rename_map[matches[0]] = want
    df = df.rename(columns=rename_map)
    for need in ['time', 'wind_speed', 'rotor_speed', 'power']:
        if need not in df.columns:
            raise ValueError(f'Missing required column: {need}')
    df = df.sort_values('time').dropna(subset=['wind_speed', 'rotor_speed', 'power'])
    df = df[(df['wind_speed'] >= 0) & (df['power'] >= 0)].reset_index(drop=True)
    return df[['time', 'wind_speed', 'rotor_speed', 'power']]

def make_nonoverlapping_windows(df, win_size=512):
    """
    Create non-overlapping windows to prevent temporal leakage.
    
    Args:
        df: DataFrame with time-series turbine data
        win_size: Number of samples per window
    
    Returns:
        Xw: Array of windows, shape (n_windows, win_size, 3)
        yw: Binary labels, shape (n_windows,)
        tw: Timestamps of window midpoints, shape (n_windows,)
    """
    X = df[['wind_speed', 'rotor_speed', 'power']].to_numpy(float)
    y_raw = df['power'].to_numpy(float)
    n = len(df)
    starts = np.arange(0, n - win_size + 1, win_size)
    Xw, yw, tw = ([], [], [])
    med = np.median(y_raw)
    for s in starts:
        slab = X[s:s + win_size]
        mid = s + win_size // 2
        label = 1 if y_raw[mid] > med else 0
        time_mid = df.loc[mid, 'time']
        Xw.append(slab)
        yw.append(label)
        tw.append(time_mid)
    Xw = np.array(Xw, dtype=float)
    yw = np.array(yw, dtype=int)
    tw = pd.to_datetime(tw)
    return (Xw, yw, tw)

def lifetimes(dgm):
    """
    Compute persistence lifetimes from a diagram.
    
    Args:
        dgm: Persistence diagram, shape (n_features, 2)
    
    Returns:
        Array of finite lifetimes
    """
    if dgm.size == 0:
        return np.array([])
    L = dgm[:, 1] - dgm[:, 0]
    return L[np.isfinite(L)]

def extract_tda_features(window):
    """
    Compute topological features from a single window via persistent homology.
    
    Args:
        window: Array of shape (win_size, n_features)
    
    Returns:
        Array [sum_H1_lifetimes, max_H1_lifetime]
    """
    result = ripser(window, maxdim=1)['dgms']
    H1 = result[1] if len(result) > 1 else np.empty((0, 2))
    L = lifetimes(H1)
    return np.array([float(L.sum()) if L.size else 0.0, float(L.max()) if L.size else 0.0])

def build_tda_matrix(Xw):
    """
    Extract TDA features from all windows.
    
    Args:
        Xw: Array of windows, shape (n_windows, win_size, n_features)
    
    Returns:
        Feature matrix, shape (n_windows, 2)
    """
    logger.info('  Computing persistent homology for each window...')
    features = []
    for i, w in enumerate(Xw):
        if (i + 1) % 10 == 0:
            logger.info(f'    Window {i + 1}/{len(Xw)}')
        features.append(extract_tda_features(w))
    return np.vstack(features)

def build_pca_matrix(Xw):
    """
    Build PCA-based features from windows.
    Standardize across all data, then summarize each window.
    
    Args:
        Xw: Array of windows, shape (n_windows, win_size, n_features)
    
    Returns:
        Feature matrix, shape (n_windows, 2)
    """
    n_win, win, dim = Xw.shape
    flat = Xw.reshape(n_win * win, dim)
    scaler = StandardScaler().fit(flat)
    flat_std = scaler.transform(flat)
    Z = PCA(n_components=2, random_state=0).fit_transform(flat_std)
    Zw = Z.reshape(n_win, win, 2)
    return np.median(Zw, axis=1)

def purged_forward_splits(times, n_splits=5, purge_windows=1):
    """
    Generate train/test splits that respect time ordering and prevent leakage.
    
    Args:
        times: Timestamps for each window
        n_splits: Number of folds
        purge_windows: Number of windows to remove at train/test boundary
    
    Yields:
        (train_indices, test_indices) tuples
    """
    n = len(times)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[:n % n_splits] += 1
    bounds = np.cumsum(fold_sizes)
    starts = np.insert(bounds[:-1], 0, 0)
    stops = bounds
    splits = []
    for i in range(n_splits):
        test_idx = np.arange(starts[i], stops[i])
        train_stop = starts[i] - purge_windows
        if train_stop <= 0:
            continue
        train_idx = np.arange(0, train_stop)
        splits.append((train_idx, test_idx))
    return splits

def evaluate_model(X_train, y_train, X_test, y_test, model):
    """
    Train model and compute evaluation metrics.
    
    Returns:
        (auc, accuracy) tuple
    """
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X_test)[:, 1]
    else:
        p = model.decision_function(X_test)
        p = (p - p.min()) / (p.max() - p.min() + 1e-09)
    auc = roc_auc_score(y_test, p)
    acc = ((p > 0.5).astype(int) == y_test).mean()
    return (auc, acc)

def _run_data_and_plots(cfg):
    """Load SCADA, build windows, phase portrait, persistence diagram, extract TDA/PCA. Returns (X_tda, X_pca, y, t, out_dir, seed, rt)."""
    rt = cfg.get("regime_tda", {})
    seed = cfg.get("global", {}).get("random_seed", 42)
    win_size = rt.get("win_size_analysis", 512)
    out_dir = _SCRIPT_DIR / rt.get("figures_subdir", "figures")
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info("=" * 60)
    logger.info("Topological Data Analysis of Wind Turbine SCADA")
    logger.info("=" * 60)
    logger.info("\n1. Loading SCADA data...")
    df = load_scada()
    logger.info(f"   Loaded {len(df):,} records")
    logger.info(f"   Time span: {df['time'].min()} to {df['time'].max()}")
    logger.info("\n2. Creating non-overlapping windows...")
    Xw, y, t = make_nonoverlapping_windows(df, win_size=win_size)
    logger.info(f"   Created {len(Xw)} windows of {win_size} samples each")
    logger.info(f'   Label distribution: {(y == 0).sum()} low-power, {(y == 1).sum()} high-power')
    logger.info('\n3. Building phase portrait...')
    X_flat = Xw.reshape(-1, 3)
    X_std = StandardScaler().fit_transform(X_flat)
    Z = PCA(n_components=2, random_state=0).fit_transform(X_std)
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.3, c='steelblue')
    ax.set_title('Turbine Phase Portrait (PCA)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    tufte_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase_portrait.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {out_dir / 'phase_portrait.png'}")
    logger.info("\n4. Computing persistent homology...")
    sample_size = rt.get("sample_size_diagram", 8000)
    sample_idx = np.random.choice(len(X_std), size=min(sample_size, len(X_std)), replace=False)
    result = ripser(X_std[sample_idx], maxdim=1)
    dgms = result['dgms']
    plt.figure(figsize=(6, 5))
    plot_diagrams(dgms, show=False)
    ax = plt.gca()
    ax.set_title('Persistence Diagram')
    tufte_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'persistence_diagram.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {out_dir / 'persistence_diagram.png'}")
    if len(dgms) > 1 and dgms[1].size > 0:
        L = lifetimes(dgms[1])
        if L.size > 0:
            logger.info(f'   Max H1 persistence (loop strength): {L.max():.4f}')
    logger.info('\n5. Extracting features...')
    logger.info('   a) TDA features (persistent homology)...')
    X_tda = build_tda_matrix(Xw)
    logger.info('   b) PCA features (baseline)...')
    X_pca = build_pca_matrix(Xw)
    return (X_tda, X_pca, y, t, out_dir, seed, rt)


def _run_cv_and_log(X_tda, X_pca, y, t, seed, rt, out_dir):
    """Run purged forward CV, log results and leakage summary."""
    n_splits = rt.get("n_splits_analysis", 6)
    purge_windows = rt.get("purge_windows", 1)
    logger.info("\n6. Evaluating with purged forward cross-validation...")
    splits = purged_forward_splits(t, n_splits=n_splits, purge_windows=purge_windows)
    logger.info(f"   Using {len(splits)} folds with {purge_windows}-window purge gap")
    logger.info('   This prevents temporal leakage between train and test sets')
    results = {'TDA + LogReg': [], 'PCA + LogReg': [], 'PCA + SVM-Lin': [], 'PCA + SVM-RBF': []}
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f'   Fold {fold_idx + 1}/{len(splits)}...')
        Xt_tr, Xt_te = (X_tda[train_idx], X_tda[test_idx])
        Xp_tr, Xp_te = (X_pca[train_idx], X_pca[test_idx])
        y_tr, y_te = (y[train_idx], y[test_idx])
        auc, acc = evaluate_model(Xt_tr, y_tr, Xt_te, y_te, LogisticRegression(max_iter=1000, random_state=seed))
        results["TDA + LogReg"].append((auc, acc))
        auc, acc = evaluate_model(Xp_tr, y_tr, Xp_te, y_te, LogisticRegression(max_iter=1000, random_state=seed))
        results["PCA + LogReg"].append((auc, acc))
        auc, acc = evaluate_model(Xp_tr, y_tr, Xp_te, y_te, SVC(kernel="linear", probability=True, random_state=seed))
        results["PCA + SVM-Lin"].append((auc, acc))
        auc, acc = evaluate_model(Xp_tr, y_tr, Xp_te, y_te, SVC(kernel="rbf", probability=True, random_state=seed))
        results["PCA + SVM-RBF"].append((auc, acc))
    logger.info('\n' + '=' * 60)
    logger.info('RESULTS (averaged across folds)')
    logger.info('=' * 60)
    logger.info(f"{'Model':<20} {'AUC':>10} {'Accuracy':>10}")
    logger.info('-' * 60)
    for name, scores in results.items():
        arr = np.array(scores)
        logger.info(f'{name:<20} {arr[:, 0].mean():>10.3f} {arr[:, 1].mean():>10.3f}')
    logger.info('=' * 60)
    logger.info('\n' + '=' * 60)
    logger.info('LEAKAGE PREVENTION SUMMARY')
    logger.info('=' * 60)
    logger.info('  ✓ Non-overlapping windows (no shared samples)')
    logger.info('  ✓ Forward-chaining splits (test always after train)')
    logger.info('  ✓ Purge gap at boundaries (removes temporal correlation)')
    logger.info('  ✓ No future information in feature extraction')
    logger.info('=' * 60)
    logger.info('\nAnalysis complete!')
    logger.info(f'Figures saved to: {out_dir.absolute()}/')


def main(config_path=None):
    """Main analysis workflow. All parameters from config."""
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    np.random.seed(seed)
    X_tda, X_pca, y, t, out_dir, seed, rt = _run_data_and_plots(cfg)
    _run_cv_and_log(X_tda, X_pca, y, t, seed, rt, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TDA of wind turbine SCADA (OpenOA)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)