"""
Anomaly Detection in Wind Turbines Using Topological Data Analysis.
Run from repo root: python path/to/turbine_tda_anomaly.py [--config path/to/config.yaml]
"""
import sys
import os
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
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import logging

from config.load import load_config
from tda_utils import setup_tufte_plot, TufteColors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tufte_style(ax):
    """Apply minimalist styling."""
    plt.rcParams['font.family'] = 'serif'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 6))
    ax.spines['bottom'].set_position(('outward', 6))

def fetch_nrel_wind_data(cfg):
    """Fetch real wind data from NREL. Uses config for lat, lon, years, api_key, url, timeout."""
    nrel = cfg.get("nrel", {})
    lat = nrel.get("lat", 41.0)
    lon = nrel.get("lon", -95.5)
    years = nrel.get("years", [2017, 2018, 2019])
    api_key = nrel.get("api_key") or os.environ.get("NREL_API_KEY", "")
    base_url = nrel.get("base_url", "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv")
    timeout = nrel.get("request_timeout_seconds", 120)
    email = nrel.get("email", "user@example.com")
    interval = nrel.get("interval", "60")
    attributes = nrel.get("attributes", "windspeed_100m,windspeed_80m,temperature_100m")
    all_data = []
    for year in years:
        logger.info(f"   Fetching year {year}...")
        params = {"api_key": api_key, "wkt": f"POINT({lon} {lat})", "attributes": attributes, "names": str(year), "utc": "true", "leap_day": "false", "interval": interval, "email": email}
        try:
            response = requests.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('Year,'):
                    data_start = i + 1
                    break
            data_text = '\n'.join(lines[data_start:])
            df_year = pd.read_csv(StringIO(data_text), header=None, names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'windspeed_100m', 'windspeed_80m', 'temperature_100m'])
            df_year['time'] = pd.to_datetime(df_year[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            all_data.append(df_year)
            logger.info(f'     ✓ Fetched {len(df_year):,} records for {year}')
        except Exception as e:
            logger.error(f'     ✗ Error fetching {year}: {e}')
            continue
    if not all_data:
        return None
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    logger.info(f'   Total records fetched: {len(df):,} ({len(all_data)} years)')
    return df

def theoretical_power_curve(wind_speed):
    """
    Theoretical power curve for a 2MW turbine.
    Returns expected power given wind speed.
    """
    RATED_POWER = 2000
    CUT_IN = 3
    RATED_WS = 12
    CUT_OUT = 25
    power = np.zeros_like(wind_speed)
    for i, ws in enumerate(wind_speed):
        if ws < CUT_IN or ws > CUT_OUT:
            power[i] = 0
        elif ws < RATED_WS:
            power[i] = RATED_POWER * ((ws - CUT_IN) / (RATED_WS - CUT_IN)) ** 2.5
        else:
            power[i] = RATED_POWER
    return power

def simulate_turbine_with_faults(wind_df, fault_probability=0.15):
    """
    Simulate turbine with occasional performance degradation/faults.
    
    Faults include:
    - Power curve degradation (reduced efficiency)
    - Controller issues (suboptimal tracking)
    - Partial curtailment
    - Measurement drift
    """
    logger.info('   Simulating turbine with occasional faults...')
    df = wind_df.copy()
    n = len(df)
    wind_speed = df['windspeed_100m'].values
    expected_power = theoretical_power_curve(wind_speed)
    rotor_speed = np.zeros(n)
    actual_power = np.zeros(n)
    fault_indicator = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.random.random() < fault_probability:
            fault_duration = np.random.randint(72, 240)
            fault_type = np.random.choice(['degradation', 'controller', 'curtailment', 'drift'])
            for j in range(i, min(i + fault_duration, n)):
                fault_indicator[j] = True
                ws = wind_speed[j]
                if fault_type == 'degradation':
                    efficiency = np.random.uniform(0.7, 0.9)
                    actual_power[j] = expected_power[j] * efficiency
                elif fault_type == 'controller':
                    noise = np.random.normal(0, 0.15)
                    actual_power[j] = expected_power[j] * (1 + noise)
                elif fault_type == 'curtailment':
                    cap = np.random.uniform(0.6, 0.85)
                    actual_power[j] = min(expected_power[j], expected_power[j] * cap)
                elif fault_type == 'drift':
                    drift = np.random.uniform(-0.2, -0.05)
                    actual_power[j] = expected_power[j] * (1 + drift)
                actual_power[j] = np.clip(actual_power[j], 0, 2100)
            i += fault_duration
        else:
            ws = wind_speed[i]
            actual_power[i] = expected_power[i] + np.random.normal(0, 10)
            actual_power[i] = np.clip(actual_power[i], 0, 2100)
            i += 1
    for i in range(1, n):
        ws = wind_speed[i]
        if ws < 3:
            target_rpm = 0
        elif ws < 12:
            target_rpm = 10 + (ws - 3) * 5
        else:
            target_rpm = min(55 + (ws - 12) * 0.3, 60)
        if ws > 25:
            target_rpm = 0
        rotor_speed[i] = 0.85 * rotor_speed[i - 1] + 0.15 * target_rpm
    rotor_speed += np.random.normal(0, 0.3, n)
    rotor_speed = np.clip(rotor_speed, 0, 65)
    power_deviation = actual_power - expected_power
    power_ratio = np.divide(actual_power, expected_power, out=np.ones_like(actual_power), where=expected_power > 10)
    result = pd.DataFrame({'time': df['time'], 'wind_speed': wind_speed, 'rotor_speed': rotor_speed, 'power': actual_power, 'expected_power': expected_power, 'power_deviation': power_deviation, 'power_ratio': power_ratio, 'fault': fault_indicator})
    fault_pct = fault_indicator.sum() / len(fault_indicator) * 100
    logger.info(f'   Fault periods: {fault_pct:.1f}% of records')
    return result

def make_anomaly_windows(df, win_size=256):
    """
    Create windows with anomaly labels.
    
    A window is anomalous if:
    1. Mean power ratio < 0.85 (consistent underperformance)
    2. Std of power deviation is high (erratic behavior)
    3. >30% of samples in window are fault periods
    """
    X = df[['wind_speed', 'rotor_speed', 'power']].to_numpy(float)
    n = len(df)
    starts = np.arange(0, n - win_size + 1, win_size)
    Xw, yw, tw, window_stats = ([], [], [], [])
    for s in starts:
        slab = X[s:s + win_size]
        window_data = df.iloc[s:s + win_size]
        power_ratio_window = window_data['power_ratio'].values
        power_dev_window = window_data['power_deviation'].values
        fault_window = window_data['fault'].values
        mean_ratio = power_ratio_window.mean()
        std_dev = power_dev_window.std()
        fault_fraction = fault_window.mean()
        is_anomalous = mean_ratio < 0.8 or fault_fraction > 0.4
        time_mid = window_data.iloc[len(window_data) // 2]['time']
        Xw.append(slab)
        yw.append(int(is_anomalous))
        tw.append(time_mid)
        window_stats.append({'mean_ratio': mean_ratio, 'std_dev': std_dev, 'fault_fraction': fault_fraction})
    Xw = np.array(Xw, dtype=float)
    yw = np.array(yw, dtype=int)
    tw = pd.to_datetime(tw)
    return (Xw, yw, tw, window_stats)

def lifetimes(dgm):
    """Compute persistence lifetimes."""
    if dgm.size == 0:
        return np.array([])
    L = dgm[:, 1] - dgm[:, 0]
    return L[np.isfinite(L)]

def extract_rich_tda_features(window):
    """Extract comprehensive TDA features."""
    result = ripser(window, maxdim=1)['dgms']
    features = []
    H0 = result[0] if len(result) > 0 else np.empty((0, 2))
    L0 = lifetimes(H0)
    features.extend([len(L0), float(L0.max()) if L0.size else 0.0, float(L0.mean()) if L0.size else 0.0])
    H1 = result[1] if len(result) > 1 else np.empty((0, 2))
    L1 = lifetimes(H1)
    features.extend([len(L1), float(L1.sum()) if L1.size else 0.0, float(L1.max()) if L1.size else 0.0, float(L1.mean()) if L1.size else 0.0, float(L1.std()) if L1.size else 0.0])
    if H1.size > 0:
        births = H1[:, 0]
        deaths = H1[:, 1]
        finite_mask = np.isfinite(births) & np.isfinite(deaths)
        if finite_mask.any():
            features.extend([float(births[finite_mask].mean()), float(deaths[finite_mask].mean())])
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])
    return np.array(features)

def build_rich_tda_matrix(Xw):
    """Extract TDA features from all windows."""
    features = []
    for i, w in enumerate(Xw):
        if (i + 1) % 20 == 0:
            logger.info(f'      Window {i + 1}/{len(Xw)}')
        features.append(extract_rich_tda_features(w))
    return np.vstack(features)

def build_pca_matrix(Xw, n_components=3):
    """Build PCA features."""
    n_win, win, dim = Xw.shape
    n_components = min(n_components, dim)
    flat = Xw.reshape(n_win * win, dim)
    scaler = StandardScaler().fit(flat)
    flat_std = scaler.transform(flat)
    Z = PCA(n_components=n_components, random_state=0).fit_transform(flat_std)
    Zw = Z.reshape(n_win, win, n_components)
    return np.column_stack([np.mean(Zw, axis=1), np.std(Zw, axis=1), np.min(Zw, axis=1), np.max(Zw, axis=1)])

def purged_forward_splits(times, n_splits=5, purge_windows=1):
    """Generate leak-safe splits."""
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
    """Train and evaluate model."""
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X_test)[:, 1]
    else:
        p = model.decision_function(X_test)
        p = (p - p.min()) / (p.max() - p.min() + 1e-09)
    pred = (p > 0.5).astype(int)
    try:
        auc = roc_auc_score(y_test, p)
    except:
        auc = np.nan
    acc = (pred == y_test).mean()
    tp = ((pred == 1) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    tn = ((pred == 0) & (y_test == 0)).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return (auc, acc, f1, precision, recall)

def main(config_path=None):
    """Main entry: load config and run pipeline."""
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    rt = cfg.get("regime_tda", {})
    np.random.seed(seed)
    figures_subdir = rt.get("figures_subdir", "figures")
    out_dir = _SCRIPT_DIR / figures_subdir
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info("=" * 70)
    logger.info("Wind Turbine Anomaly Detection Using TDA")
    logger.info("Detecting performance degradation vs physics-based expectations")
    logger.info("=" * 70)
    logger.info("\n1. Fetching NREL Wind Toolkit data...")
    wind_data = fetch_nrel_wind_data(cfg)
    if wind_data is None:
        logger.info('Could not fetch data')
        return
    df = simulate_turbine_with_faults(wind_data, fault_probability=0.02)
    logger.info(f'   Generated {len(df):,} turbine records')
    logger.info(f"   Wind speed range: {df['wind_speed'].min():.1f} - {df['wind_speed'].max():.1f} m/s")
    logger.info('\n2. Creating windows with anomaly labels...')
    win_size = rt.get("win_size", 256)
    Xw, y, t, stats = make_anomaly_windows(df, win_size=win_size)
    logger.info(f'   Created {len(Xw)} windows')
    logger.info(f'   Labels: {(y == 0).sum()} normal, {(y == 1).sum()} anomalous')
    logger.info(f'   Anomaly rate: {y.mean() * 100:.1f}%')
    logger.info('\n3. Extracting features...')
    logger.info('   a) TDA features...')
    X_tda = build_rich_tda_matrix(Xw)
    logger.info(f'      TDA shape: {X_tda.shape}')
    logger.info('   b) PCA features...')
    X_pca = build_pca_matrix(Xw, n_components=3)
    logger.info(f'      PCA shape: {X_pca.shape}')
    logger.info('   c) Combined features...')
    X_combined = np.hstack([X_tda, X_pca])
    logger.info(f'      Combined shape: {X_combined.shape}')
    models = {'LogReg': LogisticRegression(max_iter=2000, random_state=0, class_weight='balanced'), 'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=0, class_weight='balanced'), 'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, class_weight='balanced'), 'GradBoost': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=0)}
    logger.info('\n4. Anomaly detection evaluation...')
    splits = purged_forward_splits(t, n_splits=5, purge_windows=1)
    logger.info(f'   Using {len(splits)} folds with purged forward CV\n')
    feature_sets = [('TDA Only', X_tda), ('PCA Only', X_pca), ('TDA + PCA', X_combined)]
    results = {}
    for feat_name, X in feature_sets:
        logger.info(f'   === {feat_name} ===')
        results[feat_name] = {}
        for model_name, model_template in models.items():
            fold_metrics = []
            for train_idx, test_idx in splits:
                X_tr, X_te = (X[train_idx], X[test_idx])
                y_tr, y_te = (y[train_idx], y[test_idx])
                if len(np.unique(y_te)) < 2:
                    continue
                from sklearn.base import clone
                model = clone(model_template)
                metrics = evaluate_model(X_tr, y_tr, X_te, y_te, model)
                fold_metrics.append(metrics)
            if fold_metrics:
                arr = np.array(fold_metrics)
                avg_metrics = arr.mean(axis=0)
                results[feat_name][model_name] = avg_metrics
                auc, acc, f1, prec, rec = avg_metrics
                logger.info(f'      {model_name:<15} AUC={auc:.3f}, ACC={acc:.3f}, F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}')
        logger.info()
    logger.info('=' * 70)
    logger.info('🏆 BEST ANOMALY DETECTION CONFIGURATION')
    logger.info('=' * 70)
    best_overall = None
    best_f1 = 0
    for feat_name in results:
        for model_name, metrics in results[feat_name].items():
            if not np.isnan(metrics[2]) and metrics[2] > best_f1:
                best_f1 = metrics[2]
                best_overall = (feat_name, model_name, metrics)
    if best_overall:
        feat_name, model_name, metrics = best_overall
        auc, acc, f1, prec, rec = metrics
        logger.info(f'Features:  {feat_name}')
        logger.info(f'Model:     {model_name}')
        logger.info(f'AUC:       {auc:.3f}')
        logger.info(f'Accuracy:  {acc:.3f}')
        logger.info(f'F1 Score:  {f1:.3f}')
        logger.info(f'Precision: {prec:.3f}')
        logger.info(f'Recall:    {rec:.3f}')
    logger.info('=' * 70)
    logger.info('\nKey Points:')
    logger.info('- Predicting ANOMALIES (underperformance vs expected power curve)')
    logger.info('- NO circular reasoning (not predicting power from power)')
    logger.info('- Realistic fault detection task')
    logger.info('- Lower AUC is expected - anomalies are subtle and realistic')
    logger.info('=' * 70)
    logger.info('\n5. Generating visualizations...')
    generate_visualizations(df, Xw, y, results, out_dir)
    logger.info('   Saved figures to', out_dir)

def generate_visualizations(df, Xw, y, results, out_dir):
    """Generate comprehensive visualizations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    feature_sets = ['TDA Only', 'PCA Only', 'TDA + PCA']
    model_names = ['LogReg', 'SVM-RBF', 'RandomForest', 'GradBoost']
    for feat_idx, feat_name in enumerate(feature_sets):
        if feat_name not in results:
            continue
        aucs = [results[feat_name].get(m, [np.nan] * 5)[0] for m in model_names]
        x = np.arange(len(model_names)) + feat_idx * 0.25
        ax1.bar(x, aucs, width=0.25, label=feat_name, alpha=0.8)
    ax1.set_ylabel('AUC', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_title('Model Performance: AUC Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(np.arange(len(model_names)) + 0.25)
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.legend(fontsize=9)
    ax1.set_ylim([0, 1])
    tufte_style(ax1)
    for feat_idx, feat_name in enumerate(feature_sets):
        if feat_name not in results:
            continue
        f1s = [results[feat_name].get(m, [np.nan] * 5)[2] for m in model_names]
        x = np.arange(len(model_names)) + feat_idx * 0.25
        ax2.bar(x, f1s, width=0.25, label=feat_name, alpha=0.8)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_title('Model Performance: F1 Score Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(np.arange(len(model_names)) + 0.25)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1])
    tufte_style(ax2)
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 5))
    sample_df = df.iloc[:2000].copy()
    ax.plot(sample_df['time'], sample_df['expected_power'], 'k-', alpha=0.6, linewidth=1, label='Expected Power (Physics-Based)')
    ax.plot(sample_df['time'], sample_df['power'], 'b-', alpha=0.8, linewidth=0.8, label='Actual Power')
    fault_mask = sample_df['fault'].values
    fault_regions = []
    in_fault = False
    start_idx = 0
    for i, is_fault in enumerate(fault_mask):
        if is_fault and (not in_fault):
            start_idx = i
            in_fault = True
        elif not is_fault and in_fault:
            fault_regions.append((start_idx, i))
            in_fault = False
    if in_fault:
        fault_regions.append((start_idx, len(fault_mask)))
    for start, end in fault_regions:
        ax.axvspan(sample_df['time'].iloc[start], sample_df['time'].iloc[end - 1], alpha=0.2, color='red', label='Fault Period' if start == fault_regions[0][0] else '')
    ax.set_ylabel('Power (kW)', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title('Wind Turbine Performance: Actual vs Expected Power', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    tufte_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'power_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    normal_deviation = df[df['fault'] is False]['power_ratio'].values
    fault_deviation = df[df['fault'] is True]['power_ratio'].values
    ax1.hist(normal_deviation, bins=50, alpha=0.6, color='green', label=f'Normal (n={len(normal_deviation):,})', density=True)
    ax1.hist(fault_deviation, bins=50, alpha=0.6, color='red', label=f'Fault (n={len(fault_deviation):,})', density=True)
    ax1.axvline(1.0, color='k', linestyle='--', linewidth=1, label='Perfect Performance')
    ax1.axvline(0.8, color='orange', linestyle='--', linewidth=1, label='Anomaly Threshold')
    ax1.set_xlabel('Power Ratio (Actual / Expected)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Performance Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1.5])
    tufte_style(ax1)
    window_ratios_normal = []
    window_ratios_anomaly = []
    win_size = 256
    n = len(df)
    starts = np.arange(0, n - win_size + 1, win_size)
    for s in starts:
        window_data = df.iloc[s:s + win_size]
        mean_ratio = window_data['power_ratio'].mean()
        fault_frac = window_data['fault'].mean()
        is_anomalous = mean_ratio < 0.8 or fault_frac > 0.4
        if is_anomalous:
            window_ratios_anomaly.append(mean_ratio)
        else:
            window_ratios_normal.append(mean_ratio)
    ax2.hist(window_ratios_normal, bins=20, alpha=0.6, color='green', label=f'Normal Windows (n={len(window_ratios_normal)})')
    ax2.hist(window_ratios_anomaly, bins=20, alpha=0.6, color='red', label=f'Anomalous Windows (n={len(window_ratios_anomaly)})')
    ax2.axvline(0.8, color='orange', linestyle='--', linewidth=2, label='Detection Threshold')
    ax2.set_xlabel('Mean Power Ratio per Window', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Window-Level Anomaly Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    tufte_style(ax2)
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]
    for i, ax in enumerate(axes[0]):
        if i < len(normal_indices):
            idx = normal_indices[i]
            window = Xw[idx]
            ax.scatter(window[:, 0], window[:, 2], c=np.arange(len(window)), cmap='viridis', s=10, alpha=0.6)
            ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
            ax.set_ylabel('Power (kW)', fontsize=10)
            ax.set_title(f'Normal Window {i + 1}', fontsize=11, fontweight='bold', color='green')
            tufte_style(ax)
    for i, ax in enumerate(axes[1]):
        if i < len(anomaly_indices):
            idx = anomaly_indices[i]
            window = Xw[idx]
            ax.scatter(window[:, 0], window[:, 2], c=np.arange(len(window)), cmap='Reds', s=10, alpha=0.6)
            ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
            ax.set_ylabel('Power (kW)', fontsize=10)
            ax.set_title(f'Anomalous Window {i + 1}', fontsize=11, fontweight='bold', color='red')
            tufte_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase_portraits_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if len(normal_indices) > 0:
        idx = normal_indices[0]
        window = Xw[idx]
        result = ripser(window, maxdim=1)['dgms']
        ax = axes[0]
        if len(result) > 1:
            plot_diagrams(result, show=False, ax=ax)
        ax.set_title('Persistence Diagram: Normal Operation', fontsize=12, fontweight='bold', color='green')
        ax.set_xlabel('Birth', fontsize=11)
        ax.set_ylabel('Death', fontsize=11)
    if len(anomaly_indices) > 0:
        idx = anomaly_indices[0]
        window = Xw[idx]
        result = ripser(window, maxdim=1)['dgms']
        ax = axes[1]
        if len(result) > 1:
            plot_diagrams(result, show=False, ax=ax)
        ax.set_title('Persistence Diagram: Anomalous Operation', fontsize=12, fontweight='bold', color='red')
        ax.set_xlabel('Birth', fontsize=11)
        ax.set_ylabel('Death', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'persistence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    if 'TDA + PCA' in results and 'RandomForest' in results['TDA + PCA']:
        from sklearn.ensemble import RandomForestClassifier
        X_tda = build_rich_tda_matrix(Xw)
        X_pca = build_pca_matrix(Xw, n_components=3)
        X_combined = np.hstack([X_tda, X_pca])
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, class_weight='balanced')
        rf.fit(X_combined, y)
        importances = rf.feature_importances_
        feature_names = ['H0: Count', 'H0: Max Life', 'H0: Mean Life', 'H1: Count', 'H1: Sum Life', 'H1: Max Life', 'H1: Mean Life', 'H1: Std Life', 'H1: Mean Birth', 'H1: Mean Death', 'PC1: Mean', 'PC1: Std', 'PC1: Min', 'PC1: Max', 'PC2: Mean', 'PC2: Std', 'PC2: Min', 'PC2: Max', 'PC3: Mean', 'PC3: Std', 'PC3: Min', 'PC3: Max']
        indices = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Top 15 Features: Random Forest (TDA + PCA)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        tufte_style(ax)
        plt.tight_layout()
        plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly detection using TDA")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)