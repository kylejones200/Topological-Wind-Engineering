"""
Regime Transition Prediction (prototype)

Sliding-window ordinary persistence features for regime-transition prediction.
Uses hourly NREL atmospheric data with simulated turbine response.

Note: This is not a zigzag persistence implementation. Window parameters are
specified in sample counts (row indices), not minutes. See config/default.yaml.
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

import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.spatial.distance import directed_hausdorff
import warnings
import logging

from config.load import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def fetch_nrel_wind_data(cfg):
    """Fetch wind data from NREL. Uses config for lat, lon, years, api_key, url, timeout."""
    nrel = cfg.get("nrel", {})
    lat = nrel.get("lat", 41.5)
    lon = nrel.get("lon", -93.5)
    years = nrel.get("years", [2017])
    api_key = nrel.get("api_key") or os.environ.get("NREL_API_KEY", "")
    base_url = nrel.get("base_url", "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv")
    timeout = nrel.get("request_timeout_seconds", 120)
    email = nrel.get("email", "user@example.com")
    interval = nrel.get("interval", "60")
    attributes = nrel.get("attributes", "windspeed_100m,temperature_100m")
    all_data = []
    for year in years:
        logger.info(f"   Fetching year {year}...")
        params = {
            "api_key": api_key,
            "wkt": f"POINT({lon} {lat})",
            "attributes": attributes,
            "names": str(year),
            "utc": "true",
            "leap_day": "false",
            "interval": interval,
            "email": email,
        }
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
            df_year = pd.read_csv(StringIO(data_text), header=None, names=['Year', 'Month', 'Day', 'Hour', 'Minute', 'windspeed_100m', 'temperature_100m'])
            df_year['time'] = pd.to_datetime(df_year[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            all_data.append(df_year)
            logger.info(f'     ✓ Fetched {len(df_year):,} records')
        except Exception as e:
            logger.error(f'     ✗ Error: {e}')
            continue
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True).sort_values('time')

def simulate_turbine_operation(wind_df, cfg):
    """
    Simulate turbine with realistic regime transitions.
    
    Regimes:
    - Idle: wind < 3 m/s
    - Startup: 3 <= wind < 4 m/s
    - Ramp-up: 4 <= wind < 12 m/s
    - Rated: 12 <= wind < 25 m/s
    - Shutdown: wind >= 25 m/s
    """
    rated_power = cfg.get("simulation", {}).get("rated_power_kw", 2000)
    df = wind_df.copy()
    n = len(df)
    wind = df["windspeed_100m"].values
    wind_turbulent = wind + np.random.randn(n) * 0.5
    wind_turbulent = np.maximum(wind_turbulent, 0)
    rotor_speed = np.zeros(n)
    power = np.zeros(n)
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        w = wind_turbulent[i]
        if w < 3:
            regime[i] = 0
            target_rpm = 0
            target_power = 0
        elif w < 4:
            regime[i] = 1
            target_rpm = 5 + (w - 3) * 5
            target_power = rated_power * 0.01
        elif w < 12:
            regime[i] = 2
            target_rpm = 10 + (w - 4) * 5
            target_power = rated_power * ((w - 4) / (12 - 4)) ** 2.5
        elif w < 25:
            regime[i] = 3
            target_rpm = 55 + (w - 12) * 0.3
            target_power = rated_power
        else:
            regime[i] = 4
            target_rpm = 0
            target_power = 0
        rotor_speed[i] = 0.8 * rotor_speed[i - 1] + 0.2 * target_rpm
        power[i] = 0.7 * power[i - 1] + 0.3 * target_power
        rotor_speed[i] += np.random.randn() * 0.3
        power[i] += np.random.randn() * 10
        rotor_speed[i] = np.maximum(rotor_speed[i], 0)
        power[i] = np.clip(power[i], 0, rated_power * 1.1)
    df['wind_turbulent'] = wind_turbulent
    df['rotor_speed'] = rotor_speed
    df['power'] = power
    df['regime'] = regime
    return df

def detect_transitions(df, min_duration_samples=10):
    """
    Detect regime transitions.

    Args:
        df: DataFrame with regime labels
        min_duration_samples: Minimum consecutive samples for a stable regime
    
    Returns:
        List of (index, from_regime, to_regime) tuples
    """
    regime = df['regime'].values
    transitions = []
    i = 0
    while i < len(regime) - min_duration_samples:
        current_regime = regime[i]
        for j in range(i + 1, min(i + 60, len(regime))):
            if regime[j] != current_regime:
                new_regime = regime[j]
                if j + min_duration_samples < len(regime):
                    if np.all(regime[j:j + min_duration_samples] == new_regime):
                        transitions.append((j, current_regime, new_regime))
                        i = j + min_duration_samples
                        break
        else:
            i += 1
    return transitions

def extract_zigzag_window(
    df,
    transition_idx,
    window_samples=30,
    subwindow_samples=5,
    slide_samples=1,
    lead_samples=15,
):
    """
    Extract feature windows before a transition using overlapping subwindows.

    Args:
        df: DataFrame
        transition_idx: Index where transition occurs
        window_samples: Total lookback in samples (rows) before lead offset
        subwindow_samples: Subwindow length in samples
        slide_samples: Step between subwindows in samples
        lead_samples: Samples before transition_idx where the window ends
    
    Returns:
        List of subwindow DataFrames
    """
    end_idx = max(0, transition_idx - lead_samples)
    start_idx = max(0, end_idx - window_samples)
    if start_idx >= end_idx - subwindow_samples:
        return None
    subwindows = []
    current = start_idx
    while current + subwindow_samples <= end_idx:
        subwin = df.iloc[int(current):int(current + subwindow_samples)].copy()
        if len(subwin) >= subwindow_samples * 0.8:
            subwindows.append(subwin)
        current += slide_samples
    return subwindows if len(subwindows) > 10 else None

def compute_persistence_for_subwindow(subwin):
    """Compute persistence diagram for one subwindow."""
    wind = subwin['wind_turbulent'].values
    rotor = subwin['rotor_speed'].values
    power = subwin['power'].values
    X = np.column_stack([wind, rotor, power])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    result = ripser(X, maxdim=1)
    return result['dgms']

def bottleneck_distance_approx(dgm1, dgm2):
    """
    Approximate bottleneck distance between two persistence diagrams.
    Uses directed Hausdorff as a fast proxy.
    """
    H1_1 = dgm1[1] if len(dgm1) > 1 else np.empty((0, 2))
    H1_2 = dgm2[1] if len(dgm2) > 1 else np.empty((0, 2))
    H1_1_finite = H1_1[np.isfinite(H1_1[:, 1])]
    H1_2_finite = H1_2[np.isfinite(H1_2[:, 1])]
    if len(H1_1_finite) == 0 or len(H1_2_finite) == 0:
        return 0.0
    d1 = directed_hausdorff(H1_1_finite, H1_2_finite)[0]
    d2 = directed_hausdorff(H1_2_finite, H1_1_finite)[0]
    return max(d1, d2)

def extract_zigzag_features(subwindows):
    """
    Extract features from zigzag persistence across subwindows.
    
    Args:
        subwindows: List of DataFrame subwindows
    
    Returns:
        Feature dictionary
    """
    diagrams = []
    for subwin in subwindows:
        dgm = compute_persistence_for_subwindow(subwin)
        diagrams.append(dgm)
    h1_counts = []
    h1_max_lifetimes = []
    h1_entropies = []
    for dgm in diagrams:
        H1 = dgm[1] if len(dgm) > 1 else np.empty((0, 2))
        H1_finite = H1[np.isfinite(H1[:, 1])]
        if len(H1_finite) > 0:
            lifetimes = H1_finite[:, 1] - H1_finite[:, 0]
            h1_counts.append(len(lifetimes))
            h1_max_lifetimes.append(lifetimes.max())
            p = lifetimes / lifetimes.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            h1_entropies.append(entropy)
        else:
            h1_counts.append(0)
            h1_max_lifetimes.append(0)
            h1_entropies.append(0)
    bottleneck_dists = []
    for i in range(len(diagrams) - 1):
        dist = bottleneck_distance_approx(diagrams[i], diagrams[i + 1])
        bottleneck_dists.append(dist)
    features = {}
    features['h1_count_mean'] = np.mean(h1_counts)
    features['h1_count_std'] = np.std(h1_counts)
    features['h1_max_life_mean'] = np.mean(h1_max_lifetimes)
    features['h1_max_life_std'] = np.std(h1_max_lifetimes)
    features['h1_entropy_mean'] = np.mean(h1_entropies)
    features['h1_entropy_std'] = np.std(h1_entropies)
    if len(h1_entropies) > 2:
        entropy_trend = np.polyfit(range(len(h1_entropies)), h1_entropies, 1)[0]
        features['h1_entropy_trend'] = entropy_trend
    else:
        features['h1_entropy_trend'] = 0
    features['bottleneck_mean'] = np.mean(bottleneck_dists)
    features['bottleneck_std'] = np.std(bottleneck_dists)
    features['bottleneck_max'] = np.max(bottleneck_dists)
    if len(bottleneck_dists) > 2:
        bn_trend = np.polyfit(range(len(bottleneck_dists)), bottleneck_dists, 1)[0]
        features['bottleneck_trend'] = bn_trend
    else:
        features['bottleneck_trend'] = 0
    all_wind = np.concatenate([s['wind_turbulent'].values for s in subwindows])
    all_power = np.concatenate([s['power'].values for s in subwindows])
    features['wind_mean'] = np.mean(all_wind)
    features['wind_std'] = np.std(all_wind)
    features['power_mean'] = np.mean(all_power)
    features['power_std'] = np.std(all_power)
    return features

def create_dataset(df, transitions, n_negative_samples=None, window_cfg=None):
    """
    Create dataset with positive (pre-transition) and negative (stable) windows.
    
    Args:
        df: Full DataFrame
        transitions: List of transition tuples
        n_negative_samples: Number of negative samples (default: 2x positives)
    
    Returns:
        X (features), y (labels)
    """
    window_cfg = window_cfg or {}
    window_samples = window_cfg.get("window_samples", 30)
    subwindow_samples = window_cfg.get("subwindow_samples", 5)
    slide_samples = window_cfg.get("slide_samples", 1)
    lead_samples = window_cfg.get("lead_samples", 15)
    logger.info('\n   Creating dataset...')
    positive_features = []
    for trans_idx, from_regime, to_regime in transitions:
        subwindows = extract_zigzag_window(
            df,
            trans_idx,
            window_samples=window_samples,
            subwindow_samples=subwindow_samples,
            slide_samples=slide_samples,
            lead_samples=lead_samples,
        )
        if subwindows is not None:
            features = extract_zigzag_features(subwindows)
            positive_features.append(features)
    logger.info(f'     Positive samples (pre-transition): {len(positive_features)}')
    if n_negative_samples is None:
        n_negative_samples = len(positive_features) * 2
    negative_features = []
    attempts = 0
    max_attempts = n_negative_samples * 10
    while len(negative_features) < n_negative_samples and attempts < max_attempts:
        attempts += 1
        start_idx = np.random.randint(50, len(df) - 100)
        future_window = df.iloc[start_idx:start_idx + 60]
        if future_window['regime'].nunique() > 1:
            continue
        subwindows = extract_zigzag_window(
            df,
            start_idx + 45,
            window_samples=window_samples,
            subwindow_samples=subwindow_samples,
            slide_samples=slide_samples,
            lead_samples=lead_samples,
        )
        if subwindows is not None:
            features = extract_zigzag_features(subwindows)
            negative_features.append(features)
    logger.info(f'     Negative samples (stable periods): {len(negative_features)}')
    all_features = positive_features + negative_features
    y = np.array([1] * len(positive_features) + [0] * len(negative_features))
    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    return (X, y, feature_names)

def generate_visualizations(X, y, y_pred, y_prob, feature_importance, feature_names, out_dir):
    """Generate visualizations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 14
    indices = np.argsort(feature_importance)[-top_n:]
    ax.barh(range(top_n), feature_importance[indices], color='#2b2b2b', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Top Features for Transition Prediction', fontsize=12, fontweight='normal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-', linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve: Transition Prediction', fontsize=12, fontweight='normal')
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'\nVisualizations saved to {out_dir}/')

REGIME_NAMES = ['Idle', 'Startup', 'Ramp-up', 'Rated', 'Shutdown']


def _log_banner():
    """Log pipeline header."""
    logger.info("=" * 70)
    logger.info("Regime Transition Prediction (sliding-window persistence prototype)")
    logger.info("=" * 70)


def _fetch_wind_or_exit(cfg):
    """Fetch NREL wind data; log and return None on failure."""
    logger.info("\n1. Fetching NREL wind data...")
    wind_data = fetch_nrel_wind_data(cfg)
    if wind_data is None:
        logger.error("Failed to fetch data")
        return None
    logger.info(f"   Total records: {len(wind_data):,}")
    return wind_data


def _simulate_and_log_regimes(wind_data, cfg):
    """Simulate turbine operation and log regime counts. Returns DataFrame."""
    logger.info("\n2. Simulating turbine operation...")
    df = simulate_turbine_operation(wind_data, cfg)
    regime_counts = df['regime'].value_counts().sort_index()
    for regime_id, count in regime_counts.items():
        logger.info(f'   {REGIME_NAMES[regime_id]}: {count:,} records ({count / len(df) * 100:.1f}%)')
    return df


def _detect_and_log_transitions(df, cfg):
    """Detect regime transitions and log counts by type. Returns list of transitions."""
    logger.info('\n3. Detecting regime transitions...')
    rt_cfg = cfg.get("regime_transition", {})
    min_duration = rt_cfg.get("min_regime_duration_samples", 10)
    transitions = detect_transitions(df, min_duration_samples=min_duration)
    logger.info(f'   Found {len(transitions)} transitions')
    transition_types = {}
    for _, from_r, to_r in transitions:
        key = f'{REGIME_NAMES[from_r]} → {REGIME_NAMES[to_r]}'
        transition_types[key] = transition_types.get(key, 0) + 1
    logger.info('   Transition types:')
    for ttype, count in sorted(transition_types.items(), key=lambda x: -x[1])[:10]:
        logger.info(f'     {ttype}: {count}')
    return transitions


def _build_dataset_and_log(df, transitions, cfg):
    """Build persistence-feature dataset and log shape/labels. Returns (X, y, feature_names)."""
    logger.info('\n4. Creating dataset with sliding-window persistence features...')
    rt_cfg = cfg.get("regime_transition", {})
    X, y, feature_names = create_dataset(
        df,
        transitions,
        n_negative_samples=len(transitions) * 2,
        window_cfg=rt_cfg,
    )
    logger.info(f'   Dataset shape: {X.shape}')
    logger.info(f'   Transitions (positive): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)')
    logger.info(f'   Stable (negative): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)')
    return X, y, feature_names


def _stratified_split_and_scale(X, y, seed, train_ratio):
    """Split with stratification and scale. Returns (X_train_scaled, X_test_scaled, y_train, y_test)."""
    logger.info("\n5. Splitting data with stratification...")
    test_size = 1.0 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    logger.info(f'   Train: {len(X_train)} samples (positive: {(y_train == 1).sum()})')
    logger.info(f'   Test: {len(X_test)} samples (positive: {(y_test == 1).sum()})')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def _train_classifiers(X_train_scaled, y_train, X_test_scaled, y_test, seed, n_estimators):
    """Train RF, GB, SVM-RBF, LogReg and log metrics. Returns results dict."""
    logger.info('\n6. Training classifiers...')
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=12, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, max_depth=5, random_state=seed),
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
    }
    results = {}
    for name, model in models.items():
        logger.info(f'\n   Training {name}...')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test_scaled)
            y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            y_prob = model.decision_function(X_test_scaled)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'auc': auc, 'accuracy': acc, 'y_pred': y_pred, 'y_prob': y_prob}
        logger.info(f'      AUC: {auc:.3f}')
        logger.info(f'      Accuracy: {acc:.3f}')
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Stable', 'Pre-Transition'])}")
    return results


def _log_feature_importance_and_save_viz(results, feature_names, X_test_scaled, y_test, out_dir):
    """Log top feature importance and call generate_visualizations."""
    logger.info('\n7. Analyzing feature importance...')
    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]
    logger.info('\n   Top 10 features:')
    for fname, importance in top_features:
        logger.info(f'      {fname}: {importance:.4f}')
    logger.info("\n8. Generating visualizations...")
    generate_visualizations(
        X_test_scaled, y_test,
        results["Random Forest"]["y_pred"], results["Random Forest"]["y_prob"],
        feature_importance, feature_names,
        str(out_dir),
    )


def _log_final_summary(results, cfg):
    """Log completion banner and best model summary."""
    rt_cfg = cfg.get("regime_transition", {})
    interval_h = float(rt_cfg.get("sample_interval_hours", 1.0))
    lead_samples = int(rt_cfg.get("lead_samples", 15))
    lead_hours = lead_samples * interval_h
    logger.info('\n' + '=' * 70)
    logger.info('TRANSITION PREDICTION COMPLETE (synthetic prototype)')
    logger.info('=' * 70)
    logger.info(f'\nBest model: Random Forest')
    logger.info(f"AUC: {results['Random Forest']['auc']:.3f}")
    logger.info(f"Accuracy: {results['Random Forest']['accuracy']:.3f}")
    logger.info(
        f'Configured lead offset: {lead_samples} samples '
        f'(~{lead_hours:.1f} h at {interval_h:g}-h sampling)'
    )
    logger.info('Note: archived metrics are synthetic; see SYNTHETIC_RESULTS.md')
    logger.info('=' * 70)


def main(config_path=None):
    """Main entry: load config and run pipeline. All parameters from config."""
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    np.random.seed(seed)
    rt_cfg = cfg.get("regime_transition", {})
    n_est = cfg.get("farm_coordination", {}).get("n_estimators", 100) or 100
    train_ratio = rt_cfg.get("train_ratio", 0.7)
    figures_subdir = rt_cfg.get("figures_subdir", "figures_transitions")
    out_dir = _SCRIPT_DIR / figures_subdir

    _log_banner()
    wind_data = _fetch_wind_or_exit(cfg)
    if wind_data is None:
        return
    df = _simulate_and_log_regimes(wind_data, cfg)
    transitions = _detect_and_log_transitions(df, cfg)
    X, y, feature_names = _build_dataset_and_log(df, transitions, cfg)
    X_train_scaled, X_test_scaled, y_train, y_test = _stratified_split_and_scale(X, y, seed, train_ratio)
    results = _train_classifiers(X_train_scaled, y_train, X_test_scaled, y_test, seed, n_est)
    _log_feature_importance_and_save_viz(results, feature_names, X_test_scaled, y_test, out_dir)
    _log_final_summary(results, cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regime transition prediction (zigzag persistence)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)