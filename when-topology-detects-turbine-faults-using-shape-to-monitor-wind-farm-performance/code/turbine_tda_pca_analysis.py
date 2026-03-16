"""
PCA Component Analysis for Wind Turbine Classification
Tests different numbers of PCA components to find optimal configuration
"""
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors
NREL_API_KEY = 'wpaaOciW3kYdcNMvRogmZEfdEueR52NS7g7Dxv0z'
NREL_API_URL = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv'

def fetch_nrel_wind_data(lat=41.0, lon=-95.5, years=[2017, 2018, 2019]):
    """Fetch real wind data from NREL Wind Toolkit API for multiple years."""
    all_data = []
    for year in years:
        logger.info(f'   Fetching year {year}...')
        params = {'api_key': NREL_API_KEY, 'wkt': f'POINT({lon} {lat})', 'attributes': 'windspeed_100m,windspeed_80m,temperature_100m', 'names': str(year), 'utc': 'true', 'leap_day': 'false', 'interval': '60', 'email': 'kyletjones@gmail.com'}
        try:
            response = requests.get(NREL_API_URL, params=params, timeout=120)
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

def simulate_turbine_from_wind(wind_df):
    """Simulate realistic turbine response to actual wind conditions."""Simulate realistic turbine response to actual wind conditions."""
    logger.info('   Simulating turbine response...')
    df = wind_df.copy()
    n = len(df)
    wind_speed = df['windspeed_100m'].values
    rotor_speed = np.zeros(n)
    power = np.zeros(n)
    RATED_POWER = 2000
    CUT_IN = 3
    RATED_WS = 12
    CUT_OUT = 25
    for i in range(1, n):
        ws = wind_speed[i]
        if ws < CUT_IN:
            target_rpm = 0
        elif ws < RATED_WS:
            target_rpm = 10 + (ws - CUT_IN) * 5
        else:
            target_rpm = min(55 + (ws - RATED_WS) * 0.3, 60)
        if ws > CUT_OUT:
            target_rpm = 0
        rotor_speed[i] = 0.85 * rotor_speed[i - 1] + 0.15 * target_rpm
        if ws < CUT_IN:
            power[i] = 0
        elif ws < RATED_WS:
            power[i] = RATED_POWER * ((ws - CUT_IN) / (RATED_WS - CUT_IN)) ** 2.5
        else:
            power[i] = RATED_POWER
        if ws > CUT_OUT:
            power[i] = 0
        power[i] += np.random.normal(0, 5)
        power[i] = np.clip(power[i], 0, RATED_POWER * 1.05)
    rotor_speed += np.random.normal(0, 0.3, n)
    rotor_speed = np.clip(rotor_speed, 0, 65)
    result = pd.DataFrame({'time': df['time'], 'wind_speed': wind_speed, 'rotor_speed': rotor_speed, 'power': power})
    return result

def make_advanced_windows(df, win_size=256):
    """Create non-overlapping windows with capacity factor labeling."""
    X = df[['wind_speed', 'rotor_speed', 'power']].to_numpy(float)
    n = len(df)
    starts = np.arange(0, n - win_size + 1, win_size)
    RATED_POWER = 2000
    Xw, yw_cf, tw = ([], [], [])
    for s in starts:
        slab = X[s:s + win_size]
        power_window = df.iloc[s:s + win_size]['power'].values
        capacity_factor = power_window.mean() / RATED_POWER
        label_cf = 1 if capacity_factor > 0.35 else 0
        time_mid = df.loc[s + win_size // 2, 'time']
        Xw.append(slab)
        yw_cf.append(label_cf)
        tw.append(time_mid)
    Xw = np.array(Xw, dtype=float)
    yw_cf = np.array(yw_cf, dtype=int)
    tw = pd.to_datetime(tw)
    return (Xw, yw_cf, tw)

def lifetimes(dgm):
    """Compute persistence lifetimes from a diagram."""Compute persistence lifetimes from a diagram."""
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
    """Extract rich TDA features from all windows."""Extract rich TDA features from all windows."""
    features = []
    for i, w in enumerate(Xw):
        if (i + 1) % 20 == 0:
            logger.info(f'      Window {i + 1}/{len(Xw)}')
        features.append(extract_rich_tda_features(w))
    return np.vstack(features)

def build_pca_matrix_multi(Xw, n_components_list=[2, 3, 5, 10, 20]):
    """
    Build PCA features with multiple component configurations.
    Returns a dictionary of feature matrices.
    """
    n_win, win, dim = Xw.shape
    flat = Xw.reshape(n_win * win, dim)
    scaler = StandardScaler().fit(flat)
    flat_std = scaler.transform(flat)
    results = {}
    max_components = min(dim, flat_std.shape[0])
    for n_comp in n_components_list:
        if n_comp > max_components:
            logger.info(f'      Skipping {n_comp} components (max={max_components})')
            continue
        pca = PCA(n_components=n_comp, random_state=0)
        Z = pca.fit_transform(flat_std)
        Zw = Z.reshape(n_win, win, n_comp)
        features = []
        features.append(np.mean(Zw, axis=1))
        features.append(np.std(Zw, axis=1))
        features.append(np.min(Zw, axis=1))
        features.append(np.max(Zw, axis=1))
        X_pca = np.hstack(features)
        results[n_comp] = {'features': X_pca, 'variance_explained': pca.explained_variance_ratio_.sum(), 'n_features': X_pca.shape[1]}
        logger.info(f'      {n_comp} PCs: {X_pca.shape[1]} features, {pca.explained_variance_ratio_.sum():.3f} variance explained')
    return results

def purged_forward_splits(times, n_splits=5, purge_windows=1):
    """Generate leak-safe time-series splits."""
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
    """Train model and return metrics."""Train model and return metrics."""
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
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return (auc, acc, f1)

def main():
    """
    Perform main operation.

    Args:
        None

    Returns:
        Result of the operation..
    """
    np.random.seed(0)
    logger.info('=' * 70)
    logger.info('PCA Component Analysis for Wind Turbine Classification')
    logger.info('Testing different numbers of PCA components')
    logger.info('=' * 70)
    logger.info('\n1. Fetching NREL Wind Toolkit data...')
    wind_data = fetch_nrel_wind_data(lat=41.0, lon=-95.5, years=[2017, 2018, 2019])
    if wind_data is None:
        logger.info('Could not fetch data')
        return
    df = simulate_turbine_from_wind(wind_data)
    logger.info(f'   Generated {len(df):,} turbine records')
    logger.info('\n2. Creating windows...')
    Xw, y_cf, t = make_advanced_windows(df, win_size=256)
    logger.info(f'   Created {len(Xw)} windows')
    logger.info(f'   Labels: {(y_cf == 0).sum()} low-productivity, {(y_cf == 1).sum()} high-productivity')
    logger.info('\n3. Extracting TDA features...')
    X_tda = build_rich_tda_matrix(Xw)
    logger.info(f'   TDA features: {X_tda.shape}')
    logger.info('\n4. Extracting PCA features with multiple component counts...')
    pca_results = build_pca_matrix_multi(Xw, n_components_list=[2, 3, 5, 10, 20, 50, 100])
    models = {'LogReg': LogisticRegression(max_iter=2000, random_state=0), 'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=0), 'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0), 'GradBoost': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=0)}
    logger.info('\n5. Evaluating with purged forward CV...')
    splits = purged_forward_splits(t, n_splits=5, purge_windows=1)
    logger.info(f'   Using {len(splits)} folds')
    results = {}
    logger.info('\n   === PCA Component Analysis ===\n')
    for n_comp in sorted(pca_results.keys()):
        X_pca = pca_results[n_comp]['features']
        var_exp = pca_results[n_comp]['variance_explained']
        n_feat = pca_results[n_comp]['n_features']
        logger.info(f'   {n_comp} PCA Components ({n_feat} features, {var_exp:.1%} variance):')
        results[n_comp] = {}
        for model_name, model_template in models.items():
            fold_metrics = []
            for train_idx, test_idx in splits:
                X_tr, X_te = (X_pca[train_idx], X_pca[test_idx])
                y_tr, y_te = (y_cf[train_idx], y_cf[test_idx])
                if len(np.unique(y_te)) < 2:
                    continue
                from sklearn.base import clone
                model = clone(model_template)
                metrics = evaluate_model(X_tr, y_tr, X_te, y_te, model)
                fold_metrics.append(metrics)
            if fold_metrics:
                arr = np.array(fold_metrics)
                avg_metrics = arr.mean(axis=0)
                results[n_comp][model_name] = avg_metrics
                auc, acc, f1 = avg_metrics
                logger.info(f'      {model_name:<15} AUC={auc:.4f}, ACC={acc:.3f}, F1={f1:.3f}')
        logger.info()
    logger.info('\n   === TDA + PCA Combinations ===\n')
    tda_pca_results = {}
    for n_comp in sorted(pca_results.keys()):
        X_pca = pca_results[n_comp]['features']
        X_combined = np.hstack([X_tda, X_pca])
        logger.info(f'   TDA + {n_comp} PCA components ({X_combined.shape[1]} total features):')
        tda_pca_results[n_comp] = {}
        for model_name in ['RandomForest', 'GradBoost']:
            model_template = models[model_name]
            fold_metrics = []
            for train_idx, test_idx in splits:
                X_tr, X_te = (X_combined[train_idx], X_combined[test_idx])
                y_tr, y_te = (y_cf[train_idx], y_cf[test_idx])
                if len(np.unique(y_te)) < 2:
                    continue
                from sklearn.base import clone
                model = clone(model_template)
                metrics = evaluate_model(X_tr, y_tr, X_te, y_te, model)
                fold_metrics.append(metrics)
            if fold_metrics:
                arr = np.array(fold_metrics)
                avg_metrics = arr.mean(axis=0)
                tda_pca_results[n_comp][model_name] = avg_metrics
                auc, acc, f1 = avg_metrics
                logger.info(f'      {model_name:<15} AUC={auc:.4f}, ACC={acc:.3f}, F1={f1:.3f}')
        logger.info()
    logger.info('=' * 70)
    logger.info('SUMMARY: BEST RESULTS BY PCA COMPONENT COUNT')
    logger.info('=' * 70)
    logger.info('\nPCA Only:')
    logger.info(f"{'Components':<12} {'Best Model':<15} {'AUC':<10} {'Accuracy':<10} {'F1':<10}")
    logger.info('-' * 70)
    for n_comp in sorted(results.keys()):
        best_auc = 0
        best_model = ''
        best_metrics = None
        for model_name, metrics in results[n_comp].items():
            if not np.isnan(metrics[0]) and metrics[0] > best_auc:
                best_auc = metrics[0]
                best_model = model_name
                best_metrics = metrics
        if best_metrics is not None:
            auc, acc, f1 = best_metrics
            logger.info(f'{n_comp:<12} {best_model:<15} {auc:<10.4f} {acc:<10.3f} {f1:<10.3f}')
    logger.info('\nTDA + PCA Combined:')
    logger.info(f"{'Components':<12} {'Best Model':<15} {'AUC':<10} {'Accuracy':<10} {'F1':<10}")
    logger.info('-' * 70)
    for n_comp in sorted(tda_pca_results.keys()):
        best_auc = 0
        best_model = ''
        best_metrics = None
        for model_name, metrics in tda_pca_results[n_comp].items():
            if not np.isnan(metrics[0]) and metrics[0] > best_auc:
                best_auc = metrics[0]
                best_model = model_name
                best_metrics = metrics
        if best_metrics is not None:
            auc, acc, f1 = best_metrics
            logger.info(f'{n_comp:<12} {best_model:<15} {auc:<10.4f} {acc:<10.3f} {f1:<10.3f}')
    logger.info('\n' + '=' * 70)
    logger.info('🏆 OVERALL BEST CONFIGURATION')
    logger.info('=' * 70)
    best_overall = None
    best_auc_overall = 0
    for n_comp in results:
        for model_name, metrics in results[n_comp].items():
            if not np.isnan(metrics[0]) and metrics[0] > best_auc_overall:
                best_auc_overall = metrics[0]
                best_overall = ('PCA Only', n_comp, model_name, metrics)
    for n_comp in tda_pca_results:
        for model_name, metrics in tda_pca_results[n_comp].items():
            if not np.isnan(metrics[0]) and metrics[0] > best_auc_overall:
                best_auc_overall = metrics[0]
                best_overall = ('TDA+PCA', n_comp, model_name, metrics)
    if best_overall:
        feat_type, n_comp, model_name, metrics = best_overall
        auc, acc, f1 = metrics
        logger.info(f'Feature Set:  {feat_type}')
        logger.info(f'PCA Components: {n_comp}')
        logger.info(f'Model:        {model_name}')
        logger.info(f'AUC:          {auc:.4f}')
        logger.info(f'Accuracy:     {acc:.3f}')
        logger.info(f'F1 Score:     {f1:.3f}')
        if feat_type == 'PCA Only':
            var_exp = pca_results[n_comp]['variance_explained']
            n_feat = pca_results[n_comp]['n_features']
            logger.info(f'Total Features: {n_feat}')
            logger.info(f'Variance Explained: {var_exp:.1%}')
        logger.info('=' * 70)
    logger.info('\nAnalysis complete!')
if __name__ == '__main__':
    main()