"""
Wake Detection Using Persistent Homology (Improved & Optimized)

Detects when turbines operate in wake vs free-stream conditions using
topological features from power-windspeed phase space.

"""
from typing import List, Tuple, Dict
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import warnings
from tda_utils import TurbineConfig, TufteColors, setup_tufte_plot, compute_power_curve_vectorized, add_power_noise, extract_datetime_features, create_seasonal_pattern, create_diurnal_pattern, extract_persistence_lifetimes, compute_persistence_entropy, print_classification_summary
np.random.seed(42)

def fetch_nrel_wind_data(lat: float=41.5, lon: float=-100.5, years: List[int]=[2010, 2011, 2012]) -> pd.DataFrame:
    """
    Simulate NREL Wind Toolkit data fetch.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        years: List of years to simulate
    
    Returns:
        DataFrame with timestamp, windspeed_80m, wind_direction, temperature
    """
    logger.info(f'Simulating NREL wind data fetch for location ({lat}, {lon})')
    n_records = 365 * 24 * 12 * len(years)
    timestamps = pd.date_range(start=f'{years[0]}-01-01', periods=n_records, freq='5min')
    time_features = extract_datetime_features(timestamps)
    hours = time_features['hour_fractional']
    days = time_features['dayofyear']
    seasonal = create_seasonal_pattern(days, amplitude=2.0)
    diurnal = create_diurnal_pattern(hours, amplitude=1.5)
    windspeed_80m = 8.5 + seasonal + diurnal + np.random.normal(0, 2, n_records)
    windspeed_80m = np.clip(windspeed_80m, 0, 25)
    wind_direction = 180 + 60 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 15, n_records)
    wind_direction = wind_direction % 360
    temperature = 15 + 10 * np.cos(2 * np.pi * days / 365) + np.random.normal(0, 3, n_records)
    df = pd.DataFrame({'timestamp': timestamps, 'windspeed_80m': windspeed_80m, 'wind_direction': wind_direction, 'temperature': temperature})
    logger.info(f'Fetched {len(df):,} records spanning {len(years)} years')
    return df

def simulate_turbine_power(windspeed: np.ndarray, in_wake: bool=False, config: TurbineConfig=TurbineConfig()) -> np.ndarray:
    """
    Simulate turbine power output using IEC power curve (VECTORIZED).
    
    Wake conditions reduce effective windspeed and add hysteresis.
    This vectorized version is ~100x faster than the loop-based approach.
    
    Args:
        windspeed: Wind speed array (m/s)
        in_wake: Whether turbine is in wake condition
        config: Turbine configuration parameters
    
    Returns:
        Power output array (MW)
    """
    if in_wake:
        windspeed = windspeed * np.random.uniform(0.6, 0.7)
    power = compute_power_curve_vectorized(windspeed, config)
    power = add_power_noise(power, noise_std=0.05, rated_power=config.rated_power)
    if in_wake:
        alpha = 0.7
        for i in range(1, len(power)):
            power[i] = alpha * power[i] + (1 - alpha) * power[i - 1]
    return power

def create_wake_scenarios(df: pd.DataFrame, n_windows: int=120, window_size: int=288) -> Tuple[List[pd.DataFrame], np.ndarray]:
    """
    Create labeled windows of wake vs free-stream conditions.
    
    Args:
        df: Wind data DataFrame
        n_windows: Number of windows to create
        window_size: Samples per window (288 = 24 hours at 5-min resolution)
    
    Returns:
        Tuple of (list of window DataFrames, label array)
    """
    logger.info(f'\nCreating {n_windows} labeled windows (wake vs free-stream)...')
    windows = []
    labels = []
    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)
    for idx, start in enumerate(starts):
        window_df = df.iloc[start:start + window_size].copy()
        is_wake = idx < n_windows // 2
        power = simulate_turbine_power(window_df['windspeed_80m'].values, in_wake=is_wake)
        window_df['power'] = power
        windows.append(window_df)
        labels.append(1 if is_wake else 0)
    logger.info(f'Created {sum(labels)} wake windows and {len(labels) - sum(labels)} free-stream windows')
    return (windows, np.array(labels))

def compute_persistence_features(window_df: pd.DataFrame, max_dim: int=1) -> Dict[str, float]:
    """
    Compute persistent homology features from power-windspeed phase space.
    
    Wake conditions create characteristic loop structures (H1 features) due
    to hysteresis in the power response.
    
    Args:
        window_df: DataFrame with 'power' and 'windspeed_80m' columns
        max_dim: Maximum homology dimension to compute
    
    Returns:
        Dictionary of topological and statistical features
    """
    power = window_df['power'].values
    windspeed = window_df['windspeed_80m'].values
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-08)
    wind_norm = (windspeed - windspeed.min()) / (windspeed.max() - windspeed.min() + 1e-08)
    X = np.column_stack([power_norm, wind_norm])
    result = ripser(X, maxdim=max_dim)
    diagrams = result['dgms']
    features = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            features[f'H{dim}_count'] = 0
            features[f'H{dim}_sum_lifetime'] = 0
            features[f'H{dim}_max_lifetime'] = 0
            features[f'H{dim}_mean_birth'] = 0
            features[f'H{dim}_mean_death'] = 0
        else:
            lifetimes = extract_persistence_lifetimes(dgm, remove_infinite=True)
            if len(lifetimes) == 0:
                features[f'H{dim}_count'] = 0
                features[f'H{dim}_sum_lifetime'] = 0
                features[f'H{dim}_max_lifetime'] = 0
                features[f'H{dim}_mean_birth'] = 0
                features[f'H{dim}_mean_death'] = 0
            else:
                finite_dgm = dgm[np.isfinite(dgm[:, 1])]
                features[f'H{dim}_count'] = len(finite_dgm)
                features[f'H{dim}_sum_lifetime'] = np.sum(lifetimes)
                features[f'H{dim}_max_lifetime'] = np.max(lifetimes)
                features[f'H{dim}_mean_birth'] = np.mean(finite_dgm[:, 0])
                features[f'H{dim}_mean_death'] = np.mean(finite_dgm[:, 1])
    features['power_mean'] = power.mean()
    features['power_std'] = power.std()
    features['wind_mean'] = windspeed.mean()
    features['wind_std'] = windspeed.std()
    features['power_wind_corr'] = np.corrcoef(power, windspeed)[0, 1]
    return features

def extract_all_features(windows: List[pd.DataFrame], labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract topological and statistical features for all windows.
    
    Args:
        windows: List of window DataFrames
        labels: Label array
    
    Returns:
        Tuple of (feature DataFrame, label array)
    """
    logger.info('\nExtracting topological features from all windows...')
    feature_list = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info(f'  Processing window {i + 1}/{len(windows)}')
        features = compute_persistence_features(window_df, max_dim=1)
        feature_list.append(features)
    X = pd.DataFrame(feature_list)
    y = labels
    logger.info(f'\nFeature matrix: {X.shape}')
    logger.info(f'Features: {list(X.columns)}')
    logger.info(f'Label distribution: Wake={sum(y)}, Free-stream={len(y) - sum(y)}')
    return (X, y)

def train_and_evaluate_models(X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Train multiple classifiers and compare performance.
    
    Args:
        X: Feature DataFrame
        y: Label array
    
    Returns:
        Tuple of (results dict, X_train, X_test, y_train, y_test)
    """
    logger.info('\n' + '=' * 60)
    logger.info('TRAINING AND EVALUATION')
    logger.info('=' * 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logger.info(f'\nTrain set: {len(X_train)} samples')
    logger.info(f'Test set: {len(X_test)} samples')
    models = {'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000), 'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True), 'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True), 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)}
    results = {}
    for name, model in models.items():
        logger.info(f'\n{name}:')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        logger.info(f'  Accuracy: {acc:.3f}')
        logger.info(f'  F1 Score: {f1:.3f}')
        if auc is not None:
            logger.info(f'  AUC: {auc:.3f}')
        results[name] = {'model': model, 'accuracy': acc, 'f1': f1, 'auc': auc, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba}
    return (results, X_train, X_test, y_train, y_test)

def generate_visualizations(windows: List[pd.DataFrame], labels: np.ndarray, X: pd.DataFrame, y: np.ndarray, results: Dict, X_test: pd.DataFrame, y_test: np.ndarray, out_dir: Path) -> None:
"""Generate comprehensive visualizations using Tufte-style formatting."""
    logger.info('\n' + '=' * 60)
    logger.info('GENERATING VISUALIZATIONS')
    logger.info('=' * 60)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info('\n1. Model comparison bar charts...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1s = [results[m]['f1'] for m in model_names]
    axes[0].bar(range(len(model_names)), accuracies, color=TufteColors.BLACK, alpha=0.85)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    setup_tufte_plot(axes[0], '', 'Accuracy', 'Wake Detection Accuracy by Model')
    axes[1].bar(range(len(model_names)), f1s, color=TufteColors.RED, alpha=0.85)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    setup_tufte_plot(axes[1], '', 'F1 Score', 'Wake Detection F1 Score by Model')
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: model_comparison.png')
    logger.info('2. Phase space comparison...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    wake_idx = np.where(labels == 1)[0][0]
    free_idx = np.where(labels == 0)[0][0]
    wake_window = windows[wake_idx]
    free_window = windows[free_idx]
    axes[0].scatter(wake_window['windspeed_80m'], wake_window['power'], alpha=0.6, s=20, color=TufteColors.RED, edgecolors='#8b0000', linewidth=0.5)
    setup_tufte_plot(axes[0], 'Wind Speed (m/s)', 'Power (MW)', 'Wake Condition (with hysteresis loops)')
    axes[1].scatter(free_window['windspeed_80m'], free_window['power'], alpha=0.6, s=20, color=TufteColors.BLACK, edgecolors='#1a1a1a', linewidth=0.5)
    setup_tufte_plot(axes[1], 'Wind Speed (m/s)', 'Power (MW)', 'Free-Stream Condition (clean curve)')
    plt.tight_layout()
    plt.savefig(out_dir / 'phase_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: phase_space_comparison.png')
    logger.info('3. Persistence diagrams comparison...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (window, label, ax) in enumerate([(wake_window, 'Wake', axes[0]), (free_window, 'Free-Stream', axes[1])]):
        power = window['power'].values
        windspeed = window['windspeed_80m'].values
        power_norm = (power - power.min()) / (power.max() - power.min() + 1e-08)
        wind_norm = (windspeed - windspeed.min()) / (windspeed.max() - windspeed.min() + 1e-08)
        X_phase = np.column_stack([power_norm, wind_norm])
        result = ripser(X_phase, maxdim=1)
        diagrams = result['dgms']
        colors = [TufteColors.GRAY, TufteColors.RED]
        labels_dim = ['H0 (Components)', 'H1 (Loops)']
        for dim in range(2):
            dgm = diagrams[dim]
            finite_dgm = dgm[dgm[:, 1] != np.inf]
            if len(finite_dgm) > 0:
                ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], alpha=0.7, s=50, color=colors[dim], label=labels_dim[dim], edgecolors=TufteColors.BLACK, linewidth=0.5)
        max_val = max([dgm[dgm[:, 1] != np.inf][:, 1].max() if len(dgm[dgm[:, 1] != np.inf]) > 0 else 1 for dgm in diagrams])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
        setup_tufte_plot(ax, 'Birth', 'Death', f'Persistence Diagram: {label}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'persistence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: persistence_comparison.png')
    logger.info('4. Feature importance...')
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices], color=TufteColors.BLACK, alpha=0.85)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
        setup_tufte_plot(ax, '', 'Importance', 'Top 10 Feature Importances (Random Forest)')
        plt.tight_layout()
        plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'  Saved: feature_importance.png')
    logger.info('5. Confusion matrix for best model...')
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_model_name]
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix: {best_model_name}', fontweight='normal')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Free-Stream', 'Wake'])
    plt.yticks(tick_marks, ['Free-Stream', 'Wake'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=20, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: confusion_matrix.png')
    logger.info('6. H1 feature distribution...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    h1_features = ['H1_count', 'H1_max_lifetime', 'H1_sum_lifetime', 'H1_mean_birth']
    for ax, feature in zip(axes.flatten(), h1_features):
        wake_values = X[y == 1][feature]
        free_values = X[y == 0][feature]
        ax.hist(free_values, bins=20, alpha=0.6, label='Free-Stream', color=TufteColors.GRAY, edgecolor=TufteColors.BLACK, linewidth=0.8)
        ax.hist(wake_values, bins=20, alpha=0.7, label='Wake', color=TufteColors.RED, edgecolor='#8b0000', linewidth=0.8)
        setup_tufte_plot(ax, feature, 'Count', f'Distribution: {feature}')
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / 'h1_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: h1_feature_distributions.png')
    logger.info('\nAll visualizations generated successfully!')

def main() -> None:
    """Main execution function."""Main execution function."""
    logger.info('=' * 60)
    logger.info('WAKE DETECTION USING PERSISTENT HOMOLOGY')
    logger.info('=' * 60)
    df = fetch_nrel_wind_data(lat=41.5, lon=-100.5, years=[2010, 2011, 2012])
    windows, labels = create_wake_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    out_dir = Path(__file__).parent / 'figures_wake'
    generate_visualizations(windows, labels, X, y, results, X_test, y_test, out_dir)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print_classification_summary(results, best_model_name)
    logger.info('\nClassification Report:')
    best_result = results[best_model_name]
    logger.info(classification_report(best_result['y_test'], best_result['y_pred'], target_names=['Free-Stream', 'Wake']))
    logger.info(f'\nVisualizations saved to: {out_dir}/')
    logger.info('\nAnalysis complete!')
if __name__ == '__main__':
    main()