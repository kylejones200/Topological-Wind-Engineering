"""
Wind Farm Coordination Pattern Detection Using Persistent Path Homology
Classifies wake propagation, grid events, and oscillatory instabilities using
directed topology on turbine lead-lag networks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from ripser import ripser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from pathlib import Path
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(42)

def fetch_nrel_wind_data(lat=41.5, lon=-100.5, years=[2010, 2011, 2012]):
    """Simulate NREL Wind Toolkit data fetch."""
    logger.info(f'Simulating NREL wind data fetch for location ({lat}, {lon})')
    n_records = 365 * 24 * 12 * len(years)
    timestamps = pd.date_range(start=f'{years[0]}-01-01', periods=n_records, freq='5min')
    hours = np.array([t.hour + t.minute / 60 for t in timestamps])
    days = np.array([t.dayofyear for t in timestamps])
    seasonal = 2 * np.sin(2 * np.pi * days / 365)
    diurnal = 1.5 * np.sin(2 * np.pi * hours / 24)
    windspeed = 8.5 + seasonal + diurnal + np.random.normal(0, 2, n_records)
    windspeed = np.clip(windspeed, 0, 25)
    df = pd.DataFrame({'timestamp': timestamps, 'windspeed': windspeed})
    logger.info(f'Fetched {len(df)} records spanning {len(years)} years')
    return df

def simulate_turbine_power_curve(windspeed, rated_power=2.0):
    """Standard turbine power curve."""Standard turbine power curve."""
    cut_in = 3.0
    rated_speed = 12.0
    cut_out = 25.0
    power = np.zeros_like(windspeed)
    for i, ws in enumerate(windspeed):
        if ws < cut_in or ws > cut_out:
            power[i] = 0
        elif ws < rated_speed:
            power[i] = rated_power * ((ws - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            power[i] = rated_power
        power[i] += np.random.normal(0, 0.03 * rated_power)
        power[i] = max(0, power[i])
    return power

def simulate_wake_propagation(df, n_turbines=6):
    """
    Simulate wake propagation pattern.
    Turbines arranged in line, upstream affects downstream with lag.
    """
    windspeed = df['windspeed'].values
    turbine_powers = []
    for t in range(n_turbines):
        if t == 0:
            power = simulate_turbine_power_curve(windspeed)
        else:
            wake_factor = 0.85 ** t
            reduced_wind = windspeed * wake_factor
            lag_steps = t * 3
            if lag_steps < len(reduced_wind):
                lagged_wind = np.roll(reduced_wind, lag_steps)
                lagged_wind[:lag_steps] = reduced_wind[:lag_steps]
                power = simulate_turbine_power_curve(lagged_wind)
            else:
                power = simulate_turbine_power_curve(reduced_wind)
        turbine_powers.append(power)
    return np.array(turbine_powers).T

def simulate_grid_event(df, n_turbines=6):
    """
    Simulate grid event pattern.
    All turbines respond simultaneously to grid frequency change.
    """
    windspeed = df['windspeed'].values
    base_power = simulate_turbine_power_curve(windspeed)
    turbine_powers = []
    for t in range(n_turbines):
        spatial_factor = 1.0 + np.random.uniform(-0.05, 0.05)
        power = base_power * spatial_factor
        turbine_powers.append(power)
    return np.array(turbine_powers).T

def simulate_oscillatory_pattern(df, n_turbines=6):
    """
    Simulate oscillatory feedback pattern.
    Turbines influence each other in cycles, creating feedback loops.
    """
    windspeed = df['windspeed'].values
    n_steps = len(windspeed)
    turbine_powers = np.zeros((n_steps, n_turbines))
    for t in range(n_turbines):
        turbine_powers[:, t] = simulate_turbine_power_curve(windspeed)
    oscillation_freq = 0.05
    phase_shifts = np.linspace(0, 2 * np.pi, n_turbines, endpoint=False)
    for i in range(n_steps):
        oscillation = 0.15 * np.sin(2 * np.pi * oscillation_freq * i + phase_shifts)
        turbine_powers[i, :] *= 1 + oscillation
        turbine_powers[i, :] = np.clip(turbine_powers[i, :], 0, 2.0)
    return turbine_powers

def create_coordination_scenarios(df, n_windows=120, window_size=288):
    """
    Create labeled windows of coordination patterns.
    0 = wake propagation, 1 = grid event, 2 = oscillatory
    """
    logger.info(f'\nCreating {n_windows} labeled windows (wake/grid/oscillatory)...')
    windows = []
    labels = []
    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)
    n_per_class = n_windows // 3
    for idx, start in enumerate(starts):
        window_df = df.iloc[start:start + window_size].copy()
        if idx < n_per_class:
            pattern = 0
            turbine_powers = simulate_wake_propagation(window_df, n_turbines=6)
        elif idx < 2 * n_per_class:
            pattern = 1
            turbine_powers = simulate_grid_event(window_df, n_turbines=6)
        else:
            pattern = 2
            turbine_powers = simulate_oscillatory_pattern(window_df, n_turbines=6)
        for t in range(6):
            window_df[f'turbine_{t}'] = turbine_powers[:, t]
        windows.append(window_df)
        labels.append(pattern)
    labels = np.array(labels)
    label_counts = np.bincount(labels, minlength=3)
    logger.info(f'Created {label_counts[0]} wake, {label_counts[1]} grid, {label_counts[2]} oscillatory windows')
    return (windows, labels)

def compute_lead_lag_network(window_df, n_turbines=6, max_lag=10):
    """
    Compute directed lead-lag network from cross-correlations.
    Directed edge i→j if turbine i leads turbine j.
    """
    powers = [window_df[f'turbine_{t}'].values for t in range(n_turbines)]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_turbines))
    edge_weights = []
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i == j:
                continue
            max_corr = 0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                if lag < len(powers[i]):
                    corr = np.corrcoef(powers[i][:-lag], powers[j][lag:])[0, 1]
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        best_lag = lag
            if max_corr > 0.3:
                G.add_edge(i, j, weight=max_corr, lag=best_lag)
                edge_weights.append(max_corr)
    return (G, edge_weights)

def compute_path_homology_features(G, n_turbines=6):
    """
    Compute features from directed graph structure.
    Focus on cycles (feedback loops) and paths.
    """
    features = {}
    features['n_edges'] = G.number_of_edges()
    features['density'] = nx.density(G)
    try:
        cycles = list(nx.simple_cycles(G))
        features['n_cycles'] = len(cycles)
        if len(cycles) > 0:
            cycle_lengths = [len(c) for c in cycles]
            features['max_cycle_length'] = max(cycle_lengths)
            features['mean_cycle_length'] = np.mean(cycle_lengths)
        else:
            features['max_cycle_length'] = 0
            features['mean_cycle_length'] = 0
    except:
        features['n_cycles'] = 0
        features['max_cycle_length'] = 0
        features['mean_cycle_length'] = 0
    try:
        if nx.is_strongly_connected(G):
            features['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            if len(largest_scc) > 1:
                subG = G.subgraph(largest_scc)
                features['avg_path_length'] = nx.average_shortest_path_length(subG)
            else:
                features['avg_path_length'] = 0
    except:
        features['avg_path_length'] = 0
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    features['mean_in_degree'] = np.mean(in_degrees)
    features['mean_out_degree'] = np.mean(out_degrees)
    features['std_in_degree'] = np.std(in_degrees)
    features['std_out_degree'] = np.std(out_degrees)
    features['max_in_degree'] = max(in_degrees) if len(in_degrees) > 0 else 0
    features['max_out_degree'] = max(out_degrees) if len(out_degrees) > 0 else 0
    try:
        features['reciprocity'] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0
    except:
        features['reciprocity'] = 0
    features['n_sources'] = sum((1 for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0))
    features['n_sinks'] = sum((1 for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0))
    return features

def extract_all_features(windows, labels):
    """Extract path homology and directed graph features."""
    logger.info('\nExtracting path homology features from directed networks...')
    feature_list = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info(f'  Processing window {i + 1}/{len(windows)}')
        G, edge_weights = compute_lead_lag_network(window_df, n_turbines=6, max_lag=10)
        features = compute_path_homology_features(G, n_turbines=6)
        for t in range(6):
            features[f'turbine_{t}_mean'] = window_df[f'turbine_{t}'].mean()
            features[f'turbine_{t}_std'] = window_df[f'turbine_{t}'].std()
        feature_list.append(features)
    X = pd.DataFrame(feature_list)
    y = labels
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    logger.info(f'\nFeature matrix: {X.shape}')
    label_counts = np.bincount(y, minlength=3)
    logger.info(f'Label distribution: Wake={label_counts[0]}, Grid={label_counts[1]}, Oscillatory={label_counts[2]}')
    return (X, y)

def train_and_evaluate_models(X, y):
    """Train and evaluate classifiers."""Train and evaluate classifiers."""
    logger.info('\n' + '=' * 60)
    logger.info('TRAINING AND EVALUATION')
    logger.info('=' * 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logger.info(f'\nTrain set: {len(X_train)} samples')
    logger.info(f'Test set: {len(X_test)} samples')
    models = {'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial'), 'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True), 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)}
    results = {}
    for name, model in models.items():
        logger.info(f'\n{name}:')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        logger.info(f'  Accuracy: {acc:.3f}')
        logger.info(f'  F1 (macro): {f1_macro:.3f}')
        logger.info(f'  F1 (weighted): {f1_weighted:.3f}')
        results[name] = {'model': model, 'accuracy': acc, 'f1_macro': f1_macro, 'y_test': y_test, 'y_pred': y_pred}
    return results

def generate_visualizations(windows, labels, X, y, results, out_dir):
    """Generate comprehensive visualizations."""
    logger.info('\n' + '=' * 60)
    logger.info('GENERATING VISUALIZATIONS')
    logger.info('=' * 60)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info('\n1. Model comparison...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1s = [results[m]['f1_macro'] for m in model_names]
    axes[0].bar(range(len(model_names)), accuracies, color='#2b2b2b', alpha=0.85)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Coordination Pattern Classification Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].bar(range(len(model_names)), f1s, color='#d62728', alpha=0.85)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('F1 Score (macro)')
    axes[1].set_title('Coordination Pattern F1 Score')
    axes[1].set_ylim([0, 1])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: model_comparison.png')
    logger.info('2. Network structure examples...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pattern_names = ['Wake Propagation', 'Grid Event', 'Oscillatory']
    for pattern_idx, (ax, pattern_name) in enumerate(zip(axes, pattern_names)):
        window_idx = np.where(labels == pattern_idx)[0][0]
        window = windows[window_idx]
        G, _ = compute_lead_lag_network(window, n_turbines=6, max_lag=10)
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500, edgecolors='black', linewidths=2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, arrowsize=20, arrowstyle='->', width=2)
        ax.set_title(pattern_name)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'network_structures.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: network_structures.png')
    logger.info('3. Cycle count distribution...')
    plt.figure(figsize=(10, 6))
    for pattern_idx, (pattern_name, color) in enumerate([('Wake', 'blue'), ('Grid', 'green'), ('Oscillatory', 'red')]):
        values = X[y == pattern_idx]['n_cycles']
    plt.hist(values, bins=15, alpha=0.5, label=pattern_name, color=color, edgecolor='#2b2b2b')
    plt.xlabel('Number of Feedback Cycles')
    plt.ylabel('Count')
    plt.title('Feedback Cycle Distribution by Pattern Type')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / 'cycle_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f'  Saved: cycle_distribution.png')
    logger.info('4. Feature importance...')
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices], color='#2b2b2b', alpha=0.85)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance')
        ax.set_title('Top 10 Feature Importances (Random Forest)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'  Saved: feature_importance.png')
    logger.info('\nAll visualizations generated successfully!')

def main():
    """Main execution."""Main execution."""
    logger.info('=' * 60)
    logger.info('WIND FARM COORDINATION PATTERN DETECTION')
    logger.info('Persistent Path Homology on Directed Lead-Lag Networks')
    logger.info('=' * 60)
    df = fetch_nrel_wind_data()
    windows, labels = create_coordination_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results = train_and_evaluate_models(X, y)
    out_dir = Path(__file__).parent / 'figures_coordination'
    generate_visualizations(windows, labels, X, y, results, out_dir)
    logger.info('\n' + '=' * 60)
    logger.info('FINAL SUMMARY')
    logger.info('=' * 60)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_model_name]
    logger.info(f'\nBest Model: {best_model_name}')
    logger.info(f"  Accuracy: {best_result['accuracy']:.3f}")
    logger.info(f"  F1 (macro): {best_result['f1_macro']:.3f}")
    logger.info('\nClassification Report:')
    logger.info(classification_report(best_result['y_test'], best_result['y_pred'], target_names=['Wake Propagation', 'Grid Event', 'Oscillatory']))
    logger.info(f'\nVisualizations saved to: {out_dir}/')
    logger.info('\nAnalysis complete!')
if __name__ == '__main__':
    main()