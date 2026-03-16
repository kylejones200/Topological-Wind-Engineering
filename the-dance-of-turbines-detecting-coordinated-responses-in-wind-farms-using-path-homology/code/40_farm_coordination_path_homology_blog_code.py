"""
Wind Farm Coordination Pattern Detection Using Persistent Path Homology.

Classifies coordination types from lead-lag network topology extracted from
wind farm power output time series.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import requests
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
warnings.filterwarnings('ignore')

# Constants
RANDOM_SEED = 42
NREL_API_URL = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv'
REQUEST_TIMEOUT_SECONDS = 120

# Wind farm simulation constants
DEFAULT_N_TURBINES = 20
DEFAULT_GRID_SIZE = (4, 5)
DEFAULT_SPACING_M = 500
WAKE_DEFICIT_BASE = 0.3
WAKE_INFLUENCE_DISTANCE_MULTIPLIER = 3
WIND_DIRECTION_TOLERANCE_DEG = 20
WIND_NOISE_STD = 0.5

# Power curve constants
CUT_IN_WIND_SPEED_MPS = 3.0
RATED_WIND_SPEED_MPS = 12.0
RATED_POWER_KW = 2000
MAX_POWER_KW = 2200
POWER_CURVE_EXPONENT = 2.5
POWER_INERTIA_COEFF = 0.7
POWER_RESPONSE_COEFF = 0.3
POWER_NOISE_STD = 10.0

# Coordination event constants
GRID_EVENT_FREQUENCY = 20
OSCILLATORY_EVENT_FREQUENCY = 30
GRID_EVENT_MAGNITUDE_RANGE = (50, 200)
GRID_EVENT_DURATION_HOURS = 30
OSCILLATORY_TURBINE_SUBSET_SIZE = 5
OSCILLATORY_DURATION_SAMPLES = 60
OSCILLATORY_AMPLITUDE = 100
OSCILLATORY_PERIOD_SAMPLES = 20

# Network analysis constants
MAX_LAG_SAMPLES = 5
MIN_CORRELATION_THRESHOLD = 0.3
WINDOW_SIZE_SAMPLES = 60
LABEL_CONSENSUS_THRESHOLD = 0.6

# Model constants
TRAIN_TEST_SPLIT_RATIO = 0.7
N_ESTIMATORS = 200
MAX_DEPTH = 15
TOP_FEATURES_COUNT = 15

# Visualization constants
FIGURE_DPI = 300
CONFUSION_MATRIX_SIZE = (8, 8)
FEATURE_IMPORTANCE_SIZE = (10, 8)


def get_nrel_api_key() -> str:
    """
    Get NREL API key from environment variable.
    
    Returns:
        API key string.
    
    Raises:
        ValueError: If API key is not found in environment.
    """
    api_key = os.getenv('NREL_API_KEY')
    if api_key is None:
        raise ValueError(
            "NREL_API_KEY environment variable not set. "
            "Please set it before running the script."
        )
    return api_key


def fetch_nrel_wind_data(
    lat: float = 41.5,
    lon: float = -93.5,
    years: List[int] = [2017]
) -> Optional[pd.DataFrame]:
    """
    Fetch wind data from NREL Wind Toolkit API.
    
    Args:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        years: List of years to fetch (must be 2015-2023).
    
    Returns:
        DataFrame with wind data or None if fetch fails.
    
    Raises:
        ValueError: If API key is missing or years are invalid.
    """
    api_key = get_nrel_api_key()
    all_data = []
    
    for year in years:
        if year < 2015 or year > 2023:
            logger.warning(f"Year {year} out of valid range (2015-2023), skipping")
            continue
        
        logger.info(f"Fetching year {year}...")
        params = {
            'api_key': api_key,
            'wkt': f'POINT({lon} {lat})',
            'attributes': 'windspeed_100m,winddirection_100m',
            'years': str(year),
            'utc': 'true',
            'leap_day': 'false',
            'interval': '60',
            'email': 'kyletjones@gmail.com'
        }
        
        try:
            response = requests.get(
                NREL_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            data_start = next(
                (i + 1 for i, line in enumerate(lines) if line.startswith('Year,')),
                None
            )
            
            if data_start is None:
                logger.error(f"No header found for year {year}")
                continue
            
            data_text = '\n'.join(lines[data_start:])
            df_year = pd.read_csv(
                StringIO(data_text),
                header=None,
                names=[
                    'Year', 'Month', 'Day', 'Hour', 'Minute',
                    'windspeed_100m', 'winddirection_100m'
                ]
            )
            
            # Create timestamp
            df_year['time'] = pd.to_datetime(
                df_year[['Year', 'Month', 'Day', 'Hour', 'Minute']],
                format='mixed',
                errors='coerce'
            )
            df_year = df_year.dropna(subset=['time'])
            
            all_data.append(df_year)
            logger.info(f"Fetched {len(df_year):,} records for year {year}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching year {year}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error for year {year}: {e}")
            continue
    
    if not all_data:
        logger.error("Failed to fetch any data from NREL API")
        return None
    
    df = pd.concat(all_data, ignore_index=True).sort_values('time')
    logger.info(f"Total records fetched: {len(df):,}")
    return df


def calculate_power_output(wind_speed: float) -> float:
    """
    Calculate turbine power output from wind speed using power curve.
    
    Args:
        wind_speed: Wind speed in m/s.
    
    Returns:
        Power output in kW.
    """
    if wind_speed < CUT_IN_WIND_SPEED_MPS:
        return 0.0
    
    if wind_speed >= RATED_WIND_SPEED_MPS:
        return RATED_POWER_KW
    
    # Cubic power curve between cut-in and rated
    power_ratio = (wind_speed - CUT_IN_WIND_SPEED_MPS) / (
        RATED_WIND_SPEED_MPS - CUT_IN_WIND_SPEED_MPS
    )
    return RATED_POWER_KW * (power_ratio ** POWER_CURVE_EXPONENT)


def apply_wake_effects(
    positions: List[Tuple[float, float]],
    base_wind: float,
    wind_direction: float
) -> np.ndarray:
    """
    Calculate wake-affected wind speeds for all turbines.
    
    Uses vectorized operations to compute wake deficits efficiently.
    
    Args:
        positions: List of (x, y) turbine positions.
        base_wind: Base wind speed in m/s.
        wind_direction: Wind direction in degrees.
    
    Returns:
        Array of local wind speeds for each turbine.
    """
    n_turbines = len(positions)
    positions_array = np.array(positions)
    local_winds = np.full(n_turbines, base_wind)
    
    # Vectorized distance and angle calculations
    for i in range(n_turbines):
        pos_i = positions_array[i]
        other_positions = positions_array
        
        # Vectorized differences
        dx = pos_i[0] - other_positions[:, 0]
        dy = pos_i[1] - other_positions[:, 1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        
        # Vectorized angle calculations
        angles_to_others = np.degrees(np.arctan2(dy, dx))
        wind_dir_relative = (wind_direction - angles_to_others + 180) % 360 - 180
        
        # Vectorized wake condition check
        in_wake = (
            (np.abs(wind_dir_relative) < WIND_DIRECTION_TOLERANCE_DEG) &
            (distances < DEFAULT_SPACING_M * WAKE_INFLUENCE_DISTANCE_MULTIPLIER) &
            (distances > 0)  # Exclude self
        )
        
        if np.any(in_wake):
            wake_distances = distances[in_wake]
            wake_deficits = WAKE_DEFICIT_BASE * np.exp(
                -wake_distances / DEFAULT_SPACING_M
            )
            local_winds[i] *= (1 - np.max(wake_deficits))
    
    return local_winds


def simulate_wind_farm(
    wind_data: pd.DataFrame,
    n_turbines: int = DEFAULT_N_TURBINES,
    grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
    spacing_m: int = DEFAULT_SPACING_M,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Simulate wind farm with coordinated responses.
    
    Creates wake propagation patterns from wind field interactions.
    
    Args:
        wind_data: DataFrame with wind speed and direction.
        n_turbines: Number of turbines in farm.
        grid_size: Grid dimensions (rows, cols).
        spacing_m: Spacing between turbines in meters.
        random_seed: Optional random seed for reproducibility.
    
    Returns:
        Tuple of (power_outputs array, positions list).
    
    Raises:
        ValueError: If grid_size doesn't match n_turbines.
    """
    if grid_size[0] * grid_size[1] != n_turbines:
        raise ValueError(
            f"Grid size {grid_size} does not match {n_turbines} turbines"
        )
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate turbine positions
    positions = [
        (i * spacing_m, j * spacing_m)
        for i in range(grid_size[0])
        for j in range(grid_size[1])
    ]
    
    n = len(wind_data)
    power_outputs = np.zeros((n, n_turbines))
    wind_speed = wind_data['windspeed_100m'].values
    wind_dir = wind_data['winddirection_100m'].values
    
    # Initialize first timestep
    local_winds = apply_wake_effects(positions, wind_speed[0], wind_dir[0])
    power_outputs[0, :] = np.array([
        calculate_power_output(w) for w in local_winds
    ])
    
    # Simulate time series
    for t in range(1, n):
        # Add noise to base wind
        base_wind = wind_speed[t] + np.random.randn() * WIND_NOISE_STD
        wdir = wind_dir[t]
        
        # Calculate wake-affected winds
        local_winds = apply_wake_effects(positions, base_wind, wdir)
        
        # Calculate target power for each turbine
        target_powers = np.array([
            calculate_power_output(w) for w in local_winds
        ])
        
        # Apply inertia (exponential smoothing)
        power_outputs[t, :] = (
            POWER_INERTIA_COEFF * power_outputs[t - 1, :] +
            POWER_RESPONSE_COEFF * target_powers
        )
        
        # Add noise and clip
        power_outputs[t, :] += np.random.randn(n_turbines) * POWER_NOISE_STD
        power_outputs[t, :] = np.clip(
            power_outputs[t, :], 0, MAX_POWER_KW
        )
    
    return power_outputs, positions


def inject_coordination_events(
    power_outputs: np.ndarray,
    positions: List[Tuple[float, float]],
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject specific coordination patterns into power outputs.
    
    Creates grid events (simultaneous responses) and oscillatory
    instabilities (control-induced feedback loops).
    
    Args:
        power_outputs: Power output array (time x turbines).
        positions: Turbine positions (not used but kept for API consistency).
        random_seed: Optional random seed for reproducibility.
    
    Returns:
        Tuple of (modified power_outputs, labels array).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n, n_turbines = power_outputs.shape
    labels = np.zeros(n, dtype=int)
    
    # Inject grid events (label 1)
    n_grid_events = n // GRID_EVENT_FREQUENCY
    grid_events = np.random.choice(n, size=n_grid_events, replace=False)
    
    for event_t in grid_events:
        magnitude = np.random.uniform(*GRID_EVENT_MAGNITUDE_RANGE)
        event_magnitudes = magnitude * np.random.uniform(
            0.8, 1.2, size=n_turbines
        )
        power_outputs[event_t, :] += event_magnitudes
        
        # Label surrounding time window
        start_t = max(0, event_t - GRID_EVENT_DURATION_HOURS)
        end_t = min(n, event_t + GRID_EVENT_DURATION_HOURS)
        labels[start_t:end_t] = 1
    
    # Inject oscillatory events (label 2)
    n_osc_events = n // OSCILLATORY_EVENT_FREQUENCY
    osc_events = np.random.choice(n, size=n_osc_events, replace=False)
    
    for event_t in osc_events:
        turb_subset = np.random.choice(
            n_turbines, size=OSCILLATORY_TURBINE_SUBSET_SIZE, replace=False
        )
        
        for offset in range(OSCILLATORY_DURATION_SAMPLES):
            t = event_t + offset
            if t >= n:
                break
            
            phases = np.arange(len(turb_subset)) * 2 * np.pi / len(turb_subset)
            oscillations = OSCILLATORY_AMPLITUDE * np.sin(
                2 * np.pi * offset / OSCILLATORY_PERIOD_SAMPLES + phases
            )
            power_outputs[t, turb_subset] += oscillations
            labels[t] = 2
    
    return power_outputs, labels


def compute_lead_lag_network(
    power_window: np.ndarray,
    max_lag: int = MAX_LAG_SAMPLES
) -> nx.DiGraph:
    """
    Compute directed graph based on lead-lag correlations.
    
    Args:
        power_window: Power outputs for multiple turbines (time x turbines).
        max_lag: Maximum lag to consider (in samples).
    
    Returns:
        Directed graph (NetworkX DiGraph) with edge weights and lags.
    
    Raises:
        ValueError: If power_window has invalid shape.
    """
    if power_window.ndim != 2:
        raise ValueError("power_window must be 2D array (time x turbines)")
    
    n_turbines = power_window.shape[1]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_turbines))
    
    # Normalize signals
    normalized = (
        power_window - power_window.mean(axis=0, keepdims=True)
    ) / (power_window.std(axis=0, keepdims=True) + 1e-10)
    
    # Compute correlations for all pairs
    for i in range(n_turbines):
        sig_i = normalized[:, i]
        
        for j in range(n_turbines):
            if i == j:
                continue
            
            sig_j = normalized[:, j]
            corrs = []
            
            # Test all lags
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    c = np.corrcoef(sig_i[:lag], sig_j[-lag:])[0, 1]
                elif lag > 0:
                    c = np.corrcoef(sig_i[lag:], sig_j[:-lag])[0, 1]
                else:
                    c = np.corrcoef(sig_i, sig_j)[0, 1]
                
                if not np.isnan(c):
                    corrs.append((lag, c))
            
            if corrs:
                best_lag, best_corr = max(corrs, key=lambda x: abs(x[1]))
                
                if best_lag > 0 and abs(best_corr) > MIN_CORRELATION_THRESHOLD:
                    G.add_edge(
                        i, j,
                        weight=abs(best_corr),
                        lag=best_lag
                    )
    
    return G


def compute_path_homology_features(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute features from directed graph that approximate path homology.
    
    Uses directed cycle counts, path length distributions, and graph
    centrality measures as approximations of path homology.
    
    Args:
        G: Directed graph (NetworkX DiGraph).
    
    Returns:
        Dictionary of feature names and values.
    """
    features = {}
    
    # Cycle features
    try:
        cycles = list(nx.simple_cycles(G))
        features['n_cycles'] = len(cycles)
        
        if cycles:
            cycle_lengths = np.array([len(c) for c in cycles])
            features['max_cycle_length'] = float(np.max(cycle_lengths))
            features['mean_cycle_length'] = float(np.mean(cycle_lengths))
            features['min_cycle_length'] = float(np.min(cycle_lengths))
        else:
            features['max_cycle_length'] = 0.0
            features['mean_cycle_length'] = 0.0
            features['min_cycle_length'] = 0.0
    except (nx.NetworkXError, ValueError) as e:
        logger.debug(f"Error computing cycles: {e}")
        features['n_cycles'] = 0
        features['max_cycle_length'] = 0.0
        features['mean_cycle_length'] = 0.0
        features['min_cycle_length'] = 0.0
    
    # Connected component features
    scc = list(nx.strongly_connected_components(G))
    features['n_strongly_connected'] = len(scc)
    features['largest_scc_size'] = (
        max([len(c) for c in scc]) if scc else 0
    )
    
    wcc = list(nx.weakly_connected_components(G))
    features['n_weakly_connected'] = len(wcc)
    
    # Basic graph metrics
    features['n_edges'] = G.number_of_edges()
    features['n_nodes'] = G.number_of_nodes()
    features['edge_density'] = (
        nx.density(G) if G.number_of_nodes() > 0 else 0.0
    )
    
    # Degree features
    if G.number_of_edges() > 0:
        in_degrees = np.array(list(dict(G.in_degree()).values()))
        out_degrees = np.array(list(dict(G.out_degree()).values()))
        
        features['max_in_degree'] = float(np.max(in_degrees))
        features['max_out_degree'] = float(np.max(out_degrees))
        features['mean_in_degree'] = float(np.mean(in_degrees))
        features['mean_out_degree'] = float(np.mean(out_degrees))
        
        # Bidirectional ratio
        bidirectional = sum(
            1 for u, v in G.edges() if G.has_edge(v, u)
        )
        features['bidirectional_ratio'] = (
            bidirectional / G.number_of_edges()
        )
    else:
        features['max_in_degree'] = 0.0
        features['max_out_degree'] = 0.0
        features['mean_in_degree'] = 0.0
        features['mean_out_degree'] = 0.0
        features['bidirectional_ratio'] = 0.0
    
    # Edge weight features
    if G.number_of_edges() > 0:
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        features['mean_edge_weight'] = float(np.mean(weights))
        features['max_edge_weight'] = float(np.max(weights))
        features['std_edge_weight'] = float(np.std(weights))
    else:
        features['mean_edge_weight'] = 0.0
        features['max_edge_weight'] = 0.0
        features['std_edge_weight'] = 0.0
    
    return features


def create_dataset(
    power_outputs: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE_SAMPLES
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create dataset with path homology features from sliding windows.
    
    Args:
        power_outputs: Power output array (time x turbines).
        labels: Label array (time).
        window_size: Size of sliding window in samples.
    
    Returns:
        Tuple of (feature matrix, labels, feature names).
    
    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    if power_outputs.shape[0] != len(labels):
        raise ValueError(
            "power_outputs and labels must have same length"
        )
    
    logger.info("Creating dataset...")
    all_features = []
    all_labels = []
    n, n_turbines = power_outputs.shape
    
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        power_window = power_outputs[start:end, :]
        label_window = labels[start:end]
        
        # Determine consensus label
        label_counts = np.bincount(label_window.astype(int))
        label = int(np.argmax(label_counts))
        consensus_ratio = label_counts.max() / len(label_window)
        
        if consensus_ratio < LABEL_CONSENSUS_THRESHOLD:
            continue
        
        G = compute_lead_lag_network(power_window, max_lag=MAX_LAG_SAMPLES)
        
        try:
            features = compute_path_homology_features(G)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            logger.debug(f"Error computing features for window {start}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid features extracted from windows")
    
    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)
    
    logger.info(f"Total windows: {len(X)}")
    logger.info(f"Wake (0): {(y == 0).sum()}")
    logger.info(f"Grid (1): {(y == 1).sum()}")
    logger.info(f"Oscillatory (2): {(y == 2).sum()}")
    
    return X, y, feature_names


def visualize_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: np.ndarray,
    feature_names: List[str],
    out_dir: Path
) -> None:
    """
    Generate visualizations for classification results.
    
    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        feature_importance: Feature importance scores.
        feature_names: List of feature names.
        out_dir: Output directory for figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', square=True,
        xticklabels=['Wake', 'Grid', 'Oscillatory'],
        yticklabels=['Wake', 'Grid', 'Oscillatory'],
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(
        'Confusion Matrix: Coordination Classification',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    confusion_path = out_dir / 'confusion_matrix.png'
    plt.savefig(confusion_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {confusion_path}")
    
    # Feature importance
    indices = np.argsort(feature_importance)[-TOP_FEATURES_COUNT:]
    fig, ax = plt.subplots(figsize=FEATURE_IMPORTANCE_SIZE)
    ax.barh(
        range(TOP_FEATURES_COUNT),
        feature_importance[indices],
        color='steelblue',
        alpha=0.8
    )
    ax.set_yticks(range(TOP_FEATURES_COUNT))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(
        f'Top {TOP_FEATURES_COUNT} Features: Coordination Classification',
        fontsize=12, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    importance_path = out_dir / 'feature_importance.png'
    plt.savefig(importance_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance to {importance_path}")


def main() -> None:
    """
    Main execution function for wind farm coordination detection.
    
    Fetches data, simulates wind farm, extracts features, trains model,
    and generates visualizations.
    """
    np.random.seed(RANDOM_SEED)
    
    logger.info('=' * 70)
    logger.info('Wind Farm Coordination Detection via Path Homology')
    logger.info('=' * 70)
    
    # Fetch data
    logger.info('\n1. Fetching NREL wind data...')
    wind_data = fetch_nrel_wind_data(lat=41.5, lon=-93.5, years=[2017, 2018])
    if wind_data is None:
        logger.error('Failed to fetch data')
        return
    
    logger.info(f'Total records: {len(wind_data):,}')
    
    # Simulate wind farm
    logger.info('\n2. Simulating 20-turbine wind farm...')
    power_outputs, positions = simulate_wind_farm(
        wind_data,
        n_turbines=DEFAULT_N_TURBINES,
        grid_size=DEFAULT_GRID_SIZE,
        spacing_m=DEFAULT_SPACING_M,
        random_seed=RANDOM_SEED
    )
    logger.info(f'Simulated {power_outputs.shape[0]:,} timesteps')
    
    # Inject coordination events
    logger.info('\n3. Injecting coordination patterns...')
    power_outputs, labels = inject_coordination_events(
        power_outputs, positions, random_seed=RANDOM_SEED
    )
    label_counts = pd.Series(labels).value_counts().sort_index()
    logger.info(
        f"Wake propagation: {label_counts.get(0, 0)} "
        f"({label_counts.get(0, 0) / len(labels) * 100:.1f}%)"
    )
    logger.info(
        f"Grid events: {label_counts.get(1, 0)} "
        f"({label_counts.get(1, 0) / len(labels) * 100:.1f}%)"
    )
    logger.info(
        f"Oscillatory: {label_counts.get(2, 0)} "
        f"({label_counts.get(2, 0) / len(labels) * 100:.1f}%)"
    )
    
    # Extract features
    logger.info('\n4. Extracting path homology features...')
    X, y, feature_names = create_dataset(
        power_outputs, labels, window_size=WINDOW_SIZE_SAMPLES
    )
    
    # Split data
    logger.info('\n5. Splitting data...')
    split_idx = int(TRAIN_TEST_SPLIT_RATIO * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info('\n6. Training Random Forest...')
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    logger.info(f'\nAccuracy: {acc * 100:.2f}%')
    logger.info(
        f"\n{classification_report(y_test, y_pred, target_names=['Wake', 'Grid', 'Oscillatory'])}"
    )
    
    # Feature importance
    feature_importance = clf.feature_importances_
    top_features = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    logger.info('\nTop 10 features:')
    for fname, imp in top_features:
        logger.info(f'  {fname}: {imp:.4f}')
    
    # Visualize
    logger.info('\n7. Generating visualizations...')
    visualize_results(
        y_test, y_pred, feature_importance, feature_names,
        Path('figures_coordination')
    )
    
    # Summary
    logger.info('\n' + '=' * 70)
    logger.info('WIND FARM COORDINATION DETECTION COMPLETE')
    logger.info('=' * 70)
    logger.info(f'\nPath homology classification: {acc * 100:.1f}% accuracy')
    logger.info('Detects coordination patterns:')
    logger.info('  - Wake propagation: Directed chains aligned with wind')
    logger.info('  - Grid events: Star patterns from simultaneous responses')
    logger.info('  - Oscillatory instabilities: Directed cycles (feedback loops)')
    logger.info('\nKey features:')
    logger.info('  - Directed cycle counts (feedback structure)')
    logger.info('  - Strongly connected components (directed paths)')
    logger.info('  - Bidirectional ratio (simultaneity vs propagation)')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
