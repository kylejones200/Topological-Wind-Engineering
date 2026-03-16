"""
Turbulence Intensity Classification Using Persistence Images and CNNs.
Run from repo root: python path/to/37_turbulence_persistence_images_blog_code.py [--config path/to/config.yaml]
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

import numpy as np
import pandas as pd
import os
import requests
from io import StringIO
from typing import Tuple, Optional, List, Dict, Any
from ripser import ripser
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, roc_curve
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import logging

from config.load import load_config

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Turbulence intensity constants
TI_DAY_MIN = 0.16
TI_DAY_MAX = 0.22
TI_NIGHT_MIN = 0.05
TI_NIGHT_MAX = 0.09
TI_NEUTRAL_MIN = 0.1
TI_NEUTRAL_MAX = 0.14
TI_LOW_THRESHOLD = 0.1
TI_HIGH_THRESHOLD = 0.15

# Time periods
DAY_START_HOUR = 8
DAY_END_HOUR = 16
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 4

# Turbine constants
RATED_POWER_KW = 2000
CUT_IN_WIND_SPEED_MPS = 3.0
RATED_WIND_SPEED_MPS = 12.0
POWER_CURVE_EXPONENT = 2.5
MIN_ROTOR_SPEED_RPM = 10
ROTOR_SPEED_SLOPE = 5
MIN_PITCH_DEG = 2
MAX_PITCH_DEG = 90
PITCH_AT_CUT_IN = 5
PITCH_SLOPE = 2
MAX_ROTOR_SPEED_RPM = 70
ROTOR_SPEED_ABOVE_RATED = 55
ROTOR_SPEED_SLOPE_ABOVE_RATED = 0.2

# Turbulence simulation constants
LARGE_SCALE_PERSISTENCE = 0.95
LARGE_SCALE_INNOVATION = 0.05
SMALL_SCALE_FACTOR = 0.3
ROTOR_INERTIA = 0.85
ROTOR_RESPONSE = 0.15
POWER_INERTIA = 0.75
POWER_RESPONSE = 0.25
PITCH_INERTIA = 0.9
PITCH_RESPONSE = 0.1
ROTOR_NOISE_STD = 0.5
POWER_NOISE_STD = 20.0
PITCH_NOISE_STD = 0.5
POWER_OVERRATING_FACTOR = 1.1

# Persistence image constants
DEFAULT_WINDOW_SIZE = 10
DEFAULT_RESOLUTION = 20
DEFAULT_SIGMA = 0.1
DIAGRAM_PADDING = 0.1

# Model constants
TRAIN_TEST_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5
DROPOUT_RATE = 0.5
LOG_INTERVAL = 5

# CNN architecture constants
INPUT_CHANNELS = 2
NUM_CLASSES = 2
CONV1_OUT_CHANNELS = 16
CONV2_OUT_CHANNELS = 32
FC1_SIZE = 64
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
FC1_INPUT_SIZE = 32 * 5 * 5

# Visualization constants
FIGURE_DPI = 300
ROC_CURVE_SIZE = (8, 8)


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
    cfg: Optional[Dict[str, Any]] = None,
    lat: float = 41.5,
    lon: float = -93.5,
    years: Optional[List[int]] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch wind data from NREL Wind Toolkit API.
    If cfg is provided, nrel.lat, nrel.lon, nrel.years, nrel.api_key, nrel.base_url are used.
    """
    if cfg is not None:
        nrel = cfg.get("nrel", {})
        lat = nrel.get("lat", 41.5)
        lon = nrel.get("lon", -93.5)
        years = nrel.get("years", [2017, 2018])
        api_key = nrel.get("api_key") or os.environ.get("NREL_API_KEY") or get_nrel_api_key()
        base_url = nrel.get("base_url", NREL_API_URL)
        timeout = nrel.get("request_timeout_seconds", REQUEST_TIMEOUT_SECONDS)
    else:
        years = years if years is not None else [2017]
        api_key = get_nrel_api_key()
        base_url = NREL_API_URL
        timeout = REQUEST_TIMEOUT_SECONDS
    all_data = []

    for year in years:
        if year < 2015 or year > 2023:
            logger.warning(f"Year {year} out of valid range (2015-2023), skipping")
            continue
        
        logger.info(f"Fetching year {year}...")
        params = {
            'api_key': api_key,
            'wkt': f'POINT({lon} {lat})',
            'attributes': 'windspeed_100m,temperature_100m',
            'years': str(year),
            'utc': 'true',
            'leap_day': 'false',
            'interval': '60',
            'email': 'kyletjones@gmail.com'
        }
        
        try:
            response = requests.get(
                base_url, params=params, timeout=timeout
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
                    'windspeed_100m', 'temperature_100m'
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


def calculate_turbulence_intensity(hour: np.ndarray) -> np.ndarray:
    """
    Calculate turbulence intensity based on time of day using vectorized operations.
    
    TI varies with atmospheric stability:
    - Day (unstable): TI = 0.16-0.22
    - Night (stable): TI = 0.05-0.09
    - Neutral: TI = 0.10-0.14
    
    Args:
        hour: Array of hour values (0-23).
    
    Returns:
        Array of turbulence intensity values.
    """
    ti = np.zeros_like(hour, dtype=float)
    
    # Vectorized condition checks
    day_mask = (hour >= DAY_START_HOUR) & (hour < DAY_END_HOUR)
    night_mask = (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)
    neutral_mask = ~(day_mask | night_mask)
    
    # Vectorized random generation
    n = len(hour)
    ti[day_mask] = np.random.uniform(
        TI_DAY_MIN, TI_DAY_MAX, size=day_mask.sum()
    )
    ti[night_mask] = np.random.uniform(
        TI_NIGHT_MIN, TI_NIGHT_MAX, size=night_mask.sum()
    )
    ti[neutral_mask] = np.random.uniform(
        TI_NEUTRAL_MIN, TI_NEUTRAL_MAX, size=neutral_mask.sum()
    )
    
    return ti


def calculate_turbulent_wind(
    wind_mean: np.ndarray,
    ti: np.ndarray
) -> np.ndarray:
    """
    Calculate turbulent wind speed using vectorized operations.
    
    Args:
        wind_mean: Mean wind speed array.
        ti: Turbulence intensity array.
    
    Returns:
        Turbulent wind speed array.
    """
    n = len(wind_mean)
    wind_turbulent = np.zeros(n)
    
    # Initialize first value
    wind_turbulent[0] = wind_mean[0]
    
    # Vectorized calculation for remaining values
    large_scale_innovations = np.random.randn(n - 1) * wind_mean[1:] * ti[1:]
    small_scale_noise = np.random.randn(n - 1) * wind_mean[1:] * ti[1:] * SMALL_SCALE_FACTOR
    
    # Recursive calculation (can't fully vectorize due to dependency)
    for i in range(1, n):
        large_scale = (
            LARGE_SCALE_PERSISTENCE * wind_turbulent[i - 1] +
            LARGE_SCALE_INNOVATION * large_scale_innovations[i - 1]
        )
        wind_turbulent[i] = wind_mean[i] + large_scale + small_scale_noise[i - 1]
    
    return np.maximum(wind_turbulent, 0)


def calculate_turbine_response(
    wind_turbulent: np.ndarray,
    rated_power: float = RATED_POWER_KW
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate turbine rotor speed, power, and pitch using vectorized operations.
    
    Args:
        wind_turbulent: Turbulent wind speed array.
        rated_power: Rated power in kW.
    
    Returns:
        Tuple of (rotor_speed, power, pitch) arrays.
    """
    n = len(wind_turbulent)
    rotor_speed = np.zeros(n)
    power = np.zeros(n)
    pitch = np.zeros(n)
    
    # Vectorized condition masks
    below_cut_in = wind_turbulent < CUT_IN_WIND_SPEED_MPS
    below_rated = (wind_turbulent >= CUT_IN_WIND_SPEED_MPS) & (wind_turbulent < RATED_WIND_SPEED_MPS)
    above_rated = wind_turbulent >= RATED_WIND_SPEED_MPS
    
    # Vectorized target calculations
    target_power = np.zeros(n)
    target_rpm = np.zeros(n)
    target_pitch = np.zeros(n)
    
    # Below cut-in
    target_power[below_cut_in] = 0
    target_rpm[below_cut_in] = 0
    target_pitch[below_cut_in] = MAX_PITCH_DEG
    
    # Between cut-in and rated
    wind_range = RATED_WIND_SPEED_MPS - CUT_IN_WIND_SPEED_MPS
    power_ratio = ((wind_turbulent[below_rated] - CUT_IN_WIND_SPEED_MPS) / wind_range) ** POWER_CURVE_EXPONENT
    target_power[below_rated] = rated_power * power_ratio
    target_rpm[below_rated] = MIN_ROTOR_SPEED_RPM + (wind_turbulent[below_rated] - CUT_IN_WIND_SPEED_MPS) * ROTOR_SPEED_SLOPE
    target_pitch[below_rated] = PITCH_AT_CUT_IN + (RATED_WIND_SPEED_MPS - wind_turbulent[below_rated]) * PITCH_SLOPE
    
    # Above rated
    target_power[above_rated] = rated_power
    target_rpm[above_rated] = ROTOR_SPEED_ABOVE_RATED + (wind_turbulent[above_rated] - RATED_WIND_SPEED_MPS) * ROTOR_SPEED_SLOPE_ABOVE_RATED
    target_pitch[above_rated] = MIN_PITCH_DEG
    
    # Apply inertia (exponential smoothing) with vectorized operations
    for i in range(1, n):
        rotor_speed[i] = (
            ROTOR_INERTIA * rotor_speed[i - 1] +
            ROTOR_RESPONSE * target_rpm[i]
        )
        power[i] = (
            POWER_INERTIA * power[i - 1] +
            POWER_RESPONSE * target_power[i]
        )
        pitch[i] = (
            PITCH_INERTIA * pitch[i - 1] +
            PITCH_RESPONSE * target_pitch[i]
        )
    
    # Add noise and clip
    rotor_speed += np.random.randn(n) * ROTOR_NOISE_STD
    power += np.random.randn(n) * POWER_NOISE_STD
    pitch += np.random.randn(n) * PITCH_NOISE_STD
    
    rotor_speed = np.clip(rotor_speed, 0, MAX_ROTOR_SPEED_RPM)
    power = np.clip(power, 0, rated_power * POWER_OVERRATING_FACTOR)
    pitch = np.clip(pitch, 0, MAX_PITCH_DEG)
    
    return rotor_speed, power, pitch


def simulate_turbulence_and_turbine(
    wind_df: pd.DataFrame,
    rated_power: float = RATED_POWER_KW,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate turbulence intensity and turbine response.
    
    Args:
        wind_df: DataFrame with wind data.
        rated_power: Rated power in kW.
        random_seed: Optional random seed for reproducibility.
    
    Returns:
        DataFrame with added turbulence and turbine response columns.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    df = wind_df.copy()
    hour = df['time'].dt.hour.values
    wind_mean = df['windspeed_100m'].values
    
    # Calculate turbulence intensity
    ti = calculate_turbulence_intensity(hour)
    
    # Calculate turbulent wind
    wind_turbulent = calculate_turbulent_wind(wind_mean, ti)
    
    # Calculate turbine response
    rotor_speed, power, pitch = calculate_turbine_response(
        wind_turbulent, rated_power
    )
    
    # Add to dataframe
    df['turbulence_intensity'] = ti
    df['wind_turbulent'] = wind_turbulent
    df['rotor_speed'] = rotor_speed
    df['power'] = power
    df['pitch'] = pitch
    
    return df


def diagram_to_image(
    diagram: np.ndarray,
    resolution: int = DEFAULT_RESOLUTION,
    sigma: float = DEFAULT_SIGMA
) -> np.ndarray:
    """
    Convert persistence diagram to persistence image.
    
    Args:
        diagram: Persistence diagram (nx2 array).
        resolution: Image resolution (pixels per side).
        sigma: Gaussian smoothing parameter.
    
    Returns:
        Image array (resolution x resolution).
    """
    finite_mask = np.isfinite(diagram[:, 1])
    finite_pts = diagram[finite_mask]
    
    if len(finite_pts) == 0:
        return np.zeros((resolution, resolution))
    
    births = finite_pts[:, 0]
    deaths = finite_pts[:, 1]
    persistences = deaths - births
    
    # Calculate bounds
    b_min = max(0, births.min() - DIAGRAM_PADDING)
    b_max = births.max() + DIAGRAM_PADDING
    d_min = b_min
    d_max = deaths.max() + DIAGRAM_PADDING
    
    # Create grid
    x = np.linspace(b_min, b_max, resolution)
    y = np.linspace(d_min, d_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized Gaussian computation
    image = np.zeros((resolution, resolution))
    for i in range(len(finite_pts)):
        b, d, p = births[i], deaths[i], persistences[i]
        gaussian = np.exp(
            -((X - b) ** 2 + (Y - d) ** 2) / (2 * sigma ** 2)
        )
        image += p * gaussian
    
    # Normalize
    if image.max() > 0:
        image = image / image.max()
    
    return image


def create_persistence_image_dataset(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    resolution: int = DEFAULT_RESOLUTION
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset of persistence images with turbulence labels.
    
    Args:
        df: DataFrame with turbine data.
        window_size: Window size in samples.
        resolution: Image resolution.
    
    Returns:
        Tuple of (images array, labels array).
    
    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = ['wind_turbulent', 'rotor_speed', 'power', 'pitch', 'turbulence_intensity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Creating persistence images...")
    images = []
    labels = []
    n = len(df)
    
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        window = df.iloc[start:end]
        
        # Extract features
        wind = window['wind_turbulent'].values
        rotor = window['rotor_speed'].values
        power = window['power'].values
        pitch = window['pitch'].values
        
        # Normalize
        X = np.column_stack([wind, rotor, power, pitch])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        try:
            result = ripser(X, maxdim=1)
            diagrams = result['dgms']
            
            # Create images for H0 and H1
            img_h0 = diagram_to_image(diagrams[0], resolution=resolution)
            img_h1 = diagram_to_image(
                diagrams[1] if len(diagrams) > 1 else np.empty((0, 2)),
                resolution=resolution
            )
            
            # Stack channels
            img = np.stack([img_h0, img_h1], axis=0)
            
            # Determine label
            ti_mean = window['turbulence_intensity'].mean()
            if ti_mean < TI_LOW_THRESHOLD:
                label = 0
            elif ti_mean > TI_HIGH_THRESHOLD:
                label = 1
            else:
                continue  # Skip intermediate values
            
            images.append(img)
            labels.append(label)
            
        except (ValueError, IndexError, KeyError) as e:
            logger.debug(f"Error processing window {start}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid persistence images created")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    logger.info(f"Created {len(images)} persistence images")
    logger.info(f"Low turbulence: {(labels == 0).sum()}")
    logger.info(f"High turbulence: {(labels == 1).sum()}")
    
    return images, labels


class PersistenceImageDataset(Dataset):
    """PyTorch dataset for persistence images."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """
        Initialize dataset.
        
        Args:
            images: Array of persistence images (N, C, H, W).
            labels: Array of labels (N,).
        """
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        """
        Get dataset size.
        
        Returns:
            Number of samples.
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (image tensor, label tensor).
        """
        return (self.images[idx], self.labels[idx])


class PersistenceCNN(nn.Module):
    """CNN for persistence image classification."""
    
    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES
    ) -> None:
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
        """
        super(PersistenceCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, CONV1_OUT_CHANNELS,
            kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.pool1 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.conv2 = nn.Conv2d(
            CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
            kernel_size=KERNEL_SIZE, padding=PADDING
        )
        self.pool2 = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.fc1 = nn.Linear(FC1_INPUT_SIZE, FC1_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(FC1_SIZE, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width).
        
        Returns:
            Output logits (batch, num_classes).
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, FC1_INPUT_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE
) -> nn.Module:
    """
    Train the CNN model.
    
    Args:
        model: CNN model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
    
    Returns:
        Trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % LOG_INTERVAL == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%"
            )
        
        scheduler.step()
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
    
    Returns:
        Tuple of (accuracy, AUC, predictions, probabilities, true labels).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return acc, auc, all_preds, all_probs, all_labels


def plot_roc_curve(
    labels_true: np.ndarray,
    probs: np.ndarray,
    auc: float,
    out_path: Path
) -> None:
    """
    Plot and save ROC curve.

    Args:
        labels_true: True labels.
        probs: Predicted probabilities.
        auc: AUC score.
        out_path: Output file path.
    """
    fpr, tpr, _ = roc_curve(labels_true, probs)

    fig, ax = plt.subplots(figsize=ROC_CURVE_SIZE)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve: Turbulence Classification', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {out_path}")


def _log_banner() -> None:
    """Log the run header banner."""
    logger.info("=" * 70)
    logger.info("Turbulence Classification Using Persistence Images & CNN")
    logger.info("=" * 70)


def _simulate_and_log_ti(
    wind_data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Run turbulence/turbine simulation and log TI low/high counts. Returns enriched DataFrame."""
    df = simulate_turbulence_and_turbine(wind_data, random_seed=seed)
    ti_low = (df['turbulence_intensity'] < TI_LOW_THRESHOLD).sum()
    ti_high = (df['turbulence_intensity'] > TI_HIGH_THRESHOLD).sum()
    logger.info(
        f'Low TI (<{TI_LOW_THRESHOLD}): {ti_low} '
        f'({ti_low / len(df) * 100:.1f}%)'
    )
    logger.info(
        f'High TI (>{TI_HIGH_THRESHOLD}): {ti_high} '
        f'({ti_high / len(df) * 100:.1f}%)'
    )
    return df


def _train_val_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images/labels into train/val/test and log sizes. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    split_idx = int(TRAIN_TEST_SPLIT_RATIO * len(images))
    X_train_full, X_test = images[:split_idx], images[split_idx:]
    y_train_full, y_test = labels[:split_idx], labels[split_idx:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SPLIT_RATIO,
        random_state=seed,
        stratify=y_train_full,
    )
    logger.info(f'Train: {len(X_train)} samples')
    logger.info(f'Val: {len(X_val)} samples')
    logger.info(f'Test: {len(X_test)} samples')
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build PyTorch datasets and DataLoaders for train/val/test."""
    train_dataset = PersistenceImageDataset(X_train, y_train)
    val_dataset = PersistenceImageDataset(X_val, y_val)
    test_dataset = PersistenceImageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def _evaluate_and_report(
    model: nn.Module,
    test_loader: DataLoader,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation, log accuracy/AUC/classification report; return (acc, auc, preds, probs, labels_true)."""
    acc, auc, preds, probs, labels_true = evaluate_model(model, test_loader)
    logger.info(f'\nTest Accuracy: {acc * 100:.2f}%')
    logger.info(f'Test AUC: {auc:.3f}')
    logger.info(
        f"\n{classification_report(labels_true, preds, target_names=['Low TI', 'High TI'])}"
    )
    return acc, auc, preds, probs, labels_true


def _save_figures(
    labels_true: np.ndarray,
    probs: np.ndarray,
    auc: float,
    script_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    """Create figures subdir from config and save ROC curve."""
    turb = cfg.get("turbulence", {})
    figures_subdir = turb.get("figures_subdir", "figures_turbulence")
    out_dir = script_dir / figures_subdir
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_roc_curve(labels_true, probs, auc, out_dir / 'roc_curve.png')


def _log_final_summary(acc: float) -> None:
    """Log the final completion summary."""
    logger.info('\n' + '=' * 70)
    logger.info('TURBULENCE CLASSIFICATION COMPLETE')
    logger.info('=' * 70)
    logger.info(f'\nCNN on persistence images: {acc * 100:.1f}% accuracy')
    logger.info('No specialized sensors required - SCADA only')
    logger.info('Enables:')
    logger.info('  - Turbulence-aware load monitoring')
    logger.info('  - Adaptive control strategies')
    logger.info('  - Site assessment validation')
    logger.info('=' * 70)


def main(config_path: Optional[Path] = None) -> None:
    """
    Main execution: load config, then fetch data, simulate turbine response,
    create persistence images, train CNN, and evaluate.
    """
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _log_banner()

    logger.info("\n1. Fetching NREL wind data...")
    wind_data = fetch_nrel_wind_data(cfg)
    if wind_data is None:
        logger.error('Failed to fetch data')
        return
    logger.info(f'Total records: {len(wind_data):,}')

    logger.info('\n2. Simulating turbulence and turbine response...')
    df = _simulate_and_log_ti(wind_data, seed)

    logger.info('\n3. Creating persistence image dataset...')
    images, labels = create_persistence_image_dataset(
        df, window_size=DEFAULT_WINDOW_SIZE, resolution=DEFAULT_RESOLUTION
    )

    logger.info('\n4. Splitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(images, labels, seed)
    train_loader, val_loader, test_loader = _build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    logger.info('\n5. Training CNN...')
    model = PersistenceCNN(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES
    ).to(DEVICE)
    logger.info(f'Using device: {DEVICE}')
    model = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    logger.info('\n6. Evaluating on test set...')
    acc, auc, preds, probs, labels_true = _evaluate_and_report(model, test_loader)

    logger.info("\n7. Generating visualizations...")
    _save_figures(labels_true, probs, auc, _SCRIPT_DIR, cfg)

    _log_final_summary(acc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Turbulence classification (persistence images + CNN)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)
