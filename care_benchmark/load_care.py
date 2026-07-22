"""Load the CARE to Compare wind-turbine SCADA benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

META_COLUMNS = {"id", "train_test", "time_stamp", "asset_id", "status_type_id"}
STAT_MAP = {
    "average": "avg",
    "avg": "avg",
    "minimum": "min",
    "min": "min",
    "maximum": "max",
    "max": "max",
    "std_dev": "std",
    "std": "std",
    "standard_deviation": "std",
}


@dataclass
class CareEvent:
    event_id: int
    wind_farm: str
    asset_id: int
    event_label: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def has_anomaly(self) -> bool:
        return self.event_label.lower() == "anomaly"


class CareDataset:
    """Loader for CARE to Compare CSV datasets."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.wind_farms = {
            "A": self.root / "Wind Farm A",
            "B": self.root / "Wind Farm B",
            "C": self.root / "Wind Farm C",
        }
        self.feature_descriptions = {
            farm: pd.read_csv(path / "feature_description.csv", sep=";")
            for farm, path in self.wind_farms.items()
            if (path / "feature_description.csv").is_file()
        }
        self.event_info = self._load_event_info()

    def _load_event_info(self) -> pd.DataFrame:
        frames = []
        for farm, path in self.wind_farms.items():
            info_path = path / "event_info.csv"
            if not info_path.is_file():
                continue
            info = pd.read_csv(info_path, sep=";")
            info["event_start"] = pd.to_datetime(info["event_start"])
            info["event_end"] = pd.to_datetime(info["event_end"])
            info["wind_farm"] = farm
            if "asset" in info.columns:
                info = info.rename(columns={"asset": "asset_id"})
            if "event_label" not in info.columns:
                for candidate in ("label", "type", "event_type", "dataset_label"):
                    if candidate in info.columns:
                        info = info.rename(columns={candidate: "event_label"})
                        break
            if "event_label" in info.columns:
                info["event_label"] = info["event_label"].astype(str).str.lower().map(
                    lambda x: "anomaly" if x in {"anomaly", "fault", "failure", "1", "true", "yes"} else "normal"
                )
            else:
                info["event_label"] = "normal"
            frames.append(info)
        if not frames:
            raise FileNotFoundError(f"No event_info.csv found under {self.root}")
        return pd.concat(frames, ignore_index=True)

    def list_events(self, wind_farm: Optional[str] = None) -> pd.DataFrame:
        if wind_farm is None:
            return self.event_info.copy()
        return self.event_info[self.event_info["wind_farm"] == wind_farm].copy()

    def load_event(self, event_id: int, statistics: Optional[List[str]] = None) -> CareEvent:
        row = self.event_info[self.event_info["event_id"] == event_id].iloc[0]
        farm = row["wind_farm"]
        dataset = self._read_event_csv(event_id, farm, statistics=statistics)
        train = dataset[dataset["train_test"] == "train"].drop(columns=["train_test"])
        test = dataset[dataset["train_test"] == "prediction"].drop(columns=["train_test"])
        label = str(row.get("event_label", row.get("label", "normal"))).lower()
        return CareEvent(
            event_id=int(event_id),
            wind_farm=farm,
            asset_id=int(row["asset_id"]),
            event_label=label,
            event_start=row["event_start"],
            event_end=row["event_end"],
            train=train,
            test=test,
        )

    def iter_events(self, wind_farm: Optional[str] = None) -> Iterator[CareEvent]:
        events = self.list_events(wind_farm=wind_farm)
        for event_id in events["event_id"].tolist():
            yield self.load_event(int(event_id))

    def select_key_sensors(self, wind_farm: str) -> Dict[str, str]:
        """Return wind/power/rotor sensor column names (avg statistics)."""
        desc = self.feature_descriptions.get(wind_farm)
        if desc is None:
            return {}
        mapping: Dict[str, str] = {}
        for _, row in desc.iterrows():
            text = str(row.get("description", "")).lower()
            sensor = str(row["sensor_name"])
            if "wind speed" in text and "wind" not in mapping:
                mapping["wind"] = sensor
            elif text.strip() == "power" or text.startswith("power "):
                mapping.setdefault("power", sensor)
            elif "rotor speed" in text or "generator speed" in text:
                mapping.setdefault("rotor", sensor)
        return mapping

    def _read_event_csv(
        self,
        event_id: int,
        wind_farm: str,
        statistics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        path = self.wind_farms[wind_farm] / "datasets" / f"{event_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        usecols = self._selected_columns(wind_farm, path, statistics)
        dataset = pd.read_csv(
            path,
            sep=";",
            usecols=usecols,
            parse_dates=["time_stamp"],
            date_format="%Y-%m-%d %H:%M:%S",
        )
        numeric_cols = [c for c in dataset.columns if c not in META_COLUMNS]
        dataset[numeric_cols] = dataset[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return dataset

    def _selected_columns(
        self,
        wind_farm: str,
        dataset_path: Path,
        statistics: Optional[List[str]],
    ) -> List[str]:
        stats = statistics or ["avg"]
        stats = [STAT_MAP.get(s.lower(), s) for s in stats]
        header = pd.read_csv(dataset_path, sep=";", nrows=0).columns.tolist()
        selected = ["id", "train_test", "time_stamp", "asset_id", "status_type_id"]
        desc = self.feature_descriptions.get(wind_farm)
        if desc is None:
            return selected + [c for c in header if c not in META_COLUMNS][:20]
        for _, row in desc.iterrows():
            sensor = row["sensor_name"]
            stat_types = [STAT_MAP.get(s.strip().lower(), s.strip().lower()) for s in str(row["statistics_type"]).split(",")]
            for stat in stats:
                if stat not in stat_types:
                    continue
                if stat == "avg" and sensor in header:
                    selected.append(sensor)
                else:
                    col = f"{sensor}_{stat}"
                    if col in header:
                        selected.append(col)
        return list(dict.fromkeys(selected))


def make_synthetic_care_dataset(root: Path, n_assets: int = 3) -> Path:
    """Create a minimal CARE-like dataset for tests."""
    import numpy as np

    root = Path(root)
    farm_dir = root / "Wind Farm A"
    datasets_dir = farm_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng(0)
    event_id = 1
    for asset_id in range(1, n_assets + 1):
        for label in ("normal", "anomaly"):
            n_train, n_test = 500, 200
            ts_train = pd.date_range("2020-01-01", periods=n_train, freq="10min")
            ts_test = pd.date_range(ts_train[-1] + pd.Timedelta("10min"), periods=n_test, freq="10min")
            wind_train = 8 + rng.normal(0, 1, n_train)
            wind_test = 8 + rng.normal(0, 1, n_test)
            if label == "anomaly":
                wind_test[100:160] -= 2.0
            power_train = np.clip(0.08 * wind_train**2.5, 0, 2.0)
            power_test = np.clip(0.08 * wind_test**2.5, 0, 2.0)
            rotor_train = wind_train * 4
            rotor_test = wind_test * 4
            status_train = np.zeros(n_train, dtype=int)
            status_test = np.zeros(n_test, dtype=int)
            status_test[100:180] = 1
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "id": np.arange(n_train),
                            "train_test": "train",
                            "time_stamp": ts_train,
                            "asset_id": asset_id,
                            "status_type_id": status_train,
                            "sensor_wind": wind_train,
                            "sensor_power": power_train,
                            "sensor_rotor": rotor_train,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "id": np.arange(n_train, n_train + n_test),
                            "train_test": "prediction",
                            "time_stamp": ts_test,
                            "asset_id": asset_id,
                            "status_type_id": status_test,
                            "sensor_wind": wind_test,
                            "sensor_power": power_test,
                            "sensor_rotor": rotor_test,
                        }
                    ),
                ],
                ignore_index=True,
            )
            df.to_csv(datasets_dir / f"{event_id}.csv", sep=";", index=False)
            rows.append(
                {
                    "event_id": event_id,
                    "asset_id": asset_id,
                    "event_label": label,
                    "event_start": ts_test[100],
                    "event_end": ts_test[179],
                }
            )
            event_id += 1

    pd.DataFrame(rows).to_csv(farm_dir / "event_info.csv", sep=";", index=False)
    pd.DataFrame(
        [
            {
                "sensor_name": "sensor_wind",
                "description": "Wind speed",
                "statistics_type": "average",
                "is_angle": False,
                "is_counter": False,
            },
            {
                "sensor_name": "sensor_power",
                "description": "Power",
                "statistics_type": "average",
                "is_angle": False,
                "is_counter": False,
            },
            {
                "sensor_name": "sensor_rotor",
                "description": "Rotor speed",
                "statistics_type": "average",
                "is_angle": False,
                "is_counter": False,
            },
        ]
    ).to_csv(farm_dir / "feature_description.csv", sep=";", index=False)
    return root
