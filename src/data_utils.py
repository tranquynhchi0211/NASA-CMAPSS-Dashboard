import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import joblib as _joblib
except ImportError:
    _joblib = None

DATASET_META = {
    "FD001": {
        "dataset_name": "FD001",
        "operating_condition": "1 operating condition (Sea Level)",
        "fault_mode": "1 fault mode (HPC Degradation)",
        "train_engines_expected": 100,
        "test_engines_expected": 100,
    },
    "FD002": {
        "dataset_name": "FD002",
        "operating_condition": "6 operating conditions",
        "fault_mode": "1 fault mode (HPC Degradation)",
        "train_engines_expected": 260,
        "test_engines_expected": 259,
    },
    "FD003": {
        "dataset_name": "FD003",
        "operating_condition": "1 operating condition (Sea Level)",
        "fault_mode": "2 fault modes (HPC Degradation, Fan Degradation)",
        "train_engines_expected": 100,
        "test_engines_expected": 100,
    },
    "FD004": {
        "dataset_name": "FD004",
        "operating_condition": "6 operating conditions",
        "fault_mode": "2 fault modes (HPC Degradation, Fan Degradation)",
        "train_engines_expected": 248,
        "test_engines_expected": 249,
    },
}

COLUMNS = (
    ["unit_id", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def load_single_dataset(data_dir: str, dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, f"train_{dataset_id}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset_id}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset_id}.txt")

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_RUL"])

    train["dataset"] = dataset_id
    test["dataset"] = dataset_id
    rul["dataset"] = dataset_id
    rul["unit_id"] = np.arange(1, len(rul) + 1)
    train["RUL"] = train.groupby("unit_id")["time_cycles"].transform(lambda x: x.max() - x)
    return train, test, rul


def available_datasets(data_dir: str) -> List[str]:
    found = []
    for dataset_id in DATASET_META:
        if all(os.path.exists(os.path.join(data_dir, f"{prefix}_{dataset_id}.txt")) for prefix in ["train", "test", "RUL"]):
            found.append(dataset_id)
    return found


def load_all_datasets(data_dir: str, dataset_ids: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train_data, test_data, rul_data = {}, {}, {}
    for dataset_id in dataset_ids:
        train, test, rul = load_single_dataset(data_dir, dataset_id)
        train_data[dataset_id] = train
        test_data[dataset_id] = test
        rul_data[dataset_id] = rul
    return train_data, test_data, rul_data


def dataset_overview(train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_id, train in train_data.items():
        test = test_data[dataset_id]
        lifetimes = train.groupby("unit_id")["time_cycles"].max()
        meta = DATASET_META[dataset_id]
        rows.append({
            "Dataset": dataset_id,
            "Train Engines": train["unit_id"].nunique(),
            "Test Engines": test["unit_id"].nunique(),
            "Total Cycles": int(train["time_cycles"].count()),
            "Average Cycle": round(train["time_cycles"].mean(), 2),
            "Average Lifetime": round(lifetimes.mean(), 2),
            "Median Lifetime": round(lifetimes.median(), 2),
            "Min Lifetime": int(lifetimes.min()),
            "Max Lifetime": int(lifetimes.max()),
            "Operating Condition": meta["operating_condition"],
            "Fault Mode": meta["fault_mode"],
        })
    return pd.DataFrame(rows)


def lifetime_summary(train: pd.DataFrame) -> pd.DataFrame:
    lifetimes = train.groupby("unit_id")["time_cycles"].max()
    return pd.DataFrame({
        "Metric": ["Average Lifetime", "Median Lifetime", "Minimum Lifetime", "Maximum Lifetime"],
        "Value": [round(lifetimes.mean(), 2), round(lifetimes.median(), 2), int(lifetimes.min()), int(lifetimes.max())],
    })


def normalize_sensors(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    scaler = MinMaxScaler()
    df[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])
    return df


def sensor_variance_ranking(train: pd.DataFrame) -> pd.DataFrame:
    variance = train[SENSOR_COLS].var().sort_values(ascending=False)
    return pd.DataFrame({"Sensor": variance.index, "Variance": variance.values})


def sensor_distribution_long(train: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_sensors(train)
    return normalized.melt(
        id_vars=["dataset", "unit_id", "time_cycles"],
        value_vars=SENSOR_COLS,
        var_name="Sensor",
        value_name="Normalized Value",
    )


def lifecycle_sensor_profile(train: pd.DataFrame, sensor: str, bins: int = 20) -> pd.DataFrame:
    df = train.copy()
    df["lifetime"] = df.groupby("unit_id")["time_cycles"].transform("max")
    df["lifecycle_pct"] = df["time_cycles"] / df["lifetime"]
    df["lifecycle_bin"] = pd.cut(df["lifecycle_pct"], bins=bins, labels=False, include_lowest=True)
    profile = df.groupby("lifecycle_bin", as_index=False)[sensor].mean()
    profile["Lifecycle Percentage"] = (profile["lifecycle_bin"] + 1) / bins * 100
    return profile[["Lifecycle Percentage", sensor]]


def create_latest_test_fleet(test: pd.DataFrame, rul: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    latest = test.sort_values("time_cycles").groupby("unit_id").tail(1).copy()
    latest = latest.merge(rul[["unit_id", "true_RUL"]], on="unit_id", how="left")
    latest["dataset"] = dataset_id
    latest["observed_cycles"] = latest["time_cycles"]
    latest["estimated_failure_cycle"] = latest["observed_cycles"] + latest["true_RUL"]
    return latest


def assign_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = 1 / (1 + np.exp((out["predicted_RUL"] - 30) / 8))
    out["risk_level"] = pd.cut(
        out["predicted_RUL"],
        bins=[-np.inf, 15, 30, 60, np.inf],
        labels=["Critical", "High", "Medium", "Low"],
    )
    return out


def baseline_predict_rul(fleet: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    df = fleet.copy()
    avg_lifetime = train.groupby("unit_id")["time_cycles"].max().mean()
    df["predicted_RUL"] = (avg_lifetime - df["observed_cycles"]).clip(lower=1)
    return assign_risk(df)


def load_cached_predictions(prediction_dir: str, dataset_id: str) -> Optional[pd.DataFrame]:
    path = os.path.join(prediction_dir, f"predictions_{dataset_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_model_dict(model_dir: str, dataset_id: str) -> Optional[dict]:
    if _joblib is None:
        return None
    path = os.path.join(model_dir, f"model {dataset_id}.joblib")
    if not os.path.exists(path):
        return None
    try:
        return _joblib.load(path)
    except Exception:
        return None


def _select_features_for_model(train: pd.DataFrame, n_features: int) -> List[str]:
    all_features = OP_COLS + SENSOR_COLS
    var = train[all_features].var()
    top_cols = set(var.nlargest(n_features).index)
    return [f for f in all_features if f in top_cols]


def _make_test_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, List[int]]:
    sequences: List[np.ndarray] = []
    engine_ids: List[int] = []
    n = len(feature_cols)
    for uid in sorted(df["unit_id"].unique()):
        data = (
            df[df["unit_id"] == uid]
            .sort_values("time_cycles")[feature_cols]
            .values.astype(np.float32)
        )
        if len(data) < seq_len:
            pad = np.zeros((seq_len - len(data), n), dtype=np.float32)
            data = np.vstack([pad, data])
        sequences.append(data[-seq_len:])
        engine_ids.append(int(uid))
    return np.array(sequences), engine_ids


def predict_rul_with_model(
    test: pd.DataFrame,
    rul: pd.DataFrame,
    train: pd.DataFrame,
    model,
    dataset_id: str,
) -> pd.DataFrame:
    seq_len: int = model.input_shape[1]
    n_features: int = model.input_shape[2]

    feature_cols = _select_features_for_model(train, n_features)

    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])

    test_scaled = test.copy()
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

    X, engine_ids = _make_test_sequences(test_scaled, feature_cols, seq_len)
    preds = model.predict(X, verbose=0).flatten()
    preds = np.clip(preds, 0, None)

    fleet = create_latest_test_fleet(test, rul, dataset_id)
    pred_df = pd.DataFrame({"unit_id": engine_ids, "predicted_RUL": preds})
    fleet = fleet.merge(pred_df, on="unit_id", how="left")
    return assign_risk(fleet)


def cost_analysis_summary(
    fleet: pd.DataFrame,
    planned_cost: float = 10000,
    unplanned_cost: float = 80000,
    warning_threshold: float = 80,
    critical_threshold: float = 30,
) -> Dict[str, float]:
    total_engines = len(fleet)
    predictive_scheduled = int((fleet["predicted_RUL"] <= warning_threshold).sum())
    predictive_unscheduled = int((fleet["predicted_RUL"] <= critical_threshold).sum())

    traditional_scheduled = int(total_engines * 0.75)
    traditional_unscheduled = total_engines - traditional_scheduled

    traditional_total = traditional_scheduled * planned_cost + traditional_unscheduled * unplanned_cost
    predictive_total = predictive_scheduled * planned_cost + predictive_unscheduled * unplanned_cost
    savings = max(traditional_total - predictive_total, 0)
    savings_rate = savings / traditional_total * 100 if traditional_total else 0

    traditional_availability = max(0, 100 - traditional_unscheduled / total_engines * 100)
    predictive_availability = max(0, 100 - predictive_unscheduled / total_engines * 100)

    return {
        "traditional_total_cost": traditional_total,
        "predictive_total_cost": predictive_total,
        "savings": savings,
        "savings_rate": savings_rate,
        "traditional_scheduled": traditional_scheduled,
        "traditional_unscheduled": traditional_unscheduled,
        "predictive_scheduled": predictive_scheduled,
        "predictive_unscheduled": predictive_unscheduled,
        "traditional_availability": traditional_availability,
        "predictive_availability": predictive_availability,
    }
