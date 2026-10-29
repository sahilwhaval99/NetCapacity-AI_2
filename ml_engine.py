"""
Genuine Machine Learning Time-Series Forecasting Engine for NetCapacity AI
==========================================================================
Implements:
1. Leak-free Autoregressive Lag, Causal Rolling-Window, and Cyclical Calendar Feature Engineering.
2. Strict Chronological Train/Test Split (no random shuffling, no future leakage).
3. Random Forest Regressor (uses scikit-learn when installed, with a vectorized NumPy CART
   Random Forest fallback so the project runs in any environment).
4. Persistence (Naive Walk y_{t-1}) and Seasonal-Naive (y_{t-24} / y_{t-7}) Baselines.
5. Authentic MAE, RMSE, R², and MAPE calculation + Residual Analysis.
6. Multi-Step Autoregressive Forecasting for 24-Hour, 7-Day, and 30-Day Horizons
   with 90% Prediction Intervals and User Growth Scaling.
7. Model Artifact Serialization via Joblib (with pickle fallback).
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SklearnRandomForestRegressor = None
    SKLEARN_AVAILABLE = False

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Vectorized NumPy CART Regression Tree & Random Forest (Fallback Engine)
# ============================================================================
class _CARTNode:
    __slots__ = ("feature_idx", "threshold", "left", "right", "value")

    def __init__(
        self,
        value: float,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_CARTNode"] = None,
        right: Optional["_CARTNode"] = None,
    ):
        self.value = value
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right


class NumpyRandomForestRegressor:
    """
    Genuine Breiman (2001) Random Forest Regressor built with vectorized NumPy CART trees.
    Provides identical API (.fit, .predict, .feature_importances_) to
    sklearn.ensemble.RandomForestRegressor for environments without scikit-learn installed.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 10,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.trees: List[_CARTNode] = []
        self.feature_importances_: np.ndarray = np.array([])
        self.n_features_in_: int = 0

    def _select_num_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(3, int(np.ceil(n_features ** 0.5 * 1.4)))
        elif isinstance(self.max_features, float):
            return max(2, int(n_features * self.max_features))
        return n_features

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        importances: np.ndarray,
    ) -> _CARTNode:
        n_samples, n_features = X.shape
        node_mean = float(np.mean(y))

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.allclose(y, y[0], atol=1e-5)
        ):
            return _CARTNode(value=node_mean)

        total_sse = float(np.sum((y - node_mean) ** 2))
        if total_sse <= 1e-7:
            return _CARTNode(value=node_mean)

        m_try = min(n_features, self._select_num_features(n_features))
        feat_indices = rng.choice(n_features, size=m_try, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thresh = None
        best_left_mask = None

        quantiles = np.linspace(0.1, 0.9, 12)

        for f_idx in feat_indices:
            col = X[:, f_idx]
            thresholds = np.unique(np.quantile(col, quantiles))
            if len(thresholds) == 0:
                continue

            left_masks = col[None, :] <= thresholds[:, None]
            left_counts = np.sum(left_masks, axis=1)
            right_counts = n_samples - left_counts

            valid = (left_counts >= self.min_samples_leaf) & (right_counts >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            valid_masks = left_masks[valid]
            valid_thresholds = thresholds[valid]
            l_cnt = left_counts[valid].astype(np.float64)
            r_cnt = right_counts[valid].astype(np.float64)

            total_sum = float(np.sum(y))
            left_sums = valid_masks @ y
            right_sums = total_sum - left_sums

            base_term = (total_sum ** 2) / n_samples
            gains = (left_sums ** 2) / l_cnt + (right_sums ** 2) / r_cnt - base_term

            max_idx = int(np.argmax(gains))
            if gains[max_idx] > best_gain:
                best_gain = float(gains[max_idx])
                best_feat = int(f_idx)
                best_thresh = float(valid_thresholds[max_idx])
                best_left_mask = valid_masks[max_idx]

        if best_feat is None or best_left_mask is None or best_gain <= 1e-6:
            return _CARTNode(value=node_mean)

        importances[best_feat] += best_gain
        right_mask = ~best_left_mask

        left_child = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1, rng, importances)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng, importances)

        return _CARTNode(
            value=node_mean,
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def _predict_tree_batch(self, node: _CARTNode, X: np.ndarray) -> np.ndarray:
        if node.feature_idx is None or node.left is None or node.right is None:
            return np.full(X.shape[0], node.value, dtype=np.float64)
        mask = X[:, node.feature_idx] <= node.threshold
        out = np.empty(X.shape[0], dtype=np.float64)
        if np.any(mask):
            out[mask] = self._predict_tree_batch(node.left, X[mask])
        if np.any(~mask):
            out[~mask] = self._predict_tree_batch(node.right, X[~mask])
        return out

    @staticmethod
    def _predict_tree_single(node: _CARTNode, row: np.ndarray) -> float:
        curr = node
        while curr.feature_idx is not None and curr.left is not None and curr.right is not None:
            if row[curr.feature_idx] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyRandomForestRegressor":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.trees = []
        raw_importances = np.zeros(n_features, dtype=np.float64)

        master_rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            tree_seed = int(master_rng.integers(0, 2**31 - 1))
            tree_rng = np.random.default_rng(tree_seed)
            boot_idx = tree_rng.choice(n_samples, size=n_samples, replace=True)
            tree = self._build_tree(X_arr[boot_idx], y_arr[boot_idx], 0, tree_rng, raw_importances)
            self.trees.append(tree)

        total_imp = float(np.sum(raw_importances))
        if total_imp > 0:
            self.feature_importances_ = raw_importances / total_imp
        else:
            self.feature_importances_ = np.ones(n_features, dtype=np.float64) / n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[0] == 1:
            row = X_arr[0]
            vals = [self._predict_tree_single(t, row) for t in self.trees]
            return np.array([float(np.mean(vals))], dtype=np.float64)
        preds = np.column_stack([self._predict_tree_batch(t, X_arr) for t in self.trees])
        return np.mean(preds, axis=1)

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[0] == 1:
            row = X_arr[0]
            vals = np.array([self._predict_tree_single(t, row) for t in self.trees], dtype=np.float64)
            return np.array([float(np.mean(vals))]), np.array([float(np.std(vals))])
        preds = np.column_stack([self._predict_tree_batch(t, X_arr) for t in self.trees])
        return np.mean(preds, axis=1), np.std(preds, axis=1)


# ============================================================================
# Metrics & Feature Engineering Functions
# ============================================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes genuine MAE, RMSE, R², and MAPE (%) without fabrication."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    denom = np.maximum(np.abs(y_t), 1.0)
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
    }


def get_feature_config(frequency: str, total_rows: int) -> Dict[str, Any]:
    """Returns lag indices and rolling window sizes tailored to sampling frequency."""
    if frequency == "hourly":
        lags = [1, 2, 3, 6, 12, 24, 48]
        if total_rows >= 260:
            lags.append(168)
        seasonal_lag = 24
        rolling_windows = [6, 24]
    else:
        lags = [1, 2, 3, 7, 14]
        if total_rows >= 90:
            lags.append(30)
        seasonal_lag = 7
        rolling_windows = [7, 14]

    return {
        "lags": lags,
        "seasonal_lag": seasonal_lag,
        "rolling_windows": rolling_windows,
        "max_lag": max(lags),
    }


def build_supervised_features(
    df: pd.DataFrame,
    frequency: str,
    train_norm_length: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Constructs strictly causal time-series features for supervised learning.
    Every lag and rolling statistic at index t is computed ONLY from {y_{t-1}, y_{t-2}, ...}.
    """
    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    cfg = get_feature_config(frequency, len(work))
    y = work["bandwidth_mbps"].astype(float)

    feature_cols: List[str] = []

    for lag in cfg["lags"]:
        col_name = f"lag_{lag}"
        work[col_name] = y.shift(lag)
        feature_cols.append(col_name)

    y_past = y.shift(1)
    for w in cfg["rolling_windows"]:
        mean_col = f"rolling_mean_{w}"
        std_col = f"rolling_std_{w}"
        max_col = f"rolling_max_{w}"
        work[mean_col] = y_past.rolling(window=w, min_periods=2).mean()
        work[std_col] = y_past.rolling(window=w, min_periods=2).std().fillna(0.0)
        work[max_col] = y_past.rolling(window=w, min_periods=2).max()
        feature_cols.extend([mean_col, std_col, max_col])

    work["diff_1"] = y.shift(1) - y.shift(2)
    seasonal_lag = cfg["seasonal_lag"]
    work[f"diff_seasonal_{seasonal_lag}"] = y.shift(1) - y.shift(seasonal_lag)
    feature_cols.extend(["diff_1", f"diff_seasonal_{seasonal_lag}"])

    ts = pd.to_datetime(work["timestamp"])
    dow = ts.dt.dayofweek.astype(float)
    work["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    work["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    work["is_weekend"] = (dow >= 5).astype(float)
    feature_cols.extend(["dow_sin", "dow_cos", "is_weekend"])

    if frequency == "hourly":
        hour = ts.dt.hour.astype(float)
        work["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        work["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        work["is_business_hour"] = ((hour >= 9) & (hour <= 18) & (dow < 5)).astype(float)
        feature_cols.extend(["hour_sin", "hour_cos", "is_business_hour"])
    else:
        day_of_year = ts.dt.dayofyear.astype(float)
        work["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        work["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
        feature_cols.extend(["doy_sin", "doy_cos"])

    norm_denom = float(train_norm_length if train_norm_length else max(1, int(len(work) * 0.8)))
    work["trend_index"] = np.arange(len(work), dtype=float) / norm_denom
    feature_cols.append("trend_index")

    clean_supervised = work.dropna(subset=feature_cols + ["bandwidth_mbps"]).reset_index(drop=True)
    return clean_supervised, feature_cols, cfg


def _build_single_future_feature_row(
    next_ts: pd.Timestamp,
    history_values: List[float],
    frequency: str,
    cfg: Dict[str, Any],
    step_global_idx: int,
    train_norm_length: int,
    feature_cols: List[str],
) -> np.ndarray:
    """Builds the exact feature vector for 1-step-ahead recursive prediction at next_ts."""
    row_dict: Dict[str, float] = {}
    n_hist = len(history_values)

    for lag in cfg["lags"]:
        row_dict[f"lag_{lag}"] = float(history_values[n_hist - lag])

    for w in cfg["rolling_windows"]:
        window_slice = np.asarray(history_values[max(0, n_hist - w) : n_hist], dtype=np.float64)
        row_dict[f"rolling_mean_{w}"] = float(np.mean(window_slice))
        row_dict[f"rolling_std_{w}"] = float(np.std(window_slice, ddof=1)) if len(window_slice) > 1 else 0.0
        row_dict[f"rolling_max_{w}"] = float(np.max(window_slice))

    row_dict["diff_1"] = float(history_values[-1] - history_values[-2])
    s_lag = cfg["seasonal_lag"]
    row_dict[f"diff_seasonal_{s_lag}"] = float(history_values[-1] - history_values[n_hist - s_lag])

    dow = float(next_ts.dayofweek)
    row_dict["dow_sin"] = float(np.sin(2.0 * np.pi * dow / 7.0))
    row_dict["dow_cos"] = float(np.cos(2.0 * np.pi * dow / 7.0))
    row_dict["is_weekend"] = 1.0 if dow >= 5 else 0.0

    if frequency == "hourly":
        hour = float(next_ts.hour)
        row_dict["hour_sin"] = float(np.sin(2.0 * np.pi * hour / 24.0))
        row_dict["hour_cos"] = float(np.cos(2.0 * np.pi * hour / 24.0))
        row_dict["is_business_hour"] = 1.0 if (9 <= hour <= 18 and dow < 5) else 0.0
    else:
        doy = float(next_ts.dayofyear)
        row_dict["doy_sin"] = float(np.sin(2.0 * np.pi * doy / 365.25))
        row_dict["doy_cos"] = float(np.cos(2.0 * np.pi * doy / 365.25))

    row_dict["trend_index"] = float(step_global_idx) / float(max(1, train_norm_length))

    return np.array([[row_dict[col] for col in feature_cols]], dtype=np.float64)


# ============================================================================
# Model Training, Chronological Evaluation & Persistence
# ============================================================================
def train_and_evaluate_model(
    df: pd.DataFrame,
    frequency: str,
    dataset_id: int = 1,
    n_estimators: int = 50,
    max_depth: int = 10,
    test_split_ratio: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs leak-free chronological train/test split, fits Random Forest Regressor,
    compares against Persistence and Seasonal-Naive baselines, saves artifact via Joblib/pickle,
    and returns comprehensive evaluation diagnostics.
    """
    test_ratio = float(np.clip(test_split_ratio, 0.10, 0.35))
    prelim_train_len = int(len(df) * (1.0 - test_ratio))

    sup_df, feature_cols, cfg = build_supervised_features(
        df, frequency=frequency, train_norm_length=prelim_train_len
    )

    total_sup = len(sup_df)
    if total_sup < 20:
        raise ValueError(
            f"Insufficient rows ({total_sup}) after lag warm-up to perform chronological train/test split."
        )

    split_idx = int(total_sup * (1.0 - test_ratio))
    split_idx = max(12, min(total_sup - 5, split_idx))

    train_df = sup_df.iloc[:split_idx].copy()
    test_df = sup_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["bandwidth_mbps"].to_numpy(dtype=np.float64)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_df["bandwidth_mbps"].to_numpy(dtype=np.float64)

    if SKLEARN_AVAILABLE and SklearnRandomForestRegressor is not None:
        model = SklearnRandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=int(random_state),
            n_jobs=-1,
        )
        engine_used = "scikit-learn RandomForestRegressor"
    else:
        model = NumpyRandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=int(random_state),
        )
        engine_used = "RandomForestRegressor (NumPy Vectorized CART Ensemble)"

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_test_persistence = test_df["lag_1"].to_numpy(dtype=np.float64)
    s_lag = cfg["seasonal_lag"]
    y_test_seasonal = test_df[f"lag_{s_lag}"].to_numpy(dtype=np.float64)

    rf_train_metrics = compute_regression_metrics(y_train, y_train_pred)
    rf_test_metrics = compute_regression_metrics(y_test, y_test_pred)
    persistence_metrics = compute_regression_metrics(y_test, y_test_persistence)
    seasonal_metrics = compute_regression_metrics(y_test, y_test_seasonal)

    rmse_vs_persistence_pct = round(
        (1.0 - rf_test_metrics["rmse"] / max(persistence_metrics["rmse"], 1e-6)) * 100.0, 2
    )
    rmse_vs_seasonal_pct = round(
        (1.0 - rf_test_metrics["rmse"] / max(seasonal_metrics["rmse"], 1e-6)) * 100.0, 2
    )

    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    feature_importance_list = [
        {"feature": col, "importance": round(float(imp), 4), "importance_pct": round(float(imp * 100.0), 2)}
        for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    ]

    residuals = y_test - y_test_pred
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 10.0

    max_chart_points = min(len(test_df), 168)
    chart_slice = test_df.iloc[-max_chart_points:].copy()
    slice_offset = len(test_df) - max_chart_points

    test_predictions_series = []
    for i in range(max_chart_points):
        idx = slice_offset + i
        ts_val = pd.to_datetime(chart_slice.iloc[i]["timestamp"]).strftime("%Y-%m-%d %H:%M")
        actual_val = round(float(y_test[idx]), 2)
        rf_val = round(float(y_test_pred[idx]), 2)
        pers_val = round(float(y_test_persistence[idx]), 2)
        seas_val = round(float(y_test_seasonal[idx]), 2)
        test_predictions_series.append(
            {
                "timestamp": ts_val,
                "actual_mbps": actual_val,
                "rf_predicted_mbps": rf_val,
                "persistence_mbps": pers_val,
                "seasonal_naive_mbps": seas_val,
                "residual_mbps": round(actual_val - rf_val, 2),
            }
        )

    artifact = {
        "dataset_id": dataset_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "engine_used": engine_used,
        "serializer": "joblib" if JOBLIB_AVAILABLE else "pickle",
        "frequency": frequency,
        "feature_cols": feature_cols,
        "feature_config": cfg,
        "train_norm_length": prelim_train_len,
        "total_clean_rows": int(len(df)),
        "supervised_rows": total_sup,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": pd.to_datetime(train_df["timestamp"].min()).strftime("%Y-%m-%d %H:%M"),
        "train_end": pd.to_datetime(train_df["timestamp"].max()).strftime("%Y-%m-%d %H:%M"),
        "test_start": pd.to_datetime(test_df["timestamp"].min()).strftime("%Y-%m-%d %H:%M"),
        "test_end": pd.to_datetime(test_df["timestamp"].max()).strftime("%Y-%m-%d %H:%M"),
        "hyperparameters": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "test_split_ratio": test_ratio,
            "random_state": int(random_state),
        },
        "metrics": {
            "random_forest_test": rf_test_metrics,
            "random_forest_train": rf_train_metrics,
            "persistence_baseline": persistence_metrics,
            "seasonal_naive_baseline": seasonal_metrics,
            "seasonal_lag_used": s_lag,
            "rmse_improvement_over_persistence_pct": rmse_vs_persistence_pct,
            "rmse_improvement_over_seasonal_pct": rmse_vs_seasonal_pct,
            "residual_mean_mbps": round(float(np.mean(residuals)), 3),
            "residual_std_mbps": round(residual_std, 3),
        },
        "feature_importances": feature_importance_list,
        "test_comparison_series": test_predictions_series,
        "model": model,
    }

    artifact_path = MODELS_DIR / f"rf_model_dataset_{dataset_id}.joblib"
    if JOBLIB_AVAILABLE and joblib is not None:
        joblib.dump(artifact, artifact_path)
    else:
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)

    return artifact


def load_trained_artifact(dataset_id: int) -> Optional[Dict[str, Any]]:
    artifact_path = MODELS_DIR / f"rf_model_dataset_{dataset_id}.joblib"
    if not artifact_path.exists():
        return None
    try:
        if JOBLIB_AVAILABLE and joblib is not None:
            return joblib.load(artifact_path)
        with open(artifact_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def generate_multi_step_forecast(
    df: pd.DataFrame,
    artifact: Dict[str, Any],
    horizon: str = "24h",
    user_growth_pct: float = 0.0,
) -> Dict[str, Any]:
    """Generates multi-step ahead forecasts using recursive autoregressive stepping."""
    frequency = artifact["frequency"]
    feature_cols = artifact["feature_cols"]
    cfg = artifact["feature_config"]
    train_norm_length = artifact["train_norm_length"]
    model = artifact["model"]
    residual_std = float(artifact["metrics"]["residual_std_mbps"])

    horizon_clean = horizon.strip().lower()
    if frequency == "hourly":
        if horizon_clean == "24h":
            steps = 24
            step_delta = pd.Timedelta(hours=1)
        elif horizon_clean == "7d":
            steps = 168
            step_delta = pd.Timedelta(hours=1)
        elif horizon_clean == "30d":
            steps = 720
            step_delta = pd.Timedelta(hours=1)
        else:
            steps = 24
            horizon_clean = "24h"
            step_delta = pd.Timedelta(hours=1)
    else:
        step_delta = pd.Timedelta(days=1)
        if horizon_clean in ("24h", "7d"):
            steps = 7
            horizon_clean = "7d"
        else:
            steps = 30
            horizon_clean = "30d"

    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    history_values: List[float] = sorted_df["bandwidth_mbps"].astype(float).tolist()
    last_ts = pd.to_datetime(sorted_df["timestamp"].iloc[-1])
    base_global_idx = len(sorted_df)

    growth_multiplier_total = 1.0 + (float(user_growth_pct) / 100.0)

    forecast_points: List[Dict[str, Any]] = []
    s_lag = cfg["seasonal_lag"]

    for step in range(1, steps + 1):
        next_ts = last_ts + step_delta * step
        x_row = _build_single_future_feature_row(
            next_ts=next_ts,
            history_values=history_values,
            frequency=frequency,
            cfg=cfg,
            step_global_idx=base_global_idx + step,
            train_norm_length=train_norm_length,
            feature_cols=feature_cols,
        )

        if hasattr(model, "predict_with_std"):
            pred_arr, tree_std_arr = model.predict_with_std(x_row)
            base_pred = float(pred_arr[0])
            tree_std = float(tree_std_arr[0])
        elif hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(x_row)[0] for est in model.estimators_], dtype=np.float64)
            base_pred = float(np.mean(tree_preds))
            tree_std = float(np.std(tree_preds))
        else:
            base_pred = float(model.predict(x_row)[0])
            tree_std = residual_std * 0.6

        base_pred = max(5.0, base_pred)
        seasonal_ref = float(history_values[len(history_values) - s_lag])
        history_values.append(base_pred)

        growth_adjusted = base_pred * growth_multiplier_total
        horizon_uncertainty_scale = np.sqrt(1.0 + 0.18 * (step / float(s_lag)))
        combined_sigma = np.sqrt(residual_std ** 2 + (0.65 * tree_std) ** 2) * horizon_uncertainty_scale
        margin_90 = 1.645 * combined_sigma * growth_multiplier_total

        lower_90 = max(1.0, growth_adjusted - margin_90)
        upper_90 = growth_adjusted + margin_90

        ts_fmt = next_ts.strftime("%Y-%m-%d %H:%M") if frequency == "hourly" else next_ts.strftime("%Y-%m-%d")
        forecast_points.append(
            {
                "step": step,
                "timestamp": ts_fmt,
                "base_forecast_mbps": round(base_pred, 2),
                "predicted_mbps": round(growth_adjusted, 2),
                "seasonal_baseline_mbps": round(seasonal_ref * growth_multiplier_total, 2),
                "lower_bound_mbps": round(lower_90, 2),
                "upper_bound_mbps": round(upper_90, 2),
            }
        )

    hist_window = 72 if frequency == "hourly" else 30
    recent_df = sorted_df.iloc[-hist_window:]
    historical_context = [
        {
            "timestamp": (
                pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
                if frequency == "hourly"
                else pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            ),
            "actual_mbps": round(float(row["bandwidth_mbps"]), 2),
        }
        for _, row in recent_df.iterrows()
    ]

    pred_vals = np.array([p["predicted_mbps"] for p in forecast_points], dtype=np.float64)
    base_vals = np.array([p["base_forecast_mbps"] for p in forecast_points], dtype=np.float64)
    upper_vals = np.array([p["upper_bound_mbps"] for p in forecast_points], dtype=np.float64)
    peak_idx = int(np.argmax(pred_vals))

    return {
        "horizon": horizon_clean,
        "frequency": frequency,
        "steps": steps,
        "user_growth_pct": round(float(user_growth_pct), 2),
        "supported_horizons": ["24h", "7d", "30d"] if frequency == "hourly" else ["7d", "30d"],
        "summary": {
            "base_peak_mbps": round(float(np.max(base_vals)), 2),
            "predicted_peak_mbps": round(float(np.max(pred_vals)), 2),
            "predicted_avg_mbps": round(float(np.mean(pred_vals)), 2),
            "predicted_p95_mbps": round(float(np.percentile(pred_vals, 95)), 2),
            "predicted_min_mbps": round(float(np.min(pred_vals)), 2),
            "upper_90_peak_mbps": round(float(np.max(upper_vals)), 2),
            "peak_timestamp": forecast_points[peak_idx]["timestamp"],
        },
        "historical_context": historical_context,
        "forecast_series": forecast_points,
    }
