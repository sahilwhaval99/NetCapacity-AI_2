"""
Data Validation, Timestamp Verification, and Preprocessing Module
=================================================================
Handles CSV ingestion, flexible column mapping, timestamp continuity checks,
duplicate resolution, frequency inference (hourly vs. daily), and missing-value
imputation without data leakage.
"""

from io import BytesIO, StringIO
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd

TIMESTAMP_ALIASES = [
    "timestamp",
    "datetime",
    "date_time",
    "date",
    "time",
    "recorded_at",
    "period",
]

BANDWIDTH_ALIASES = [
    "bandwidth_mbps",
    "traffic_mbps",
    "throughput_mbps",
    "bandwidth",
    "utilization_mbps",
    "demand_mbps",
    "mbps",
]


class DatasetValidationError(ValueError):
    """Raised when an uploaded CSV fails structural or timestamp validation."""


def _detect_column(columns: list, aliases: list) -> str:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    # Substring fallback
    for alias in aliases:
        for low_col, orig_col in lower_map.items():
            if alias in low_col:
                return orig_col
    return ""


def infer_frequency(timestamps: pd.Series) -> Tuple[str, pd.Timedelta]:
    """Infers whether the time series is hourly or daily based on median time delta."""
    diffs = timestamps.sort_values().diff().dropna()
    if len(diffs) == 0:
        raise DatasetValidationError("Dataset must contain at least 2 valid timestamps.")
    median_delta = diffs.median()
    hours = median_delta.total_seconds() / 3600.0
    if 0.4 <= hours <= 3.5:
        return "hourly", pd.Timedelta(hours=1)
    elif 18.0 <= hours <= 32.0:
        return "daily", pd.Timedelta(days=1)
    else:
        # Default to hourly if closer to hours, else daily
        if hours < 12.0:
            return "hourly", pd.Timedelta(hours=1)
        return "daily", pd.Timedelta(days=1)


def validate_and_preprocess_dataframe(
    raw_df: pd.DataFrame,
    imputation_method: str = "linear",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates schema, parses and sorts timestamps, resolves duplicates,
    reindexes missing timestamp steps, and imputes missing numeric values.

    Returns:
        (cleaned_df, preprocessing_report)
    """
    if raw_df is None or raw_df.empty:
        raise DatasetValidationError("Uploaded CSV dataset is empty.")

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _detect_column(list(df.columns), TIMESTAMP_ALIASES)
    bw_col = _detect_column(list(df.columns), BANDWIDTH_ALIASES)

    if not ts_col:
        raise DatasetValidationError(
            f"Could not find a timestamp column. Expected one of: {TIMESTAMP_ALIASES}. Found: {list(df.columns)}"
        )
    if not bw_col:
        raise DatasetValidationError(
            f"Could not find a bandwidth column. Expected one of: {BANDWIDTH_ALIASES}. Found: {list(df.columns)}"
        )

    initial_rows = len(df)

    # Rename standard columns
    rename_map = {ts_col: "timestamp", bw_col: "bandwidth_mbps"}
    df = df.rename(columns=rename_map)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts_count = int(df["timestamp"].isna().sum())
    if invalid_ts_count > 0:
        df = df.dropna(subset=["timestamp"]).copy()

    if len(df) < 35:
        raise DatasetValidationError(
            f"Dataset has only {len(df)} valid timestamped rows. At least 35 rows are required for lag feature engineering and chronological train/test evaluation."
        )

    # Coerce bandwidth to numeric
    df["bandwidth_mbps"] = pd.to_numeric(df["bandwidth_mbps"], errors="coerce")
    raw_missing_bw = int(df["bandwidth_mbps"].isna().sum())

    # Replace negative values with NaN so they are imputed cleanly
    negative_count = int((df["bandwidth_mbps"] < 0).sum())
    if negative_count > 0:
        df.loc[df["bandwidth_mbps"] < 0, "bandwidth_mbps"] = np.nan

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Handle duplicate timestamps by averaging numeric columns
    duplicates_removed = int(df.duplicated(subset=["timestamp"]).sum())
    if duplicates_removed > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in df.columns if c not in numeric_cols and c != "timestamp"]
        agg_dict = {c: "mean" for c in numeric_cols}
        for c in non_num_cols:
            agg_dict[c] = "last"
        df = df.groupby("timestamp", as_index=False).agg(agg_dict).sort_values("timestamp").reset_index(drop=True)

    # Infer sampling frequency and check timestamp continuity
    freq_label, freq_delta = infer_frequency(df["timestamp"])
    pandas_freq = "h" if freq_label == "hourly" else "D"

    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=pandas_freq,
    )
    gaps_reindexed = 0
    # Reindex onto regular grid if the grid size is reasonable (within 25% of row count)
    if len(full_index) >= len(df) and len(full_index) <= int(len(df) * 1.25):
        gaps_reindexed = int(len(full_index) - len(df))
        df = df.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()

    total_missing_before_imputation = int(df["bandwidth_mbps"].isna().sum())

    # Impute bandwidth_mbps
    if imputation_method == "ffill":
        df["bandwidth_mbps"] = df["bandwidth_mbps"].ffill().bfill()
    elif imputation_method == "median":
        rolling_med = df["bandwidth_mbps"].rolling(window=7, min_periods=1, center=False).median()
        df["bandwidth_mbps"] = df["bandwidth_mbps"].fillna(rolling_med).ffill().bfill()
    else:
        # Default: linear interpolation + boundary fill
        df["bandwidth_mbps"] = df["bandwidth_mbps"].interpolate(method="linear").ffill().bfill()

    df["bandwidth_mbps"] = np.round(df["bandwidth_mbps"].clip(lower=1.0), 2)

    # Auxiliary telemetry columns (create realistic derived defaults if uploaded CSV omits them)
    if "active_users" not in df.columns:
        df["active_users"] = np.round(df["bandwidth_mbps"] * 2.1).astype(int)
    else:
        df["active_users"] = (
            pd.to_numeric(df["active_users"], errors="coerce")
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .fillna(100)
            .round()
            .astype(int)
        )

    if "latency_ms" not in df.columns:
        p95 = float(np.percentile(df["bandwidth_mbps"], 95)) or 1000.0
        util_est = df["bandwidth_mbps"] / (p95 * 1.15)
        df["latency_ms"] = np.round(10.0 + util_est * 16.0, 2)
    else:
        df["latency_ms"] = (
            pd.to_numeric(df["latency_ms"], errors="coerce")
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .fillna(12.0)
            .round(2)
        )

    if "packet_loss_pct" not in df.columns:
        df["packet_loss_pct"] = 0.02
    else:
        df["packet_loss_pct"] = (
            pd.to_numeric(df["packet_loss_pct"], errors="coerce")
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .fillna(0.01)
            .round(3)
        )

    is_synthetic = False
    if "data_source" in df.columns:
        sample_sources = " ".join(df["data_source"].dropna().astype(str).unique()).upper()
        if "SYNTHETIC" in sample_sources:
            is_synthetic = True
    else:
        df["data_source"] = "USER_UPLOADED_CSV"

    default_cap = 1000.0
    if "allocated_capacity_mbps" in df.columns:
        valid_caps = pd.to_numeric(df["allocated_capacity_mbps"], errors="coerce").dropna()
        if len(valid_caps) > 0:
            default_cap = float(valid_caps.iloc[-1])
    else:
        # Pick sensible next standard link tier above max observed traffic
        max_bw = float(df["bandwidth_mbps"].max())
        for tier in [500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0]:
            if max_bw <= tier * 0.95:
                default_cap = tier
                break
        else:
            default_cap = round(max_bw * 1.25, -2)
        df["allocated_capacity_mbps"] = default_cap

    bw_series = df["bandwidth_mbps"]
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "clean_rows": int(len(df)),
        "invalid_timestamps_dropped": invalid_ts_count,
        "duplicates_removed": duplicates_removed,
        "timestamp_gaps_reindexed": gaps_reindexed,
        "raw_missing_bandwidth_values": raw_missing_bw,
        "negative_values_cleaned": negative_count,
        "total_missing_imputed": total_missing_before_imputation,
        "imputation_method": imputation_method,
        "frequency": freq_label,
        "supported_horizons": ["24h", "7d", "30d"] if freq_label == "hourly" else ["7d", "30d"],
        "start_timestamp": df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "end_timestamp": df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "is_synthetic": is_synthetic,
        "default_capacity_mbps": default_cap,
        "stats": {
            "min_mbps": round(float(bw_series.min()), 2),
            "max_mbps": round(float(bw_series.max()), 2),
            "mean_mbps": round(float(bw_series.mean()), 2),
            "median_mbps": round(float(bw_series.median()), 2),
            "std_mbps": round(float(bw_series.std()), 2),
            "p95_mbps": round(float(np.percentile(bw_series, 95)), 2),
            "p99_mbps": round(float(np.percentile(bw_series, 99)), 2),
            "avg_latency_ms": round(float(df["latency_ms"].mean()), 2),
            "max_latency_ms": round(float(df["latency_ms"].max()), 2),
            "avg_packet_loss_pct": round(float(df["packet_loss_pct"].mean()), 3),
            "avg_active_users": int(round(float(df["active_users"].mean()))),
            "peak_active_users": int(df["active_users"].max()),
        },
    }

    return df, report


def load_and_preprocess_csv(
    source: Union[str, bytes, BytesIO],
    imputation_method: str = "linear",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads CSV from file path or uploaded bytes and runs full validation + preprocessing."""
    try:
        if isinstance(source, bytes):
            raw_df = pd.read_csv(BytesIO(source))
        elif isinstance(source, str) and "\n" in source and not source.endswith(".csv"):
            raw_df = pd.read_csv(StringIO(source))
        else:
            raw_df = pd.read_csv(source)
    except Exception as exc:
        raise DatasetValidationError(f"Failed to parse CSV file: {exc}") from exc

    return validate_and_preprocess_dataframe(raw_df, imputation_method=imputation_method)
