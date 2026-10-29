"""
SQLite Persistence Layer for NetCapacity AI
===========================================
Manages persistent settings, dataset metadata, model training run history,
and network capacity alert logs using Python's built-in sqlite3 module.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "netcapacity.db"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "organization_name": "CampusCore Backbone Network (Pune-DC01)",
    "allocated_bandwidth_mbps": 1000.0,
    "utilization_threshold_pct": 80.0,
    "expected_user_growth_pct": 12.0,
    "safety_headroom_pct": 15.0,
    "active_dataset_id": 1,
    "rf_n_estimators": 50,
    "rf_max_depth": 10,
    "test_split_ratio": 0.20,
    "imputation_method": "linear",
    "alert_latency_threshold_ms": 35.0,
    "alert_packet_loss_threshold_pct": 0.5,
    "cost_per_100mbps_usd": 140.0,
}


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = db_path or DB_PATH
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1,
            frequency TEXT NOT NULL DEFAULT 'hourly',
            row_count INTEGER NOT NULL DEFAULT 0,
            start_timestamp TEXT,
            end_timestamp TEXT,
            missing_values_filled INTEGER NOT NULL DEFAULT 0,
            duplicates_removed INTEGER NOT NULL DEFAULT 0,
            default_capacity_mbps REAL NOT NULL DEFAULT 1000.0,
            summary_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            trained_at TEXT NOT NULL,
            engine_used TEXT NOT NULL,
            train_rows INTEGER NOT NULL,
            test_rows INTEGER NOT NULL,
            rf_mae REAL NOT NULL,
            rf_rmse REAL NOT NULL,
            rf_r2 REAL NOT NULL,
            rf_mape REAL NOT NULL,
            persistence_mae REAL NOT NULL,
            persistence_rmse REAL NOT NULL,
            persistence_r2 REAL NOT NULL,
            seasonal_mae REAL NOT NULL,
            seasonal_rmse REAL NOT NULL,
            seasonal_r2 REAL NOT NULL,
            feature_importance_json TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            metric_value REAL,
            threshold_value REAL,
            event_timestamp TEXT NOT NULL,
            acknowledged INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    # Seed default settings if missing
    now_str = datetime.utcnow().isoformat()
    for key, val in DEFAULT_SETTINGS.items():
        cursor.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(val), now_str),
        )

    conn.commit()
    conn.close()


def get_settings(db_path: Optional[Path] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    result = dict(DEFAULT_SETTINGS)
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except Exception:
            result[row["key"]] = row["value"]
    return result


def update_settings(updates: Dict[str, Any], db_path: Optional[Path] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    for key, val in updates.items():
        if key in DEFAULT_SETTINGS:
            cursor.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, json.dumps(val), now_str),
            )
    conn.commit()
    conn.close()
    return get_settings(db_path)


def register_dataset(
    name: str,
    filename: str,
    description: str,
    is_synthetic: bool,
    frequency: str,
    row_count: int,
    start_timestamp: str,
    end_timestamp: str,
    missing_values_filled: int,
    duplicates_removed: int,
    default_capacity_mbps: float,
    summary_dict: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT INTO datasets (
            name, filename, description, is_synthetic, frequency,
            row_count, start_timestamp, end_timestamp,
            missing_values_filled, duplicates_removed,
            default_capacity_mbps, summary_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            name = excluded.name,
            description = excluded.description,
            is_synthetic = excluded.is_synthetic,
            frequency = excluded.frequency,
            row_count = excluded.row_count,
            start_timestamp = excluded.start_timestamp,
            end_timestamp = excluded.end_timestamp,
            missing_values_filled = excluded.missing_values_filled,
            duplicates_removed = excluded.duplicates_removed,
            default_capacity_mbps = excluded.default_capacity_mbps,
            summary_json = excluded.summary_json
        """,
        (
            name,
            filename,
            description,
            1 if is_synthetic else 0,
            frequency,
            row_count,
            start_timestamp,
            end_timestamp,
            missing_values_filled,
            duplicates_removed,
            default_capacity_mbps,
            json.dumps(summary_dict),
            now_str,
        ),
    )
    conn.commit()
    row = cursor.execute("SELECT id FROM datasets WHERE filename = ?", (filename,)).fetchone()
    dataset_id = int(row["id"])
    conn.close()
    return dataset_id


def list_datasets(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM datasets ORDER BY id ASC").fetchall()
    conn.close()
    items: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["is_synthetic"] = bool(d["is_synthetic"])
        d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else {}
        items.append(d)
    return items


def get_dataset_by_id(dataset_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["is_synthetic"] = bool(d["is_synthetic"])
    d["summary"] = json.loads(d["summary_json"]) if d.get("summary_json") else {}
    return d


def delete_dataset_record(dataset_id: int, db_path: Optional[Path] = None) -> bool:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def log_model_run(
    dataset_id: int,
    engine_used: str,
    train_rows: int,
    test_rows: int,
    rf_metrics: Dict[str, float],
    persistence_metrics: Dict[str, float],
    seasonal_metrics: Dict[str, float],
    feature_importances: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    cursor.execute(
        """
        INSERT INTO model_runs (
            dataset_id, trained_at, engine_used, train_rows, test_rows,
            rf_mae, rf_rmse, rf_r2, rf_mape,
            persistence_mae, persistence_rmse, persistence_r2,
            seasonal_mae, seasonal_rmse, seasonal_r2,
            feature_importance_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            now_str,
            engine_used,
            train_rows,
            test_rows,
            rf_metrics["mae"],
            rf_metrics["rmse"],
            rf_metrics["r2"],
            rf_metrics["mape"],
            persistence_metrics["mae"],
            persistence_metrics["rmse"],
            persistence_metrics["r2"],
            seasonal_metrics["mae"],
            seasonal_metrics["rmse"],
            seasonal_metrics["r2"],
            json.dumps(feature_importances),
        ),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return int(run_id)


def replace_alerts_for_dataset(
    dataset_id: int, alerts: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts WHERE dataset_id = ? AND acknowledged = 0", (dataset_id,))
    now_str = datetime.utcnow().isoformat()
    for a in alerts:
        cursor.execute(
            """
            INSERT INTO alerts (
                dataset_id, severity, category, title, message,
                metric_value, threshold_value, event_timestamp, acknowledged, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                dataset_id,
                a["severity"],
                a["category"],
                a["title"],
                a["message"],
                a.get("metric_value"),
                a.get("threshold_value"),
                a.get("event_timestamp", now_str),
                now_str,
            ),
        )
    conn.commit()
    conn.close()


def get_alerts(dataset_id: Optional[int] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    if dataset_id is not None:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE dataset_id = ? ORDER BY acknowledged ASC, id DESC",
            (dataset_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM alerts ORDER BY acknowledged ASC, id DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        item = dict(r)
        item["acknowledged"] = bool(item["acknowledged"])
        result.append(item)
    return result


def acknowledge_alert(alert_id: int, db_path: Optional[Path] = None) -> bool:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated
