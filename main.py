"""
FastAPI Backend Application for NetCapacity AI
==============================================
Exposes REST APIs for:
- Dashboard KPIs, historical telemetry, and capacity status overview
- Multi-horizon ML bandwidth forecasting (24h, 7d, 30d)
- Chronological train/test model evaluation (Random Forest vs. Persistence & Seasonal-Naive)
- Interactive Capacity Planner simulator & upgrade recommendation engine
- Dataset Manager & CSV upload with validation, timestamp checks, and imputation
- Alert management & acknowledgement
- Persistent SQLite settings
- Downloadable CSV reports (Forecast CSV & Capacity Audit Report CSV)
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from capacity_planner import evaluate_capacity_plan, generate_system_alerts
from database import (
    acknowledge_alert,
    delete_dataset_record,
    get_alerts,
    get_dataset_by_id,
    get_settings,
    init_db,
    list_datasets,
    log_model_run,
    register_dataset,
    replace_alerts_for_dataset,
    update_settings,
)
from ml_engine import (
    SKLEARN_AVAILABLE,
    generate_multi_step_forecast,
    load_trained_artifact,
    train_and_evaluate_model,
)
from preprocessing import DatasetValidationError, load_and_preprocess_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
UPLOADS_DIR = Path(__file__).resolve().parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="NetCapacity AI - Intelligent Network Capacity Planning API",
    description=(
        "B.Tech AI & DS Project Backend: Analyzes historical network traffic telemetry, "
        "forecasts future bandwidth demand using Random Forest with lag features, "
        "detects SLA capacity shortages, and recommends infrastructure upgrades."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CLEAN_DF_CACHE: Dict[int, pd.DataFrame] = {}
_ARTIFACT_CACHE: Dict[int, Dict[str, Any]] = {}
_INITIALIZED: bool = False

BUILTIN_DATASETS = [
    {
        "name": "Campus Core Router (60-Day Hourly Telemetry)",
        "filename": "campus_core_hourly_60d.csv",
        "description": (
            "Synthetic 60-day hourly telemetry (1,440 hrs) from a 1 Gbps University/Enterprise Campus Core router "
            "with diurnal work/evening peaks, weekend drop, +18% organic traffic growth, and 1% missing values."
        ),
    },
    {
        "name": "Regional ISP Edge Trunk (90-Day Hourly Telemetry)",
        "filename": "isp_edge_hourly_90d.csv",
        "description": (
            "Synthetic 90-day hourly telemetry (2,160 hrs) from a 2.5 Gbps Regional ISP Aggregation link "
            "exhibiting heavy evening streaming surges (19:00-23:30) and weekend bandwidth expansion."
        ),
    },
    {
        "name": "Data Center WAN Interconnect (180-Day Daily Telemetry)",
        "filename": "datacenter_wan_daily_180d.csv",
        "description": (
            "Synthetic 180-day daily telemetry from a 5 Gbps Cloud Data Center WAN Interconnect link "
            "with weekly business cycles and steady cloud migration growth."
        ),
    },
]


class CapacityPlanRequest(BaseModel):
    allocated_bandwidth_mbps: float = Field(..., gt=10.0, le=500000.0)
    utilization_threshold_pct: float = Field(80.0, ge=40.0, le=99.0)
    expected_user_growth_pct: float = Field(12.0, ge=-50.0, le=300.0)
    safety_headroom_pct: float = Field(15.0, ge=0.0, le=100.0)
    horizon: str = Field("24h")
    persist_settings: bool = Field(True)


class TrainModelRequest(BaseModel):
    n_estimators: int = Field(50, ge=10, le=250)
    max_depth: int = Field(10, ge=3, le=25)
    test_split_ratio: float = Field(0.20, ge=0.10, le=0.35)
    imputation_method: str = Field("linear")


class SettingsUpdateRequest(BaseModel):
    organization_name: Optional[str] = None
    allocated_bandwidth_mbps: Optional[float] = None
    utilization_threshold_pct: Optional[float] = None
    expected_user_growth_pct: Optional[float] = None
    safety_headroom_pct: Optional[float] = None
    rf_n_estimators: Optional[int] = None
    rf_max_depth: Optional[int] = None
    test_split_ratio: Optional[float] = None
    imputation_method: Optional[str] = None
    alert_latency_threshold_ms: Optional[float] = None
    alert_packet_loss_threshold_pct: Optional[float] = None
    cost_per_100mbps_usd: Optional[float] = None


def _resolve_dataset_path(filename: str) -> Path:
    p1 = DATASETS_DIR / filename
    if p1.exists():
        return p1
    p2 = UPLOADS_DIR / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Dataset file '{filename}' not found in datasets/ or uploads/.")


def _get_or_load_clean_df(dataset_id: int, imputation_method: str = "linear") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ds_meta = get_dataset_by_id(dataset_id)
    if not ds_meta:
        raise HTTPException(status_code=404, detail=f"Dataset ID {dataset_id} not found.")

    if dataset_id in _CLEAN_DF_CACHE:
        return _CLEAN_DF_CACHE[dataset_id], ds_meta

    csv_path = _resolve_dataset_path(ds_meta["filename"])
    clean_df, _ = load_and_preprocess_csv(str(csv_path), imputation_method=imputation_method)
    _CLEAN_DF_CACHE[dataset_id] = clean_df
    return clean_df, ds_meta


def _get_or_train_artifact(dataset_id: int, force_retrain: bool = False) -> Dict[str, Any]:
    if not force_retrain and dataset_id in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[dataset_id]

    if not force_retrain:
        disk_artifact = load_trained_artifact(dataset_id)
        if disk_artifact is not None:
            _ARTIFACT_CACHE[dataset_id] = disk_artifact
            return disk_artifact

    settings = get_settings()
    clean_df, ds_meta = _get_or_load_clean_df(dataset_id, imputation_method=settings.get("imputation_method", "linear"))

    artifact = train_and_evaluate_model(
        df=clean_df,
        frequency=ds_meta["frequency"],
        dataset_id=dataset_id,
        n_estimators=int(settings.get("rf_n_estimators", 50)),
        max_depth=int(settings.get("rf_max_depth", 10)),
        test_split_ratio=float(settings.get("test_split_ratio", 0.20)),
    )
    _ARTIFACT_CACHE[dataset_id] = artifact

    log_model_run(
        dataset_id=dataset_id,
        engine_used=artifact["engine_used"],
        train_rows=artifact["train_rows"],
        test_rows=artifact["test_rows"],
        rf_metrics=artifact["metrics"]["random_forest_test"],
        persistence_metrics=artifact["metrics"]["persistence_baseline"],
        seasonal_metrics=artifact["metrics"]["seasonal_naive_baseline"],
        feature_importances=artifact["feature_importances"],
    )
    return artifact


def ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    init_db()
    existing = list_datasets()
    existing_filenames = {d["filename"] for d in existing}

    for item in BUILTIN_DATASETS:
        fpath = DATASETS_DIR / item["filename"]
        if fpath.exists() and item["filename"] not in existing_filenames:
            clean_df, report = load_and_preprocess_csv(str(fpath), imputation_method="linear")
            ds_id = register_dataset(
                name=item["name"],
                filename=item["filename"],
                description=item["description"],
                is_synthetic=True,
                frequency=report["frequency"],
                row_count=report["clean_rows"],
                start_timestamp=report["start_timestamp"],
                end_timestamp=report["end_timestamp"],
                missing_values_filled=report["total_missing_imputed"],
                duplicates_removed=report["duplicates_removed"],
                default_capacity_mbps=report["default_capacity_mbps"],
                summary_dict=report,
            )
            _CLEAN_DF_CACHE[ds_id] = clean_df

    settings = get_settings()
    all_ds = list_datasets()
    if all_ds:
        active_id = int(settings.get("active_dataset_id", all_ds[0]["id"]))
        valid_ids = {d["id"] for d in all_ds}
        if active_id not in valid_ids:
            active_id = all_ds[0]["id"]
            update_settings({"active_dataset_id": active_id})

        clean_df, ds_meta = _get_or_load_clean_df(active_id)
        artifact = _get_or_train_artifact(active_id)
        default_horizon = "24h" if ds_meta["frequency"] == "hourly" else "7d"
        forecast_res = generate_multi_step_forecast(
            df=clean_df,
            artifact=artifact,
            horizon=default_horizon,
            user_growth_pct=float(settings.get("expected_user_growth_pct", 12.0)),
        )
        cap_plan = evaluate_capacity_plan(
            df=clean_df,
            forecast_result=forecast_res,
            allocated_bandwidth_mbps=float(settings.get("allocated_bandwidth_mbps", 1000.0)),
            utilization_threshold_pct=float(settings.get("utilization_threshold_pct", 80.0)),
            expected_user_growth_pct=float(settings.get("expected_user_growth_pct", 12.0)),
            safety_headroom_pct=float(settings.get("safety_headroom_pct", 15.0)),
        )
        alerts = generate_system_alerts(
            df=clean_df,
            capacity_plan=cap_plan,
            latency_threshold_ms=float(settings.get("alert_latency_threshold_ms", 35.0)),
            packet_loss_threshold_pct=float(settings.get("alert_packet_loss_threshold_pct", 0.5)),
        )
        replace_alerts_for_dataset(active_id, alerts)

    _INITIALIZED = True


@app.on_event("startup")
def on_startup() -> None:
    ensure_initialized()


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_ds = get_dataset_by_id(int(settings["active_dataset_id"]))
    return {
        "status": "healthy",
        "service": "NetCapacity AI Backend",
        "version": "1.0.0",
        "sklearn_available": SKLEARN_AVAILABLE,
        "ml_engine": (
            "scikit-learn RandomForestRegressor"
            if SKLEARN_AVAILABLE
            else "RandomForestRegressor (NumPy Vectorized CART Ensemble)"
        ),
        "active_dataset": active_ds["name"] if active_ds else None,
        "active_dataset_id": settings["active_dataset_id"],
    }


@app.get("/api/dashboard")
def get_dashboard_overview(
    horizon: Optional[str] = Query(None, description="Forecast horizon: 24h, 7d, or 30d"),
) -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    clean_df, ds_meta = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    effective_horizon = horizon or ("24h" if ds_meta["frequency"] == "hourly" else "7d")
    if ds_meta["frequency"] == "daily" and effective_horizon == "24h":
        effective_horizon = "7d"

    forecast_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=effective_horizon,
        user_growth_pct=float(settings["expected_user_growth_pct"]),
    )

    cap_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=forecast_res,
        allocated_bandwidth_mbps=float(settings["allocated_bandwidth_mbps"]),
        utilization_threshold_pct=float(settings["utilization_threshold_pct"]),
        expected_user_growth_pct=float(settings["expected_user_growth_pct"]),
        safety_headroom_pct=float(settings["safety_headroom_pct"]),
    )

    alerts = get_alerts(dataset_id=active_id)
    if not alerts:
        generated = generate_system_alerts(
            df=clean_df,
            capacity_plan=cap_plan,
            latency_threshold_ms=float(settings["alert_latency_threshold_ms"]),
            packet_loss_threshold_pct=float(settings["alert_packet_loss_threshold_pct"]),
        )
        replace_alerts_for_dataset(active_id, generated)
        alerts = get_alerts(dataset_id=active_id)

    tail_n = 96 if ds_meta["frequency"] == "hourly" else 60
    recent_slice = clean_df.tail(tail_n)
    telemetry_series = [
        {
            "timestamp": (
                pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
                if ds_meta["frequency"] == "hourly"
                else pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d")
            ),
            "bandwidth_mbps": round(float(r["bandwidth_mbps"]), 2),
            "active_users": int(r["active_users"]),
            "latency_ms": round(float(r["latency_ms"]), 2),
            "packet_loss_pct": round(float(r["packet_loss_pct"]), 3),
        }
        for _, r in recent_slice.iterrows()
    ]

    temp_df = clean_df.copy()
    temp_df["hour"] = pd.to_datetime(temp_df["timestamp"]).dt.hour
    temp_df["dow"] = pd.to_datetime(temp_df["timestamp"]).dt.dayofweek
    hourly_profile = []
    if ds_meta["frequency"] == "hourly":
        grp = temp_df.groupby("hour")["bandwidth_mbps"].agg(["mean", "max", "min"]).reset_index()
        for _, row in grp.iterrows():
            hourly_profile.append(
                {
                    "hour": f"{int(row['hour']):02d}:00",
                    "avg_mbps": round(float(row["mean"]), 2),
                    "peak_mbps": round(float(row["max"]), 2),
                    "min_mbps": round(float(row["min"]), 2),
                }
            )
    else:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        grp = temp_df.groupby("dow")["bandwidth_mbps"].agg(["mean", "max", "min"]).reset_index()
        for _, row in grp.iterrows():
            hourly_profile.append(
                {
                    "hour": dow_names[int(row["dow"])],
                    "avg_mbps": round(float(row["mean"]), 2),
                    "peak_mbps": round(float(row["max"]), 2),
                    "min_mbps": round(float(row["min"]), 2),
                }
            )

    return {
        "organization_name": settings["organization_name"],
        "active_dataset": ds_meta,
        "settings": settings,
        "capacity_plan": cap_plan,
        "forecast": forecast_res,
        "model_metrics": artifact["metrics"],
        "engine_used": artifact["engine_used"],
        "telemetry_series": telemetry_series,
        "seasonality_profile": hourly_profile,
        "alerts": alerts,
    }


@app.get("/api/forecast")
def get_bandwidth_forecast(
    horizon: str = Query("24h", description="24h, 7d, or 30d"),
    user_growth_pct: Optional[float] = Query(None),
) -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    clean_df, _ = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    growth = float(user_growth_pct) if user_growth_pct is not None else float(settings["expected_user_growth_pct"])

    forecast_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=horizon,
        user_growth_pct=growth,
    )

    cap_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=forecast_res,
        allocated_bandwidth_mbps=float(settings["allocated_bandwidth_mbps"]),
        utilization_threshold_pct=float(settings["utilization_threshold_pct"]),
        expected_user_growth_pct=growth,
        safety_headroom_pct=float(settings["safety_headroom_pct"]),
    )

    return {
        "forecast": forecast_res,
        "capacity_plan": cap_plan,
    }


@app.post("/api/capacity/plan")
def run_capacity_planner_simulation(req: CapacityPlanRequest) -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    clean_df, _ = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    if req.persist_settings:
        settings = update_settings(
            {
                "allocated_bandwidth_mbps": req.allocated_bandwidth_mbps,
                "utilization_threshold_pct": req.utilization_threshold_pct,
                "expected_user_growth_pct": req.expected_user_growth_pct,
                "safety_headroom_pct": req.safety_headroom_pct,
            }
        )

    forecast_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=req.horizon,
        user_growth_pct=req.expected_user_growth_pct,
    )

    cap_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=forecast_res,
        allocated_bandwidth_mbps=req.allocated_bandwidth_mbps,
        utilization_threshold_pct=req.utilization_threshold_pct,
        expected_user_growth_pct=req.expected_user_growth_pct,
        safety_headroom_pct=req.safety_headroom_pct,
    )

    if req.persist_settings:
        alerts = generate_system_alerts(
            df=clean_df,
            capacity_plan=cap_plan,
            latency_threshold_ms=float(settings["alert_latency_threshold_ms"]),
            packet_loss_threshold_pct=float(settings["alert_packet_loss_threshold_pct"]),
        )
        replace_alerts_for_dataset(active_id, alerts)

    return {
        "capacity_plan": cap_plan,
        "forecast": forecast_res,
    }


@app.get("/api/model/evaluation")
def get_model_evaluation() -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    _, ds_meta = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    return {
        "dataset": ds_meta,
        "trained_at": artifact["trained_at"],
        "engine_used": artifact["engine_used"],
        "serializer": artifact["serializer"],
        "frequency": artifact["frequency"],
        "total_clean_rows": artifact["total_clean_rows"],
        "supervised_rows": artifact["supervised_rows"],
        "train_rows": artifact["train_rows"],
        "test_rows": artifact["test_rows"],
        "train_window": f"{artifact['train_start']} to {artifact['train_end']}",
        "test_window": f"{artifact['test_start']} to {artifact['test_end']}",
        "hyperparameters": artifact["hyperparameters"],
        "metrics": artifact["metrics"],
        "feature_importances": artifact["feature_importances"],
        "test_comparison_series": artifact["test_comparison_series"],
    }


@app.post("/api/model/train")
def retrain_model(req: TrainModelRequest) -> Dict[str, Any]:
    ensure_initialized()
    settings = update_settings(
        {
            "rf_n_estimators": req.n_estimators,
            "rf_max_depth": req.max_depth,
            "test_split_ratio": req.test_split_ratio,
            "imputation_method": req.imputation_method,
        }
    )
    active_id = int(settings["active_dataset_id"])
    _CLEAN_DF_CACHE.pop(active_id, None)
    _ARTIFACT_CACHE.pop(active_id, None)

    clean_df, ds_meta = _get_or_load_clean_df(active_id, imputation_method=req.imputation_method)
    artifact = train_and_evaluate_model(
        df=clean_df,
        frequency=ds_meta["frequency"],
        dataset_id=active_id,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        test_split_ratio=req.test_split_ratio,
    )
    _ARTIFACT_CACHE[active_id] = artifact

    log_model_run(
        dataset_id=active_id,
        engine_used=artifact["engine_used"],
        train_rows=artifact["train_rows"],
        test_rows=artifact["test_rows"],
        rf_metrics=artifact["metrics"]["random_forest_test"],
        persistence_metrics=artifact["metrics"]["persistence_baseline"],
        seasonal_metrics=artifact["metrics"]["seasonal_naive_baseline"],
        feature_importances=artifact["feature_importances"],
    )

    return get_model_evaluation()


@app.get("/api/datasets")
def get_all_datasets() -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    datasets = list_datasets()
    return {
        "active_dataset_id": int(settings["active_dataset_id"]),
        "datasets": datasets,
    }


@app.post("/api/datasets/upload")
async def upload_csv_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    imputation_method: str = Form("linear"),
    activate_immediately: bool = Form(True),
) -> Dict[str, Any]:
    ensure_initialized()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        clean_df, report = load_and_preprocess_csv(raw_bytes, imputation_method=imputation_method)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_filename = Path(file.filename).name.replace(" ", "_")
    target_path = UPLOADS_DIR / safe_filename
    clean_df.to_csv(target_path, index=False)

    ds_name = name.strip() if name and name.strip() else f"Uploaded: {safe_filename}"
    ds_desc = (
        description.strip()
        if description and description.strip()
        else f"User-uploaded network traffic telemetry ({report['clean_rows']} {report['frequency']} records)."
    )

    ds_id = register_dataset(
        name=ds_name,
        filename=safe_filename,
        description=ds_desc,
        is_synthetic=report["is_synthetic"],
        frequency=report["frequency"],
        row_count=report["clean_rows"],
        start_timestamp=report["start_timestamp"],
        end_timestamp=report["end_timestamp"],
        missing_values_filled=report["total_missing_imputed"],
        duplicates_removed=report["duplicates_removed"],
        default_capacity_mbps=report["default_capacity_mbps"],
        summary_dict=report,
    )
    _CLEAN_DF_CACHE[ds_id] = clean_df
    _ARTIFACT_CACHE.pop(ds_id, None)

    if activate_immediately:
        update_settings(
            {
                "active_dataset_id": ds_id,
                "allocated_bandwidth_mbps": report["default_capacity_mbps"],
                "imputation_method": imputation_method,
            }
        )
        artifact = _get_or_train_artifact(ds_id, force_retrain=True)
        settings = get_settings()
        default_horizon = "24h" if report["frequency"] == "hourly" else "7d"
        f_res = generate_multi_step_forecast(
            df=clean_df,
            artifact=artifact,
            horizon=default_horizon,
            user_growth_pct=float(settings["expected_user_growth_pct"]),
        )
        c_plan = evaluate_capacity_plan(
            df=clean_df,
            forecast_result=f_res,
            allocated_bandwidth_mbps=float(report["default_capacity_mbps"]),
            utilization_threshold_pct=float(settings["utilization_threshold_pct"]),
            expected_user_growth_pct=float(settings["expected_user_growth_pct"]),
            safety_headroom_pct=float(settings["safety_headroom_pct"]),
        )
        alerts = generate_system_alerts(df=clean_df, capacity_plan=c_plan)
        replace_alerts_for_dataset(ds_id, alerts)

    return {
        "message": "CSV dataset validated, preprocessed, and trained successfully.",
        "dataset_id": ds_id,
        "preprocessing_report": report,
        "datasets": list_datasets(),
    }


@app.post("/api/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: int) -> Dict[str, Any]:
    ensure_initialized()
    ds_meta = get_dataset_by_id(dataset_id)
    if not ds_meta:
        raise HTTPException(status_code=404, detail=f"Dataset ID {dataset_id} not found.")

    settings = update_settings(
        {
            "active_dataset_id": dataset_id,
            "allocated_bandwidth_mbps": float(ds_meta["default_capacity_mbps"]),
        }
    )

    clean_df, _ = _get_or_load_clean_df(dataset_id)
    artifact = _get_or_train_artifact(dataset_id)
    default_horizon = "24h" if ds_meta["frequency"] == "hourly" else "7d"
    f_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=default_horizon,
        user_growth_pct=float(settings["expected_user_growth_pct"]),
    )
    c_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=f_res,
        allocated_bandwidth_mbps=float(settings["allocated_bandwidth_mbps"]),
        utilization_threshold_pct=float(settings["utilization_threshold_pct"]),
        expected_user_growth_pct=float(settings["expected_user_growth_pct"]),
        safety_headroom_pct=float(settings["safety_headroom_pct"]),
    )
    alerts = generate_system_alerts(df=clean_df, capacity_plan=c_plan)
    replace_alerts_for_dataset(dataset_id, alerts)

    return {
        "message": f"Activated dataset '{ds_meta['name']}'",
        "active_dataset": ds_meta,
        "settings": settings,
    }


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int) -> Dict[str, Any]:
    ensure_initialized()
    ds_meta = get_dataset_by_id(dataset_id)
    if not ds_meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    builtin_names = {b["filename"] for b in BUILTIN_DATASETS}
    if ds_meta["filename"] in builtin_names:
        raise HTTPException(
            status_code=400,
            detail="Built-in reference synthetic datasets cannot be deleted.",
        )

    delete_dataset_record(dataset_id)
    _CLEAN_DF_CACHE.pop(dataset_id, None)
    _ARTIFACT_CACHE.pop(dataset_id, None)

    settings = get_settings()
    if int(settings["active_dataset_id"]) == dataset_id:
        remaining = list_datasets()
        if remaining:
            activate_dataset(remaining[0]["id"])

    return {"message": "Dataset deleted.", "datasets": list_datasets()}


@app.get("/api/alerts")
def list_system_alerts() -> Dict[str, Any]:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    alerts = get_alerts(dataset_id=active_id)
    return {
        "active_dataset_id": active_id,
        "alerts": alerts,
        "unacknowledged_count": sum(1 for a in alerts if not a["acknowledged"]),
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
def mark_alert_acknowledged(alert_id: int) -> Dict[str, Any]:
    ensure_initialized()
    ok = acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Alert ID {alert_id} not found.")
    return list_system_alerts()


@app.get("/api/settings")
def read_system_settings() -> Dict[str, Any]:
    ensure_initialized()
    return get_settings()


@app.put("/api/settings")
def save_system_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    ensure_initialized()
    payload = {k: v for k, v in req.dict().items() if v is not None}
    updated = update_settings(payload)

    active_id = int(updated["active_dataset_id"])
    clean_df, ds_meta = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)
    default_horizon = "24h" if ds_meta["frequency"] == "hourly" else "7d"
    f_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=default_horizon,
        user_growth_pct=float(updated["expected_user_growth_pct"]),
    )
    c_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=f_res,
        allocated_bandwidth_mbps=float(updated["allocated_bandwidth_mbps"]),
        utilization_threshold_pct=float(updated["utilization_threshold_pct"]),
        expected_user_growth_pct=float(updated["expected_user_growth_pct"]),
        safety_headroom_pct=float(updated["safety_headroom_pct"]),
    )
    alerts = generate_system_alerts(
        df=clean_df,
        capacity_plan=c_plan,
        latency_threshold_ms=float(updated["alert_latency_threshold_ms"]),
        packet_loss_threshold_pct=float(updated["alert_packet_loss_threshold_pct"]),
    )
    replace_alerts_for_dataset(active_id, alerts)

    return updated


@app.get("/api/reports/forecast-csv")
def download_forecast_csv_report(
    horizon: str = Query("24h", description="24h, 7d, or 30d"),
) -> Response:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    clean_df, ds_meta = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    forecast_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=horizon,
        user_growth_pct=float(settings["expected_user_growth_pct"]),
    )

    c_alloc = float(settings["allocated_bandwidth_mbps"])
    tau_pct = float(settings["utilization_threshold_pct"])
    c_eff = round(c_alloc * (tau_pct / 100.0), 2)

    rows = []
    for pt in forecast_res["forecast_series"]:
        pred = pt["predicted_mbps"]
        util_pct = round((pred / c_alloc) * 100.0, 2)
        if pred >= c_alloc:
            sla_state = "CRITICAL_SATURATION"
        elif pred > c_eff:
            sla_state = "SLA_BREACH"
        else:
            sla_state = "WITHIN_SLA"

        rows.append(
            {
                "step": pt["step"],
                "timestamp": pt["timestamp"],
                "dataset_name": ds_meta["name"],
                "horizon": forecast_res["horizon"],
                "base_ml_forecast_mbps": pt["base_forecast_mbps"],
                "user_growth_pct": settings["expected_user_growth_pct"],
                "growth_adjusted_demand_mbps": pred,
                "lower_90_ci_mbps": pt["lower_bound_mbps"],
                "upper_90_ci_mbps": pt["upper_bound_mbps"],
                "seasonal_naive_baseline_mbps": pt["seasonal_baseline_mbps"],
                "allocated_bandwidth_mbps": c_alloc,
                "sla_ceiling_mbps": c_eff,
                "projected_utilization_pct": util_pct,
                "capacity_gap_mbps": round(max(0.0, pred - c_eff), 2),
                "sla_status": sla_state,
            }
        )

    out_df = pd.DataFrame(rows)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    filename = f"NetCapacity_Forecast_{forecast_res['horizon']}_Dataset{active_id}.csv"

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reports/capacity-csv")
def download_capacity_audit_csv_report(
    horizon: str = Query("24h", description="24h, 7d, or 30d"),
) -> Response:
    ensure_initialized()
    settings = get_settings()
    active_id = int(settings["active_dataset_id"])
    clean_df, ds_meta = _get_or_load_clean_df(active_id)
    artifact = _get_or_train_artifact(active_id)

    forecast_res = generate_multi_step_forecast(
        df=clean_df,
        artifact=artifact,
        horizon=horizon,
        user_growth_pct=float(settings["expected_user_growth_pct"]),
    )
    cap_plan = evaluate_capacity_plan(
        df=clean_df,
        forecast_result=forecast_res,
        allocated_bandwidth_mbps=float(settings["allocated_bandwidth_mbps"]),
        utilization_threshold_pct=float(settings["utilization_threshold_pct"]),
        expected_user_growth_pct=float(settings["expected_user_growth_pct"]),
        safety_headroom_pct=float(settings["safety_headroom_pct"]),
    )

    rf_m = artifact["metrics"]["random_forest_test"]
    pers_m = artifact["metrics"]["persistence_baseline"]
    seas_m = artifact["metrics"]["seasonal_naive_baseline"]
    gap = cap_plan["capacity_gap_analysis"]
    rec_tier = cap_plan["recommended_upgrade_tier"]

    summary_rows = [
        {"section": "ORGANIZATION", "metric": "Organization / Node", "value": settings["organization_name"], "unit": ""},
        {"section": "DATASET", "metric": "Active Dataset", "value": ds_meta["name"], "unit": ""},
        {"section": "DATASET", "metric": "Sampling Frequency", "value": ds_meta["frequency"], "unit": ""},
        {"section": "DATASET", "metric": "Clean Observations", "value": ds_meta["row_count"], "unit": "rows"},
        {"section": "DATASET", "metric": "Synthetic Dataset Flag", "value": str(ds_meta["is_synthetic"]), "unit": "boolean"},
        {"section": "MODEL_EVAL", "metric": "ML Engine", "value": artifact["engine_used"], "unit": ""},
        {"section": "MODEL_EVAL", "metric": "Random Forest Test MAE", "value": rf_m["mae"], "unit": "Mbps"},
        {"section": "MODEL_EVAL", "metric": "Random Forest Test RMSE", "value": rf_m["rmse"], "unit": "Mbps"},
        {"section": "MODEL_EVAL", "metric": "Random Forest Test R2", "value": rf_m["r2"], "unit": "score"},
        {"section": "MODEL_EVAL", "metric": "Persistence Baseline RMSE", "value": pers_m["rmse"], "unit": "Mbps"},
        {"section": "MODEL_EVAL", "metric": "Seasonal-Naive Baseline RMSE", "value": seas_m["rmse"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "Forecast Horizon", "value": forecast_res["horizon"], "unit": ""},
        {"section": "CAPACITY_PLAN", "metric": "Allocated Link Bandwidth", "value": cap_plan["parameters"]["allocated_bandwidth_mbps"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "SLA Utilization Threshold", "value": cap_plan["parameters"]["utilization_threshold_pct"], "unit": "%"},
        {"section": "CAPACITY_PLAN", "metric": "Effective SLA Ceiling", "value": cap_plan["parameters"]["effective_sla_ceiling_mbps"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "Expected User Growth", "value": cap_plan["parameters"]["expected_user_growth_pct"], "unit": "%"},
        {"section": "CAPACITY_PLAN", "metric": "Growth-Adjusted Peak Demand", "value": cap_plan["forecast_metrics"]["growth_adjusted_peak_mbps"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "Projected Peak Utilization", "value": cap_plan["forecast_metrics"]["peak_utilization_pct"], "unit": "%"},
        {"section": "CAPACITY_PLAN", "metric": "SLA Capacity Gap", "value": gap["sla_capacity_gap_mbps"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "Required Target Link Capacity (with Headroom)", "value": gap["required_link_capacity_mbps"], "unit": "Mbps"},
        {"section": "CAPACITY_PLAN", "metric": "Capacity Status", "value": gap["status_label"], "unit": ""},
        {"section": "RECOMMENDATION", "metric": "Recommended Carrier Tier", "value": rec_tier["label"], "unit": ""},
        {"section": "RECOMMENDATION", "metric": "Optical Interface Spec", "value": rec_tier["interface_spec"], "unit": ""},
        {"section": "RECOMMENDATION", "metric": "Estimated Monthly Circuit Budget", "value": rec_tier["typical_monthly_usd"], "unit": "USD/month"},
    ]

    out_df = pd.DataFrame(summary_rows)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    filename = f"NetCapacity_Executive_Audit_{forecast_res['horizon']}.csv"

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
