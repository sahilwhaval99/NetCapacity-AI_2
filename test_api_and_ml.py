"""
Automated Test Suite for NetCapacity AI Backend & ML Pipeline
=============================================================
Compatible with both `pytest` and `python3 -m unittest discover -s tests -v`.
Tests:
1. CSV validation, timestamp sorting, duplicate deduplication, and missing-value imputation.
2. Leak-free lag & causal rolling-window feature engineering.
3. Chronological train/test split (no future leakage) and Random Forest vs. Baseline metrics.
4. Multi-step forecasting across 24h, 7d, and 30d horizons.
5. Capacity gap math, SLA ceiling calculations, and carrier tier upgrade recommendations.
6. End-to-end FastAPI REST endpoints (health, dashboard, forecast, capacity planner, CSV upload, CSV download, alerts, settings).
"""

import io
import sys
import unittest
from pathlib import Path

# Ensure backend root is on sys.path whether invoked from root or backend/
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from capacity_planner import evaluate_capacity_plan
from main import app
from ml_engine import (
    build_supervised_features,
    compute_regression_metrics,
    generate_multi_step_forecast,
    train_and_evaluate_model,
)
from preprocessing import DatasetValidationError, load_and_preprocess_csv, validate_and_preprocess_dataframe


class TestNetCapacityMLAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_preprocessing_missing_values_and_duplicates(self):
        """Verifies timestamp sorting, duplicate merging, and missing-value imputation."""
        ts = pd.date_range("2026-09-01 00:00:00", periods=60, freq="h")
        bw = np.linspace(300.0, 500.0, 60)
        df = pd.DataFrame({"timestamp": ts, "bandwidth_mbps": bw})
        df.loc[10, "bandwidth_mbps"] = np.nan
        df.loc[25, "bandwidth_mbps"] = -40.0
        dup_row = df.iloc[[5]].copy()
        df_messy = pd.concat([df, dup_row], ignore_index=True).sample(frac=1.0, random_state=42)

        clean_df, report = validate_and_preprocess_dataframe(df_messy, imputation_method="linear")
        self.assertEqual(len(clean_df), 60)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["negative_values_cleaned"], 1)
        self.assertEqual(int(clean_df["bandwidth_mbps"].isna().sum()), 0)
        self.assertTrue(clean_df["timestamp"].is_monotonic_increasing)

    def test_02_validation_rejects_invalid_csv(self):
        """Verifies that CSVs without required columns or too few rows raise DatasetValidationError."""
        bad_df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        with self.assertRaises(DatasetValidationError):
            validate_and_preprocess_dataframe(bad_df)

    def test_03_leak_free_chronological_split_and_rf_evaluation(self):
        """Verifies strictly chronological split and authentic Random Forest vs. Baseline evaluation."""
        res = self.client.get("/api/model/evaluation")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        train_end_str = data["train_window"].split(" to ")[1]
        test_start_str = data["test_window"].split(" to ")[0]
        self.assertLess(pd.to_datetime(train_end_str), pd.to_datetime(test_start_str))

        metrics = data["metrics"]
        rf_test = metrics["random_forest_test"]
        pers_base = metrics["persistence_baseline"]
        seas_base = metrics["seasonal_naive_baseline"]

        print(
            f"\n[Evaluation Metrics] RF Test -> MAE: {rf_test['mae']} Mbps, RMSE: {rf_test['rmse']} Mbps, R²: {rf_test['r2']} | "
            f"Persistence RMSE: {pers_base['rmse']} Mbps | Seasonal-Naive RMSE: {seas_base['rmse']} Mbps"
        )

        self.assertGreater(rf_test["mae"], 0.0)
        self.assertGreater(rf_test["rmse"], rf_test["mae"])
        self.assertGreater(rf_test["r2"], 0.70)
        self.assertLess(rf_test["rmse"], pers_base["rmse"])
        self.assertGreater(len(data["feature_importances"]), 5)

    def test_04_multi_horizon_forecasts_24h_7d_30d(self):
        """Verifies interactive 24-hour, 7-day, and 30-day forecast horizons."""
        for horizon, expected_steps in [("24h", 24), ("7d", 168), ("30d", 720)]:
            resp = self.client.get(f"/api/forecast?horizon={horizon}&user_growth_pct=15")
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()["forecast"]
            self.assertEqual(payload["horizon"], horizon)
            self.assertEqual(payload["steps"], expected_steps)
            self.assertEqual(len(payload["forecast_series"]), expected_steps)
            first_pt = payload["forecast_series"][0]
            self.assertLess(first_pt["lower_bound_mbps"], first_pt["predicted_mbps"])
            self.assertGreater(first_pt["upper_bound_mbps"], first_pt["predicted_mbps"])
            self.assertAlmostEqual(
                first_pt["predicted_mbps"],
                round(first_pt["base_forecast_mbps"] * 1.15, 2),
                delta=0.05,
            )

    def test_05_capacity_planner_calculations_and_recommendations(self):
        """Verifies exact SLA ceiling, capacity gap, and upgrade tier recommendations."""
        resp_shortage = self.client.post(
            "/api/capacity/plan",
            json={
                "allocated_bandwidth_mbps": 600.0,
                "utilization_threshold_pct": 80.0,
                "expected_user_growth_pct": 25.0,
                "safety_headroom_pct": 15.0,
                "horizon": "24h",
                "persist_settings": False,
            },
        )
        self.assertEqual(resp_shortage.status_code, 200)
        plan = resp_shortage.json()["capacity_plan"]
        self.assertEqual(plan["parameters"]["effective_sla_ceiling_mbps"], 480.0)
        gap = plan["capacity_gap_analysis"]
        self.assertIn(gap["status"], ("CRITICAL_SHORTAGE", "SLA_THRESHOLD_BREACH"))
        self.assertGreater(gap["sla_capacity_gap_mbps"], 0.0)
        self.assertGreater(plan["recommended_upgrade_tier"]["tier_mbps"], 600.0)

        resp_healthy = self.client.post(
            "/api/capacity/plan",
            json={
                "allocated_bandwidth_mbps": 2500.0,
                "utilization_threshold_pct": 80.0,
                "expected_user_growth_pct": 5.0,
                "safety_headroom_pct": 15.0,
                "horizon": "24h",
                "persist_settings": False,
            },
        )
        self.assertEqual(resp_healthy.status_code, 200)
        gap_healthy = resp_healthy.json()["capacity_plan"]["capacity_gap_analysis"]
        self.assertEqual(gap_healthy["sla_capacity_gap_mbps"], 0.0)

    def test_06_csv_upload_and_download_reports(self):
        """Verifies uploading a custom CSV dataset and downloading CSV forecast/capacity reports."""
        ts = pd.date_range("2026-10-01 00:00:00", periods=120, freq="h")
        rng = np.random.default_rng(99)
        bw = 500.0 + 120.0 * np.sin(2 * np.pi * ts.hour / 24.0) + rng.normal(0, 15, size=120)
        upload_df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "bandwidth_mbps": np.round(bw, 2)})
        csv_bytes = upload_df.to_csv(index=False).encode("utf-8")

        resp = self.client.post(
            "/api/datasets/upload",
            files={"file": ("custom_branch_office.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={
                "name": "Custom Branch Office Telemetry",
                "description": "Uploaded via automated test suite",
                "imputation_method": "linear",
                "activate_immediately": "false",
            },
        )
        self.assertEqual(resp.status_code, 200)
        upload_res = resp.json()
        self.assertEqual(upload_res["preprocessing_report"]["clean_rows"], 120)

        # Clean up test dataset record so default state remains pristine
        uploaded_id = upload_res["dataset_id"]
        del_resp = self.client.delete(f"/api/datasets/{uploaded_id}")
        self.assertEqual(del_resp.status_code, 200)

        f_csv = self.client.get("/api/reports/forecast-csv?horizon=24h")
        self.assertEqual(f_csv.status_code, 200)
        self.assertIn("text/csv", f_csv.headers["content-type"])
        self.assertIn("growth_adjusted_demand_mbps", f_csv.text)

        c_csv = self.client.get("/api/reports/capacity-csv?horizon=24h")
        self.assertEqual(c_csv.status_code, 200)
        self.assertIn("Recommended Carrier Tier", c_csv.text)

    def test_07_dashboard_and_settings_persistence(self):
        """Verifies dashboard overview payload and persistent SQLite settings."""
        dash = self.client.get("/api/dashboard?horizon=24h")
        self.assertEqual(dash.status_code, 200)
        d_json = dash.json()
        self.assertIn("capacity_plan", d_json)
        self.assertIn("forecast", d_json)
        self.assertIn("model_metrics", d_json)
        self.assertGreater(len(d_json["telemetry_series"]), 20)

        # Update setting and verify persistence
        put_res = self.client.put("/api/settings", json={"expected_user_growth_pct": 14.0})
        self.assertEqual(put_res.status_code, 200)
        self.assertEqual(put_res.json()["expected_user_growth_pct"], 14.0)

        # Restore default 12.0%
        self.client.put("/api/settings", json={"expected_user_growth_pct": 12.0})


if __name__ == "__main__":
    unittest.main()
