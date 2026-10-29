"""
Intelligent Network Capacity Planning & Infrastructure Upgrade Engine
=====================================================================
Analyzes historical telemetry and ML-forecasted bandwidth demand against:
- Allocated Link Bandwidth Capacity (C_alloc in Mbps)
- Safe SLA Utilization Threshold (tau_thresh in %)
- Expected User / Traffic Growth Rate (g_user in %)
- Safety Burst Headroom Buffer (h_safety in %)

Computes exact capacity gaps, SLA breach windows, congestion risk scores,
actionable alerts, and standard IEEE/Telco optical carrier upgrade recommendations.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Standard Enterprise / ISP Carrier Optical & Ethernet Link Tiers
CARRIER_TIERS = [
    {
        "tier_mbps": 500.0,
        "label": "500 Mbps Metro Ethernet (Sub-Rate 1G)",
        "interface_spec": "1000BASE-LX SFP (Committed Information Rate 500M)",
        "typical_monthly_usd": 450,
    },
    {
        "tier_mbps": 1000.0,
        "label": "1 Gbps Dedicated Fiber Link (1GBASE-LX)",
        "interface_spec": "SFP 1000BASE-LX/LH Single-Mode Fiber (1310nm)",
        "typical_monthly_usd": 850,
    },
    {
        "tier_mbps": 2500.0,
        "label": "2.5 Gbps NBASE-T / Dual-1G LACP Trunk",
        "interface_spec": "2.5GBASE-T / 2x1G LACP Port-Channel Aggregation",
        "typical_monthly_usd": 1650,
    },
    {
        "tier_mbps": 5000.0,
        "label": "5 Gbps Multi-Gigabit Optical Trunk",
        "interface_spec": "SFP+ 10GBASE-LR (Policed 5G CIR) + Enterprise QoS",
        "typical_monthly_usd": 2900,
    },
    {
        "tier_mbps": 10000.0,
        "label": "10 Gbps Carrier-Grade Optical Uplink (10GbE)",
        "interface_spec": "SFP+ 10GBASE-LR / DWDM 10GbE Core Router Uplink",
        "typical_monthly_usd": 4800,
    },
    {
        "tier_mbps": 25000.0,
        "label": "25 Gbps High-Density Spine/WAN Uplink (25GbE)",
        "interface_spec": "SFP28 25GBASE-LR Redundant Dual-Homed BGP Link",
        "typical_monthly_usd": 9200,
    },
    {
        "tier_mbps": 40000.0,
        "label": "40 Gbps QSFP+ Core Backbone Trunk (40GbE)",
        "interface_spec": "QSFP+ 40GBASE-LR4 Optical Core Interconnect",
        "typical_monthly_usd": 14000,
    },
    {
        "tier_mbps": 100000.0,
        "label": "100 Gbps Coherent DWDM Terabit-Ready Core (100GbE)",
        "interface_spec": "QSFP28 100GBASE-LR4 / ZR4 Carrier Core Transport",
        "typical_monthly_usd": 26500,
    },
]


def evaluate_capacity_plan(
    df: pd.DataFrame,
    forecast_result: Dict[str, Any],
    allocated_bandwidth_mbps: float = 1000.0,
    utilization_threshold_pct: float = 80.0,
    expected_user_growth_pct: float = 12.0,
    safety_headroom_pct: float = 15.0,
) -> Dict[str, Any]:
    """
    Calculates historical and forecasted utilization, SLA threshold breaches,
    capacity gap, and infrastructure upgrade recommendations.
    """
    c_alloc = max(10.0, float(allocated_bandwidth_mbps))
    tau_pct = float(np.clip(utilization_threshold_pct, 40.0, 99.0))
    tau_ratio = tau_pct / 100.0
    headroom_pct = float(np.clip(safety_headroom_pct, 0.0, 100.0))
    headroom_ratio = headroom_pct / 100.0

    # Effective SLA safe ceiling in Mbps: C_eff = C_alloc * (tau_pct / 100)
    effective_ceiling_mbps = round(c_alloc * tau_ratio, 2)

    # 1. Historical utilization statistics
    bw_hist = df["bandwidth_mbps"].to_numpy(dtype=np.float64)
    hist_peak_mbps = round(float(np.max(bw_hist)), 2)
    hist_avg_mbps = round(float(np.mean(bw_hist)), 2)
    hist_p95_mbps = round(float(np.percentile(bw_hist, 95)), 2)
    hist_current_mbps = round(float(bw_hist[-1]), 2)

    hist_peak_util_pct = round((hist_peak_mbps / c_alloc) * 100.0, 2)
    hist_avg_util_pct = round((hist_avg_mbps / c_alloc) * 100.0, 2)
    hist_current_util_pct = round((hist_current_mbps / c_alloc) * 100.0, 2)

    # 2. Forecasted demand statistics (already incorporates expected_user_growth_pct)
    f_series = forecast_result["forecast_series"]
    pred_vals = np.array([pt["predicted_mbps"] for pt in f_series], dtype=np.float64)
    base_vals = np.array([pt["base_forecast_mbps"] for pt in f_series], dtype=np.float64)
    upper_vals = np.array([pt["upper_bound_mbps"] for pt in f_series], dtype=np.float64)

    pred_peak_mbps = round(float(np.max(pred_vals)), 2)
    pred_base_peak_mbps = round(float(np.max(base_vals)), 2)
    pred_avg_mbps = round(float(np.mean(pred_vals)), 2)
    pred_p95_mbps = round(float(np.percentile(pred_vals, 95)), 2)
    pred_upper90_peak_mbps = round(float(np.max(upper_vals)), 2)

    pred_peak_util_pct = round((pred_peak_mbps / c_alloc) * 100.0, 2)
    pred_p95_util_pct = round((pred_p95_mbps / c_alloc) * 100.0, 2)

    # 3. Capacity Gap & Required Target Bandwidth
    # Gap relative to SLA safe threshold ceiling: max(0, D_peak - C_eff)
    sla_capacity_gap_mbps = round(max(0.0, pred_peak_mbps - effective_ceiling_mbps), 2)
    # Gap relative to raw physical link saturation: max(0, D_peak - C_alloc)
    physical_link_deficit_mbps = round(max(0.0, pred_peak_mbps - c_alloc), 2)
    # Available headroom before hitting SLA threshold
    available_sla_margin_mbps = round(effective_ceiling_mbps - pred_peak_mbps, 2)

    # Target link capacity required so that (D_peak * (1 + h_safety)) <= C_target * tau_ratio
    required_link_capacity_mbps = round(
        (pred_peak_mbps * (1.0 + headroom_ratio)) / tau_ratio, 2
    )
    additional_bandwidth_needed_mbps = round(
        max(0.0, required_link_capacity_mbps - c_alloc), 2
    )

    # 4. Breach window detection across forecast steps
    sla_breach_points = [pt for pt in f_series if pt["predicted_mbps"] > effective_ceiling_mbps]
    link_saturation_points = [pt for pt in f_series if pt["predicted_mbps"] >= c_alloc]
    upper_bound_breach_points = [pt for pt in f_series if pt["upper_bound_mbps"] > effective_ceiling_mbps]

    total_steps = len(f_series)
    sla_breach_count = len(sla_breach_points)
    saturation_count = len(link_saturation_points)
    breach_probability_pct = round((len(upper_bound_breach_points) / max(1, total_steps)) * 100.0, 1)

    first_sla_breach_ts = sla_breach_points[0]["timestamp"] if sla_breach_points else None
    first_saturation_ts = link_saturation_points[0]["timestamp"] if link_saturation_points else None

    # 5. Determine Risk Status & Recommend Carrier Tier
    if pred_peak_mbps >= c_alloc or saturation_count > 0:
        status = "CRITICAL_SHORTAGE"
        status_label = "Critical Link Saturation Projected"
        severity = "CRITICAL"
    elif sla_breach_count > 0 or pred_peak_mbps > effective_ceiling_mbps:
        status = "SLA_THRESHOLD_BREACH"
        status_label = "SLA Utilization Threshold Exceeded"
        severity = "WARNING"
    elif pred_upper90_peak_mbps > effective_ceiling_mbps:
        status = "NEAR_CAPACITY_WATCH"
        status_label = "90% Burst Interval Approaches Threshold"
        severity = "MODERATE"
    elif pred_peak_util_pct < 30.0:
        status = "OVERPROVISIONED"
        status_label = "Link Significantly Under-Utilized"
        severity = "INFO"
    else:
        status = "HEALTHY_OPTIMAL"
        status_label = "Capacity Within Safe SLA Envelope"
        severity = "OPTIMAL"

    # Select recommended carrier tier
    recommended_tier = CARRIER_TIERS[-1]
    for tier in CARRIER_TIERS:
        if tier["tier_mbps"] >= required_link_capacity_mbps:
            recommended_tier = tier
            break

    # Actionable Engineering Recommendations
    recommendations: List[Dict[str, str]] = []
    if status == "CRITICAL_SHORTAGE":
        recommendations.append(
            {
                "priority": "P1 - Immediate Hardware/Circuit Upgrade",
                "category": "Physical Link Expansion",
                "title": f"Upgrade WAN/Core Link from {c_alloc:g} Mbps to {recommended_tier['label']}",
                "detail": (
                    f"Forecasted peak demand ({pred_peak_mbps} Mbps) exceeds physical link capacity ({c_alloc:g} Mbps) "
                    f"across {saturation_count} forecasted intervals (first saturation at {first_saturation_ts}). "
                    f"Provision {recommended_tier['interface_spec']} to guarantee {tau_pct:g}% SLA utilization with {headroom_pct:g}% burst headroom."
                ),
            }
        )
        recommendations.append(
            {
                "priority": "P1 - Immediate Traffic Engineering",
                "category": "QoS & Traffic Shaping",
                "title": "Activate Hierarchical QoS (HQoS) & DSCP Priority Queuing",
                "detail": (
                    "Enforce strict priority queuing (EF) for VoIP/real-time telemetry and apply WRED / token-bucket "
                    "rate-limiting on bulk cloud backups and software updates during peak hours (09:00–18:00 & 20:00–23:00)."
                ),
            }
        )
    elif status == "SLA_THRESHOLD_BREACH":
        recommendations.append(
            {
                "priority": "P2 - Planned Capacity Upgrade",
                "category": "Circuit Provisioning",
                "title": f"Expand Link Capacity to {recommended_tier['label']} (+{additional_bandwidth_needed_mbps:g} Mbps Required)",
                "detail": (
                    f"Forecasted peak demand ({pred_peak_mbps} Mbps, {pred_peak_util_pct}% utilization) breaches your "
                    f"{tau_pct:g}% SLA threshold ({effective_ceiling_mbps} Mbps) by {sla_capacity_gap_mbps} Mbps across "
                    f"{sla_breach_count} intervals (starting {first_sla_breach_ts})."
                ),
            }
        )
        recommendations.append(
            {
                "priority": "P2 - Off-Peak Scheduling",
                "category": "Workload Shifting",
                "title": "Reschedule Non-Critical Synchronization & Backup Windows",
                "detail": (
                    "Shift bulk replication, OS patch caching, and offsite snapshot jobs to the 01:00–05:00 trough window "
                    "to shave ~12–18% off peak daytime utilization prior to circuit upgrade completion."
                ),
            }
        )
    elif status == "NEAR_CAPACITY_WATCH":
        recommendations.append(
            {
                "priority": "P3 - Proactive Procurement Watch",
                "category": "Capacity Planning",
                "title": f"Initiate Lead-Time Quote for {recommended_tier['label']}",
                "detail": (
                    f"Mean forecasted peak ({pred_peak_mbps} Mbps) remains inside the {effective_ceiling_mbps} Mbps SLA ceiling, "
                    f"but the 90% upper burst envelope ({pred_upper90_peak_mbps} Mbps) exceeds the threshold. "
                    "Enable burst-burstable billing or LACP port-channel standby capacity."
                ),
            }
        )
    elif status == "OVERPROVISIONED":
        recommendations.append(
            {
                "priority": "P3 - Cost Optimization",
                "category": "Right-Sizing & FinOps",
                "title": f"Consider Right-Sizing Committed Information Rate (CIR) to {recommended_tier['label']}",
                "detail": (
                    f"Peak forecasted utilization is only {pred_peak_util_pct}% ({pred_peak_mbps} Mbps on a {c_alloc:g} Mbps link). "
                    f"Right-sizing to {recommended_tier['tier_mbps']:g} Mbps maintains full {headroom_pct:g}% burst headroom while reducing recurring circuit OpEx."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "priority": "Routine - Optimal Operation",
                "category": "SLA Compliance",
                "title": f"Maintain Current {c_alloc:g} Mbps Allocation (Healthy Margin: +{available_sla_margin_mbps} Mbps)",
                "detail": (
                    f"Forecasted peak demand ({pred_peak_mbps} Mbps, {pred_peak_util_pct}% of link) stays safely below the "
                    f"{effective_ceiling_mbps} Mbps SLA threshold with zero projected breaches."
                ),
            }
        )

    return {
        "parameters": {
            "allocated_bandwidth_mbps": c_alloc,
            "utilization_threshold_pct": tau_pct,
            "effective_sla_ceiling_mbps": effective_ceiling_mbps,
            "expected_user_growth_pct": round(float(expected_user_growth_pct), 2),
            "safety_headroom_pct": headroom_pct,
            "horizon": forecast_result["horizon"],
        },
        "historical_metrics": {
            "current_mbps": hist_current_mbps,
            "current_utilization_pct": hist_current_util_pct,
            "peak_mbps": hist_peak_mbps,
            "peak_utilization_pct": hist_peak_util_pct,
            "avg_mbps": hist_avg_mbps,
            "avg_utilization_pct": hist_avg_util_pct,
            "p95_mbps": hist_p95_mbps,
        },
        "forecast_metrics": {
            "base_peak_mbps": pred_base_peak_mbps,
            "growth_adjusted_peak_mbps": pred_peak_mbps,
            "growth_adjusted_avg_mbps": pred_avg_mbps,
            "growth_adjusted_p95_mbps": pred_p95_mbps,
            "upper_90_peak_mbps": pred_upper90_peak_mbps,
            "peak_utilization_pct": pred_peak_util_pct,
            "p95_utilization_pct": pred_p95_util_pct,
            "peak_timestamp": forecast_result["summary"]["peak_timestamp"],
        },
        "capacity_gap_analysis": {
            "status": status,
            "status_label": status_label,
            "severity": severity,
            "sla_capacity_gap_mbps": sla_capacity_gap_mbps,
            "physical_link_deficit_mbps": physical_link_deficit_mbps,
            "available_sla_margin_mbps": available_sla_margin_mbps,
            "required_link_capacity_mbps": required_link_capacity_mbps,
            "additional_bandwidth_needed_mbps": additional_bandwidth_needed_mbps,
            "sla_breach_intervals_count": sla_breach_count,
            "physical_saturation_intervals_count": saturation_count,
            "total_forecast_intervals": total_steps,
            "breach_probability_pct": breach_probability_pct,
            "first_sla_breach_timestamp": first_sla_breach_ts,
            "first_saturation_timestamp": first_saturation_ts,
        },
        "recommended_upgrade_tier": recommended_tier,
        "all_carrier_tiers": CARRIER_TIERS,
        "recommendations": recommendations,
    }


def generate_system_alerts(
    df: pd.DataFrame,
    capacity_plan: Dict[str, Any],
    latency_threshold_ms: float = 35.0,
    packet_loss_threshold_pct: float = 0.5,
) -> List[Dict[str, Any]]:
    """Generates structured alerts from both forecasted capacity gaps and recent telemetry anomalies."""
    alerts: List[Dict[str, Any]] = []
    gap = capacity_plan["capacity_gap_analysis"]
    params = capacity_plan["parameters"]
    f_metrics = capacity_plan["forecast_metrics"]

    # 1. Forecasted Physical Link Saturation Alert
    if gap["physical_saturation_intervals_count"] > 0:
        alerts.append(
            {
                "severity": "CRITICAL",
                "category": "CAPACITY_SATURATION",
                "title": "Projected Physical Link Saturation (100% Bandwidth Exhaustion)",
                "message": (
                    f"Forecasted peak traffic ({f_metrics['growth_adjusted_peak_mbps']} Mbps) exceeds physical link "
                    f"capacity ({params['allocated_bandwidth_mbps']:g} Mbps) by {gap['physical_link_deficit_mbps']} Mbps "
                    f"across {gap['physical_saturation_intervals_count']} intervals (first breach: {gap['first_saturation_timestamp']})."
                ),
                "metric_value": f_metrics["growth_adjusted_peak_mbps"],
                "threshold_value": params["allocated_bandwidth_mbps"],
                "event_timestamp": str(gap["first_saturation_timestamp"]),
            }
        )

    # 2. Forecasted SLA Threshold Breach Alert
    if gap["sla_breach_intervals_count"] > 0:
        alerts.append(
            {
                "severity": "WARNING" if gap["physical_saturation_intervals_count"] == 0 else "CRITICAL",
                "category": "SLA_THRESHOLD_BREACH",
                "title": f"Forecast Breaches {params['utilization_threshold_pct']:g}% SLA Utilization Ceiling",
                "message": (
                    f"Projected peak demand reaches {f_metrics['peak_utilization_pct']}% utilization "
                    f"({f_metrics['growth_adjusted_peak_mbps']} Mbps vs. {params['effective_sla_ceiling_mbps']} Mbps SLA limit), "
                    f"creating a {gap['sla_capacity_gap_mbps']} Mbps capacity gap across {gap['sla_breach_intervals_count']} intervals."
                ),
                "metric_value": f_metrics["growth_adjusted_peak_mbps"],
                "threshold_value": params["effective_sla_ceiling_mbps"],
                "event_timestamp": str(gap["first_sla_breach_timestamp"]),
            }
        )

    # 3. 90% Confidence Envelope Warning
    if gap["sla_breach_intervals_count"] == 0 and f_metrics["upper_90_peak_mbps"] > params["effective_sla_ceiling_mbps"]:
        alerts.append(
            {
                "severity": "INFO",
                "category": "BURST_HEADROOM_RISK",
                "title": "90% Upper Confidence Burst Envelope Exceeds SLA Ceiling",
                "message": (
                    f"While mean forecast ({f_metrics['growth_adjusted_peak_mbps']} Mbps) is below the SLA ceiling, "
                    f"the 90% upper prediction bound ({f_metrics['upper_90_peak_mbps']} Mbps) exceeds {params['effective_sla_ceiling_mbps']} Mbps."
                ),
                "metric_value": f_metrics["upper_90_peak_mbps"],
                "threshold_value": params["effective_sla_ceiling_mbps"],
                "event_timestamp": str(f_metrics["peak_timestamp"]),
            }
        )

    # 4. Historical Telemetry Latency & Packet Loss Anomaly Checks (last 48 rows)
    recent = df.tail(48)
    max_lat_idx = recent["latency_ms"].idxmax()
    max_lat_row = recent.loc[max_lat_idx]
    if float(max_lat_row["latency_ms"]) >= latency_threshold_ms:
        alerts.append(
            {
                "severity": "WARNING",
                "category": "LATENCY_DEGRADATION",
                "title": "Recent Queuing Delay / Latency Spike Detected",
                "message": (
                    f"Observed round-trip latency reached {float(max_lat_row['latency_ms']):.1f} ms "
                    f"(threshold: {latency_threshold_ms:.1f} ms) during high traffic load ({float(max_lat_row['bandwidth_mbps']):.1f} Mbps)."
                ),
                "metric_value": round(float(max_lat_row["latency_ms"]), 2),
                "threshold_value": latency_threshold_ms,
                "event_timestamp": pd.to_datetime(max_lat_row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            }
        )

    max_loss_idx = recent["packet_loss_pct"].idxmax()
    max_loss_row = recent.loc[max_loss_idx]
    if float(max_loss_row["packet_loss_pct"]) >= packet_loss_threshold_pct:
        alerts.append(
            {
                "severity": "WARNING",
                "category": "PACKET_LOSS_ANOMALY",
                "title": "Elevated Interface Packet Drop Rate Observed",
                "message": (
                    f"Packet loss reached {float(max_loss_row['packet_loss_pct']):.2f}% "
                    f"(SLA ceiling: {packet_loss_threshold_pct:.2f}%) at {pd.to_datetime(max_loss_row['timestamp']).strftime('%Y-%m-%d %H:%M')}."
                ),
                "metric_value": round(float(max_loss_row["packet_loss_pct"]), 3),
                "threshold_value": packet_loss_threshold_pct,
                "event_timestamp": pd.to_datetime(max_loss_row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            }
        )

    return alerts
